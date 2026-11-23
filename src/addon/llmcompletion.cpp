#include "../backends/step_runner.hpp"
#include "../core/rwkv8_engine.hpp"
#include "../core/text_ops.hpp"
#include "../core/unicode_codec.hpp"

#include <fcitx-utils/event.h>
#include <fcitx-utils/eventloopinterface.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/action.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/statusarea.h>
#include <fcitx/surroundingtext.h>
#include <fcitx/text.h>
#include <fcitx/userinterfacemanager.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fcitx {

namespace {

using namespace fcitx5_llm;

constexpr size_t kMaxRingBytes = 6144;               // ~2048 CJK chars
constexpr size_t kMaxUnackedBytes = 1024;            // unabsorbed commits awaiting SurroundingText sync
constexpr uint64_t kSelfCommitResetGuardUs = 250000; // 250ms protection against client Reset echo
constexpr uint64_t kUnackedStaleUs = 800000;         // 800ms max age for unacked commit merge

std::string getEnvTrimmed(const char *name) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        return {};
    }
    return trimWhitespace(v);
}

struct RuntimePaths {
    std::filesystem::path modelDir;
    std::filesystem::path cacheDir;
    std::filesystem::path blacklistFile;
    std::filesystem::path logFile;
    std::string preferredDevice = "NPU";
};

RuntimePaths resolveRuntimePaths() {
    RuntimePaths p;
    std::string dev = getEnvTrimmed("FCITX5_LLM_NPU_DEVICE");
    if (!dev.empty()) {
        p.preferredDevice = std::move(dev);
    }

#ifdef __ANDROID__
    std::filesystem::path root =
        "/storage/emulated/0/Android/data/org.fcitx.fcitx5.android/files/data/llm-completion";
    p.modelDir = root / "models/unicode-37m-rwkv8";
    p.cacheDir = root / "cache";
    p.blacklistFile = root / "blacklist.txt";
    p.logFile = root / "predictions.jsonl";
#else
    std::string home = getEnvTrimmed("HOME");
    if (home.empty()) {
        home = "/tmp";
    }
    std::string xdgData = getEnvTrimmed("XDG_DATA_HOME");
    if (xdgData.empty()) {
        xdgData = home + "/.local/share";
    }
    std::string xdgCache = getEnvTrimmed("XDG_CACHE_HOME");
    if (xdgCache.empty()) {
        xdgCache = home + "/.cache";
    }
    std::string xdgCfg = getEnvTrimmed("XDG_CONFIG_HOME");
    if (xdgCfg.empty()) {
        xdgCfg = home + "/.config";
    }
    std::string xdgState = getEnvTrimmed("XDG_STATE_HOME");
    if (xdgState.empty()) {
        xdgState = home + "/.local/state";
    }

    std::string customModel = getEnvTrimmed("FCITX5_LLM_MODEL_DIR");
    p.modelDir = customModel.empty()
                     ? std::filesystem::path(xdgData) /
                           "fcitx5-llm-completion/models/unicode-37m-rwkv8"
                     : std::filesystem::path(customModel);
    p.cacheDir =
        std::filesystem::path(xdgCache) / "fcitx5-llm-completion/ov_cache";
    p.blacklistFile =
        std::filesystem::path(xdgCfg) / "fcitx5-llm-completion/blacklist.txt";
    if (getEnvTrimmed("FCITX5_LLM_NO_LOG") != "1") {
        p.logFile = std::filesystem::path(xdgState) /
                    "fcitx5-llm-completion/predictions.jsonl";
    }
#endif
    return p;
}

void trimUtf8Buffer(std::string &s, size_t maxBytes) {
    if (s.size() <= maxBytes) {
        return;
    }
    size_t cut = s.size() - maxBytes;
    while (cut < s.size() &&
           (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        ++cut;
    }
    s.erase(0, cut);
}

#ifndef __ANDROID__
Text makeProposalAuxText(const std::string &text) {
    return Text("⇥ " + text);
}
#endif

size_t firstTokenByteLen(const std::string &s) {
    if (s.empty()) {
        return 0;
    }
    unsigned char c = static_cast<unsigned char>(s[0]);
    size_t charLen = 1;
    if ((c & 0xE0) == 0xC0) {
        charLen = 2;
    } else if ((c & 0xF0) == 0xE0) {
        charLen = 3;
    } else if ((c & 0xF8) == 0xF0) {
        charLen = 4;
    }
    return std::min(s.size(), charLen);
}

void sliceAfterPrefixBytes(const std::vector<std::string> &inChunks,
                           size_t skipBytes,
                           std::vector<std::string> &outChunks) {
    outChunks.clear();
    size_t rem = skipBytes;
    for (const auto &c : inChunks) {
        if (rem == 0) {
            outChunks.push_back(c);
        } else if (c.size() <= rem) {
            rem -= c.size();
        } else {
            outChunks.push_back(c.substr(rem));
            rem = 0;
        }
    }
}

struct ContextState {
    std::string fallbackRing;
    std::string unackedCommits;
    std::string lastSeenStPrefix;
    std::string lastSeenStSuffix;
    bool hasSeenSt = false;
    uint64_t lastCommitUs = 0;
    uint64_t lastBackspaceUs = 0;
};

struct QueuedEventItem {
    TapeLogRecord logRec;
    InteractionEvent stateEv;
    bool feedState = false;
};

struct PendingSuggestTask {
    uint64_t reqId = 0;
    std::string program;
    std::string trigger;
    bool surroundingValid = false;
    int cursor = -1;
    std::string pendingCommit;
    std::string prefix;
    std::string suffix;
};

struct CompletedProposal {
    uint64_t reqId = 0;
    std::string text;
    std::vector<std::string> chunks;
};

} // namespace

class LlmCompletion : public AddonInstance {
public:
    explicit LlmCompletion(Instance *instance)
        : instance_(instance), paths_(resolveRuntimePaths()),
          blacklist_(paths_.blacklistFile), logger_(paths_.logFile) {
        // Set up non-blocking wakeup pipe from background NPU thread -> Fcitx5 main EventLoop
        if (::pipe2(wakeupPipe_, O_NONBLOCK | O_CLOEXEC) == 0) {
            wakeupIoEvent_ = instance_->eventLoop().addIOEvent(
                wakeupPipe_[0], IOEventFlags{IOEventFlag::In},
                [this](EventSourceIO *, int, IOEventFlags) {
                    onWorkerWakeup();
                    return true;
                });
        }

        // Launch background NPU / event-logger worker thread (zero blocking on Fcitx5 startup)
        workerThread_ = std::thread([this]() { workerLoop(); });

        // Register tray / status area toggle action.
        toggleAction_.setCheckable(true);
        toggleAction_.setIcon("edit-find-replace");
        toggleAction_.setHotkey(
            {Key(FcitxKey_i, KeyStates{KeyState::Ctrl, KeyState::Alt})});
        updateActionState(nullptr);
        toggleConn_ = toggleAction_.connect<SimpleAction::Activated>(
            [this](InputContext *ic) { toggleEnabled(ic); });
        instance_->userInterfaceManager().registerAction("llm-completion",
                                                         &toggleAction_);

        // Pre-IM: toggle hotkey, accept/dismiss active proposal, cancel inflight.
        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextKeyEvent, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto &keyEvent = static_cast<KeyEvent &>(event);
                if (keyEvent.isRelease()) {
                    return;
                }
                auto *ic = keyEvent.inputContext();

                if (keyEvent.key().check(
                        Key(FcitxKey_i, KeyStates{KeyState::Ctrl, KeyState::Alt})) ||
                    keyEvent.key().check(
                        Key(FcitxKey_I, KeyStates{KeyState::Ctrl, KeyState::Alt}))) {
                    toggleEnabled(ic);
                    keyEvent.filterAndAccept();
                    return;
                }

                if (keyEvent.key().isModifier()) {
                    return;
                }

                if (hasProposal_ && activeIC_ == ic) {
                    // Ctrl+Tab: accept one word/token chunk from the proposal.
                    if (keyEvent.key().check(Key(FcitxKey_Tab, KeyState::Ctrl))) {
                        acceptChunkCount(ic, 1);
                        keyEvent.filterAndAccept();
                        return;
                    }
                    if (keyEvent.key().check(Key(FcitxKey_Escape))) {
                        uint64_t reqID = proposalReqID_;
                        std::string rem = proposalText_;
                        int64_t shownMs = currentShownMs();
                        clearSelfCommitGuard();
                        clearProposal(ic);
                        cancelPending();
                        if (isRecordable(ic)) {
                            enqueueProposalEvent(reqID, ic->program(),
                                                 "dismiss_esc", "", rem,
                                                 shownMs);
                        }
                        keyEvent.filterAndAccept();
                        return;
                    }
                    // Any other key clears our InputPanel popup and passes through natively (zero clientPreedit).
                    uint64_t reqID = proposalReqID_;
                    std::string rem = proposalText_;
                    int64_t shownMs = currentShownMs();
                    clearSelfCommitGuard();
                    clearProposal(ic);
                    if (isRecordable(ic)) {
                        enqueueProposalEvent(reqID, ic->program(),
                                             "dismiss_key", "", rem, shownMs);
                    }
                } else if (activeIC_ == ic && inflightReqID_ != 0 &&
                           inflightReqID_ == acceptedStreamReqID_ &&
                           keyEvent.key().check(Key(FcitxKey_Tab, KeyState::Ctrl))) {
                    keyEvent.filterAndAccept();
                    return;
                }

                clearSelfCommitGuard();
                cancelPending();
            }));

        // Post-IM: preserve active proposal popup across idle KeyRelease/Modifier updates,
        // and track plain ASCII typing when main IM is idle.
        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextKeyEvent, EventWatcherPhase::PostInputMethod,
            [this](Event &event) {
                auto &keyEvent = static_cast<KeyEvent &>(event);
                auto *ic = keyEvent.inputContext();
                if ((keyEvent.isRelease() || keyEvent.key().isModifier()) &&
                    hasProposal_ && activeIC_ == ic && !proposalText_.empty() &&
                    isMainImIdle(ic) &&
                    ic->inputPanel().clientPreedit().empty()) {
#ifndef __ANDROID__
                    if (ic->inputPanel().auxUp().empty()) {
                        showProposal(ic, proposalText_, proposalChunks_,
                                     proposalReqID_);
                    }
#else
                    showProposal(ic, proposalText_, proposalChunks_,
                                 proposalReqID_);
#endif
                }
                if (keyEvent.isRelease() || keyEvent.filtered()) {
                    return;
                }
                const Key &k = keyEvent.key();

                if (k.check(Key(FcitxKey_BackSpace))) {
                    auto &ctx = contexts_[ic];
                    std::string deleted = popLastUtf8Char(ctx.fallbackRing);
                    popLastUtf8Char(ctx.unackedCommits);
                    ctx.lastBackspaceUs = now(CLOCK_MONOTONIC);
                    if (isRecordable(ic)) {
                        enqueueTypingEvent(ic->program(), "backspace", "key",
                                           deleted);
                    }
                    return;
                }
                if (k.check(Key(FcitxKey_Return)) ||
                    k.check(Key(FcitxKey_KP_Enter))) {
                    auto &ctx = contexts_[ic];
                    ctx.fallbackRing.clear();
                    ctx.unackedCommits.clear();
                    if (isRecordable(ic)) {
                        enqueueTypingEvent(ic->program(), "user_type", "return",
                                           "\n");
                    }
                    return;
                }
                if (k.check(Key(FcitxKey_Left)) ||
                    k.check(Key(FcitxKey_Right)) ||
                    k.check(Key(FcitxKey_Up)) ||
                    k.check(Key(FcitxKey_Down)) ||
                    k.check(Key(FcitxKey_Home)) ||
                    k.check(Key(FcitxKey_End)) ||
                    k.check(Key(FcitxKey_Page_Up)) ||
                    k.check(Key(FcitxKey_Page_Down)) ||
                    k.check(Key(FcitxKey_Delete))) {
                    auto &ctx = contexts_[ic];
                    ctx.fallbackRing.clear();
                    ctx.unackedCommits.clear();
                    return;
                }

                if (k.isModifier() ||
                    k.states().testAny(KeyStates{KeyState::Ctrl, KeyState::Alt,
                                                 KeyState::Super})) {
                    return;
                }

                auto chr = Key::keySymToUnicode(k.sym());
                if (chr >= 0x20 && chr != 0x7f) {
                    std::string typed = utf8::UCS4ToUTF8(chr);
                    recordCommittedText(ic, typed);
                    if (isRecordable(ic)) {
                        enqueueTypingEvent(ic->program(), "user_type", "key",
                                           typed);
                    }
                    scheduleTrigger(ic, "key");
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextCommitString,
            EventWatcherPhase::PostInputMethod, [this](Event &event) {
                if (suppressNextCommit_) {
                    return;
                }
                auto &ce = static_cast<CommitStringEvent &>(event);
                auto *ic = ce.inputContext();
                if (!ce.text().empty()) {
                    recordCommittedText(ic, ce.text());
                    if (isRecordable(ic)) {
                        enqueueTypingEvent(ic->program(), "user_type", "commit",
                                           ce.text());
                    }
                    scheduleTrigger(ic, "commit");
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextSurroundingTextUpdated,
            EventWatcherPhase::PostInputMethod, [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                syncSurroundingSnapshot(ic, true);
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextFocusIn, EventWatcherPhase::PostInputMethod,
            [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                ic->statusArea().addAction(StatusGroup::AfterInputMethod,
                                           &toggleAction_);
                updateActionState(ic);
                syncSurroundingSnapshot(ic, false);
                if (isRecordable(ic)) {
                    std::string stP, stS;
                    int cur = -1;
                    bool valid = readRawSurrounding(ic, stP, stS, cur);
                    enqueueContextAnchorEvent(ic->program(), "focus_in", valid,
                                              cur, stP, stS);
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextInputMethodActivated,
            EventWatcherPhase::PostInputMethod, [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                ic->statusArea().addAction(StatusGroup::AfterInputMethod,
                                           &toggleAction_);
                updateActionState(ic);
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextFocusOut, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                if (activeIC_ == ic) {
                    clearSelfCommitGuard();
                    clearProposal(ic);
                    cancelPending();
                    activeIC_ = nullptr;
                }
                contexts_.erase(ic);
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextReset, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                if (activeIC_ == ic && hasProposal_) {
                    int64_t shownMs = currentShownMs();
                    if (ic == selfCommitIC_ &&
                        now(CLOCK_MONOTONIC) <= selfCommitProtectUntilUs_) {
                        if (isRecordable(ic)) {
                            enqueueProposalEvent(proposalReqID_, ic->program(),
                                                 "reset_ignored", "",
                                                 proposalText_, shownMs);
                        }
                        scheduleReassertProposal(ic);
                        return;
                    }
                    if (isRecordable(ic)) {
                        enqueueProposalEvent(proposalReqID_, ic->program(),
                                             "reset_cleared", "", proposalText_,
                                             shownMs);
                    }
                    clearProposal(ic);
                }
                if (ic != selfCommitIC_ ||
                    now(CLOCK_MONOTONIC) > selfCommitProtectUntilUs_) {
                    if (auto it = contexts_.find(ic); it != contexts_.end()) {
                        it->second.unackedCommits.clear();
                    }
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextUpdateUI, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto &uiEvent = static_cast<InputContextUpdateUIEvent &>(event);
                if (uiEvent.component() == UserInterfaceComponent::InputPanel) {
                    restoreProposalUIIfWiped(uiEvent.inputContext());
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextFlushUI, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto &uiEvent = static_cast<InputContextFlushUIEvent &>(event);
                if (uiEvent.component() == UserInterfaceComponent::InputPanel) {
                    restoreProposalUIIfWiped(uiEvent.inputContext());
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextDestroyed, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic =
                    static_cast<InputContextEvent &>(event).inputContext();
                if (activeIC_ == ic) {
                    clearSelfCommitGuard();
                    cancelPending();
                    hasProposal_ = false;
                    proposalText_.clear();
                    proposalChunks_.clear();
                    proposalReqID_ = 0;
                    activeIC_ = nullptr;
                }
                contexts_.erase(ic);
            }));
    }

    ~LlmCompletion() override {
        activeReqId_.store(0, std::memory_order_relaxed);
        stopRequested_.store(true, std::memory_order_release);
        workerCv_.notify_all();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
        wakeupIoEvent_.reset();
        if (wakeupPipe_[0] >= 0) {
            ::close(wakeupPipe_[0]);
            wakeupPipe_[0] = -1;
        }
        if (wakeupPipe_[1] >= 0) {
            ::close(wakeupPipe_[1]);
            wakeupPipe_[1] = -1;
        }
    }

private:
    bool isStopRequested() const {
        return stopRequested_.load(std::memory_order_acquire);
    }

    void workerLoop() {
        std::unique_ptr<Rwkv8CompletionEngine> engine;
        std::error_code modelEc;
        if (std::filesystem::is_regular_file(
                paths_.modelDir / "model_config.json", modelEc)) {
            try {
                auto runner = createPlatformNpuRunner(
                    paths_.modelDir, paths_.cacheDir, paths_.preferredDevice);
                engine = std::make_unique<Rwkv8CompletionEngine>(
                    std::move(runner), paths_.modelDir, paths_.logFile);
                engineReady_.store(true, std::memory_order_release);
                FCITX_INFO() << "llmcompletion NPU engine ready: "
                             << engine->modelLabel()
                             << " (warmed_chars=" << engine->warmedChars()
                             << ", npu_ttt="
                             << (engine->hasHardwareOnlineAdapter() ? 1 : 0)
                             << ", online_updates=" << engine->onlineUpdates()
                             << ")";
            } catch (const std::exception &e) {
                FCITX_WARN()
                    << "llmcompletion failed to initialize NPU engine ("
                    << paths_.modelDir.string()
                    << "), falling back to collector-only mode: " << e.what();
            }
        } else {
            FCITX_INFO()
                << "llmcompletion running in collector-only mode (no model at "
                << paths_.modelDir.string() << ", log=" << paths_.logFile.string()
                << ")";
        }

        while (!isStopRequested()) {
            std::deque<QueuedEventItem> batchEvents;
            std::optional<PendingSuggestTask> nextSuggest;

            {
                std::unique_lock<std::mutex> lk(workerMutex_);
                workerCv_.wait(lk, [&] {
                    return isStopRequested() || !eventQueue_.empty() ||
                           pendingSuggest_.has_value();
                });
                if (isStopRequested()) {
                    break;
                }
                batchEvents.swap(eventQueue_);
                if (pendingSuggest_.has_value()) {
                    nextSuggest = std::move(pendingSuggest_);
                    pendingSuggest_.reset();
                }
            }

            for (const auto &item : batchEvents) {
                if (isStopRequested()) {
                    return;
                }
                if (!blacklist_.isAllowed(item.logRec.program)) {
                    continue;
                }
                logger_.log(item.logRec);
                if (engine && item.feedState) {
                    try {
                        engine->handleEvent(item.stateEv);
                    } catch (const std::exception &e) {
                        FCITX_WARN() << "llmcompletion event error: " << e.what();
                    }
                }
            }

            if (!nextSuggest.has_value()) {
                continue;
            }
            if (!engine) {
                uint64_t expected = nextSuggest->reqId;
                activeReqId_.compare_exchange_strong(expected, 0);
                continue;
            }

            const auto &task = *nextSuggest;
            if (activeReqId_.load(std::memory_order_relaxed) != task.reqId) {
                continue;
            }

            if (!blacklist_.isAllowed(task.program)) {
                TapeLogRecord skipRec;
                skipRec.event = "suggest_skipped";
                skipRec.reqId = task.reqId;
                skipRec.program = task.program;
                skipRec.reason = "blacklisted";
                logger_.log(skipRec);
                activeReqId_.compare_exchange_strong(
                    const_cast<uint64_t &>(task.reqId), 0);
                publishCompletedProposal({task.reqId, "", {}});
                continue;
            }

            std::string plannedMode =
                trimWhitespace(stripInvisibleFormattingUtf8(task.suffix)).empty()
                    ? "tail"
                    : "fim";

            TapeLogRecord startRec;
            startRec.event = "suggest_start";
            startRec.reqId = task.reqId;
            startRec.program = task.program;
            startRec.trigger = task.trigger;
            startRec.model = engine->modelLabel();
            startRec.mode = plannedMode;
            startRec.surroundingValid = task.surroundingValid;
            startRec.cursor = task.cursor;
            startRec.pendingCommit = task.pendingCommit;
            startRec.prefixLen = utf8CharCount(task.prefix);
            startRec.prefixTail = PredictionLogger::previewTail(task.prefix);
            startRec.suffixLen = utf8CharCount(task.suffix);
            startRec.suffixHead = PredictionLogger::previewHead(task.suffix);
            logger_.log(startRec);

            try {
                SuggestParams sp;
                sp.program = task.program;
                sp.prefix = task.prefix;
                sp.suffix = task.suffix;
                sp.maxTokens = 12;

                auto res = engine->suggest(sp, [&]() {
                    return isStopRequested() ||
                           activeReqId_.load(std::memory_order_relaxed) !=
                               task.reqId;
                });

                uint64_t expected = task.reqId;
                if (!res.cancelled &&
                    activeReqId_.compare_exchange_strong(expected, 0)) {
                    TapeLogRecord doneRec;
                    doneRec.event = "suggest_done";
                    doneRec.reqId = task.reqId;
                    doneRec.program = task.program;
                    doneRec.trigger = task.trigger;
                    doneRec.model = res.model;
                    doneRec.mode = res.mode;
                    doneRec.ttftMs = res.ttftMs;
                    doneRec.latencyMs = res.latencyMs;
                    doneRec.finishReason = res.finishReason;
                    if (res.rawText != res.text) {
                        doneRec.rawText = res.rawText;
                    }
                    doneRec.text = res.text;
                    doneRec.chunks = res.chunks;
                    logger_.log(doneRec);

                    publishCompletedProposal(
                        {task.reqId, std::move(res.text), std::move(res.chunks)});
                } else {
                    TapeLogRecord cancelRec;
                    cancelRec.event = "suggest_cancelled";
                    cancelRec.reqId = task.reqId;
                    cancelRec.program = task.program;
                    cancelRec.model = res.model;
                    cancelRec.mode = res.mode;
                    cancelRec.reason = "aborted_mid_stream";
                    logger_.log(cancelRec);
                }
            } catch (const std::exception &e) {
                TapeLogRecord errRec;
                errRec.event = "suggest_error";
                errRec.reqId = task.reqId;
                errRec.program = task.program;
                errRec.model = engine->modelLabel();
                errRec.mode = plannedMode;
                errRec.error = e.what();
                logger_.log(errRec);
                uint64_t expected = task.reqId;
                activeReqId_.compare_exchange_strong(expected, 0);
            }
        }
    }

    void publishCompletedProposal(CompletedProposal prop) {
        {
            std::lock_guard<std::mutex> lk(resultMutex_);
            completedQueue_.push_back(std::move(prop));
        }
        if (wakeupPipe_[1] >= 0) {
            char b = 1;
            [[maybe_unused]] ssize_t n = ::write(wakeupPipe_[1], &b, 1);
        }
    }

    void onWorkerWakeup() {
        char drain[64];
        while (wakeupPipe_[0] >= 0 &&
               ::read(wakeupPipe_[0], drain, sizeof(drain)) > 0) {
        }
        std::deque<CompletedProposal> ready;
        {
            std::lock_guard<std::mutex> lk(resultMutex_);
            ready.swap(completedQueue_);
        }
        for (auto &prop : ready) {
            handleCompletedProposal(prop);
        }
    }

    void handleCompletedProposal(CompletedProposal &prop) {
        if (prop.reqId != inflightReqID_) {
            return;
        }
        inflightReqID_ = 0;

        std::string text = std::move(prop.text);
        std::vector<std::string> chunks = std::move(prop.chunks);

        if (prop.reqId == acceptedStreamReqID_ && !acceptedStreamPrefix_.empty()) {
            if (text.size() >= acceptedStreamPrefix_.size() &&
                text.compare(0, acceptedStreamPrefix_.size(),
                             acceptedStreamPrefix_) == 0) {
                size_t skip = acceptedStreamPrefix_.size();
                text.erase(0, skip);
                std::vector<std::string> slicedChunks;
                sliceAfterPrefixBytes(chunks, skip, slicedChunks);
                chunks = std::move(slicedChunks);
            } else {
                text.clear();
                chunks.clear();
            }
            acceptedStreamReqID_ = 0;
            acceptedStreamPrefix_.clear();
            if (text.empty() && activeIC_) {
                scheduleTrigger(activeIC_, "chain_accept");
                return;
            }
        }

        if (text.empty()) {
            if (hasProposal_ && proposalReqID_ == prop.reqId && activeIC_) {
                clearProposal(activeIC_);
            }
            return;
        }
        if (activeIC_) {
            showProposal(activeIC_, text, chunks, prop.reqId);
        }
    }

    bool isRecordable(InputContext *ic) const {
        if (!enabled_ || !ic) {
            return false;
        }
        return !ic->capabilityFlags().test(CapabilityFlag::Password);
    }

    int64_t currentShownMs() const {
        if (!hasProposal_ || proposalShownAtUs_ == 0) {
            return 0;
        }
        uint64_t nowUs = now(CLOCK_MONOTONIC);
        return nowUs >= proposalShownAtUs_
                   ? static_cast<int64_t>((nowUs - proposalShownAtUs_) / 1000)
                   : 0;
    }

    bool isMainImIdle(InputContext *ic) const {
        if (!ic || !ic->inputPanel().preedit().empty()) {
            return false;
        }
#ifndef __ANDROID__
        if (auto cand = ic->inputPanel().candidateList();
            cand && !cand->empty()) {
            return false;
        }
#endif
        return true;
    }

    void restoreProposalUIIfWiped(InputContext *ic) {
        if (updatingOwnUI_ || !hasProposal_ || proposalText_.empty() ||
            ic != activeIC_ || !ic || !ic->hasFocus()) {
            return;
        }
        if (!isMainImIdle(ic) || !ic->inputPanel().clientPreedit().empty()) {
            return;
        }
#ifndef __ANDROID__
        if (ic->inputPanel().auxUp().empty()) {
            ic->inputPanel().setAuxUp(makeProposalAuxText(proposalText_));
        }
#endif
    }

    bool isEligible(InputContext *ic, bool allowOwnProposal = false) const {
        if (!enabled_ || !ic || !ic->hasFocus()) {
            return false;
        }
        auto caps = ic->capabilityFlags();
        if (caps.test(CapabilityFlag::Password)) {
            return false;
        }
        if (!ic->inputPanel().preedit().empty() ||
            !ic->inputPanel().clientPreedit().empty()) {
            return false;
        }
#ifndef __ANDROID__
        (void)allowOwnProposal;
        if (auto cand = ic->inputPanel().candidateList();
            cand && !cand->empty()) {
            return false;
        }
#else
        if (!allowOwnProposal) {
            if (auto cand = ic->inputPanel().candidateList();
                cand && !cand->empty()) {
                return false;
            }
        }
#endif
        return true;
    }

    void recordCommittedText(InputContext *ic, const std::string &text) {
        if (!ic || text.empty()) {
            return;
        }
        auto &ctx = contexts_[ic];
        ctx.fallbackRing += text;
        trimUtf8Buffer(ctx.fallbackRing, kMaxRingBytes);
        ctx.unackedCommits += text;
        trimUtf8Buffer(ctx.unackedCommits, kMaxUnackedBytes);
        ctx.lastCommitUs = now(CLOCK_MONOTONIC);
    }

    bool readRawSurrounding(InputContext *ic, std::string &stPrefix,
                            std::string &stSuffix, int &cursorOut) const {
        stPrefix.clear();
        stSuffix.clear();
        cursorOut = -1;
        if (!ic ||
            !ic->capabilityFlags().test(CapabilityFlag::SurroundingText)) {
            return false;
        }
        const auto &st = ic->surroundingText();
        if (!st.isValid()) {
            return false;
        }
        const auto &text = st.text();
        auto cursor = st.cursor();
        auto byteOff = utf8::ncharByteLength(text.begin(), cursor);
        if (byteOff < 0 || static_cast<size_t>(byteOff) > text.size()) {
            return false;
        }
        stPrefix = stripInvisibleFormattingUtf8(
            std::string_view(text).substr(0, static_cast<size_t>(byteOff)));
        stSuffix = stripInvisibleFormattingUtf8(
            std::string_view(text).substr(static_cast<size_t>(byteOff)));
        auto cleanLen = utf8::lengthValidated(stPrefix);
        cursorOut = (cleanLen != utf8::INVALID_LENGTH)
                        ? static_cast<int>(cleanLen)
                        : static_cast<int>(cursor);
        return true;
    }

    bool reconcileSurrounding(ContextState &ctx, const std::string &stPrefix,
                              const std::string &stSuffix) {
        uint64_t nowUs = now(CLOCK_MONOTONIC);
        bool externalJump = false;
        const bool inBackspaceWindow =
            (ctx.lastBackspaceUs != 0 && nowUs - ctx.lastBackspaceUs <= 250000);

        if (ctx.hasSeenSt) {
            if (stPrefix == ctx.lastSeenStPrefix &&
                stSuffix == ctx.lastSeenStSuffix) {
            } else if (stSuffix == ctx.lastSeenStSuffix &&
                       stPrefix.size() > ctx.lastSeenStPrefix.size() &&
                       stPrefix.compare(0, ctx.lastSeenStPrefix.size(),
                                        ctx.lastSeenStPrefix) == 0) {
                size_t deltaLen = stPrefix.size() - ctx.lastSeenStPrefix.size();
                std::string delta = stPrefix.substr(ctx.lastSeenStPrefix.size());
                if (!ctx.unackedCommits.empty() && ctx.lastCommitUs != 0 &&
                    nowUs - ctx.lastCommitUs <= kUnackedStaleUs &&
                    ctx.unackedCommits.size() >= deltaLen &&
                    ctx.unackedCommits.compare(0, deltaLen, delta) == 0) {
                    ctx.unackedCommits.erase(0, deltaLen);
                } else {
                    ctx.unackedCommits.clear();
                    externalJump = true;
                }
            } else if (inBackspaceWindow &&
                       ctx.lastSeenStPrefix.size() > stPrefix.size() &&
                       ctx.lastSeenStPrefix.compare(0, stPrefix.size(),
                                                    stPrefix) == 0 &&
                       (stSuffix == ctx.lastSeenStSuffix ||
                        (stSuffix.size() ==
                             (ctx.lastSeenStPrefix.size() - stPrefix.size()) +
                                 ctx.lastSeenStSuffix.size() &&
                         stSuffix.compare(
                             0, ctx.lastSeenStPrefix.size() - stPrefix.size(),
                             ctx.lastSeenStPrefix, stPrefix.size(),
                             ctx.lastSeenStPrefix.size() - stPrefix.size()) ==
                             0 &&
                         stSuffix.compare(
                             ctx.lastSeenStPrefix.size() - stPrefix.size(),
                             ctx.lastSeenStSuffix.size(),
                             ctx.lastSeenStSuffix) == 0))) {
            } else if (inBackspaceWindow && stPrefix == ctx.lastSeenStPrefix &&
                       ctx.lastSeenStSuffix.size() > stSuffix.size() &&
                       ctx.lastSeenStSuffix.compare(
                           ctx.lastSeenStSuffix.size() - stSuffix.size(),
                           stSuffix.size(), stSuffix) == 0) {
            } else {
                ctx.unackedCommits.clear();
                externalJump = true;
            }
        } else if (!ctx.unackedCommits.empty()) {
            if (ctx.lastCommitUs == 0 ||
                nowUs - ctx.lastCommitUs > kUnackedStaleUs ||
                stPrefix.ends_with(ctx.unackedCommits)) {
                ctx.unackedCommits.clear();
            }
        }

        ctx.lastSeenStPrefix = stPrefix;
        ctx.lastSeenStSuffix = stSuffix;
        ctx.hasSeenSt = true;
        return externalJump;
    }

    void syncSurroundingSnapshot(InputContext *ic, bool emitJump) {
        if (!ic) {
            return;
        }
        std::string stPrefix, stSuffix;
        int cursor = -1;
        if (readRawSurrounding(ic, stPrefix, stSuffix, cursor)) {
            bool jumped =
                reconcileSurrounding(contexts_[ic], stPrefix, stSuffix);
            if (emitJump && jumped) {
                if (hasProposal_ && activeIC_ == ic &&
                    (ic != selfCommitIC_ ||
                     now(CLOCK_MONOTONIC) > selfCommitProtectUntilUs_)) {
                    int64_t shownMs = currentShownMs();
                    if (isRecordable(ic)) {
                        enqueueProposalEvent(proposalReqID_, ic->program(),
                                             "cursor_jump_clear", "",
                                             proposalText_, shownMs);
                    }
                    clearProposal(ic);
                    cancelPending();
                }
                if (isRecordable(ic) && ic->hasFocus()) {
                    enqueueContextAnchorEvent(ic->program(), "cursor_jump",
                                              true, cursor, stPrefix, stSuffix);
                }
            }
        }
    }

    void extractContext(InputContext *ic, std::string &prefix,
                        std::string &suffix, bool &surroundingValid,
                        int &cursorOut, std::string &appliedPending) {
        prefix.clear();
        suffix.clear();
        appliedPending.clear();
        surroundingValid = false;
        cursorOut = -1;

        auto &ctx = contexts_[ic];
        std::string stPrefix, stSuffix;
        if (readRawSurrounding(ic, stPrefix, stSuffix, cursorOut)) {
            surroundingValid = true;
            reconcileSurrounding(ctx, stPrefix, stSuffix);
            prefix = stPrefix;
            suffix = stSuffix;
            if (!ctx.unackedCommits.empty()) {
                prefix += ctx.unackedCommits;
                appliedPending = ctx.unackedCommits;
            }
            if (prefix.empty() && suffix.empty() && !ctx.fallbackRing.empty()) {
                prefix = ctx.fallbackRing;
            }
        } else {
            prefix = ctx.fallbackRing;
        }
    }

    void updateActionState(InputContext *ic) {
        toggleAction_.setChecked(enabled_);
        toggleAction_.setShortText(enabled_ ? "内联补全: 开" : "内联补全: 关");
        toggleAction_.setLongText(enabled_ ? "关闭内联补全 (Ctrl+Alt+I)"
                                           : "开启内联补全 (Ctrl+Alt+I)");
        if (ic) {
            toggleAction_.update(ic);
            ic->updateUserInterface(UserInterfaceComponent::StatusArea);
        }
    }

    void toggleEnabled(InputContext *ic) {
        enabled_ = !enabled_;
        clearSelfCommitGuard();
        clearProposal(ic);
        cancelPending();
        updateActionState(ic);
        flashStatus(ic, enabled_ ? "[LLM Completion: ON]"
                                 : "[LLM Completion: OFF]");
    }

    void clearSelfCommitGuard() {
        selfCommitIC_ = nullptr;
        selfCommitProtectUntilUs_ = 0;
        reassertDefer_.reset();
    }

    void scheduleReassertProposal(InputContext *ic) {
        reassertDefer_ = instance_->eventLoop().addDeferEvent(
            [this, ic](EventSource *) {
                reassertDefer_.reset();
                if (hasProposal_ && activeIC_ == ic && !proposalText_.empty()) {
                    showProposal(ic, proposalText_, proposalChunks_,
                                 proposalReqID_);
                }
                return true;
            });
    }

    void scheduleTrigger(InputContext *ic, const char *reason) {
        if (!enabled_ || !ic || !engineReady_.load(std::memory_order_acquire)) {
            return;
        }
        activeIC_ = ic;
        pendingTriggerReason_ = reason ? reason : "unknown";
        deferTrigger_ = instance_->eventLoop().addDeferEvent(
            [this](EventSource *) {
                deferTrigger_.reset();
                triggerRequest();
                return true;
            });
    }

    void cancelPending() {
        deferTrigger_.reset();
        acceptedStreamReqID_ = 0;
        acceptedStreamPrefix_.clear();
        inflightReqID_ = 0;
        uint64_t prev = activeReqId_.exchange(0, std::memory_order_relaxed);
        if (prev != 0) {
            std::lock_guard<std::mutex> lk(workerMutex_);
            if (pendingSuggest_.has_value() &&
                pendingSuggest_->reqId == prev) {
                pendingSuggest_.reset();
            }
        }
    }

    void enqueueWorkerEvent(QueuedEventItem item) {
        {
            std::lock_guard<std::mutex> lk(workerMutex_);
            eventQueue_.push_back(std::move(item));
        }
        workerCv_.notify_one();
    }

    void enqueueProposalEvent(uint64_t reqID, const std::string &program,
                              const char *eventName,
                              const std::string &accepted,
                              const std::string &remaining, int64_t shownMs) {
        std::string ev = eventName ? eventName : "";
        QueuedEventItem q;
        q.logRec.event = ev;
        q.logRec.reqId = reqID;
        q.logRec.program = program;
        q.logRec.shownMs = shownMs;
        q.logRec.accepted = accepted;
        q.logRec.remaining = remaining;

        if (ev == "accept_chunk" || ev == "accept_word" || ev == "accept_all" ||
            ev == "dismiss_key" || ev == "dismiss_esc" ||
            ev == "dismiss_escape") {
            q.feedState = true;
            q.stateEv.event = ev;
            q.stateEv.program = program;
            q.stateEv.accepted = accepted;
            q.stateEv.remaining = remaining;
        }
        enqueueWorkerEvent(std::move(q));
    }

    void enqueueTypingEvent(const std::string &program, const char *eventName,
                            const char *trigger, const std::string &text) {
        std::string ev = eventName ? eventName : "";
        QueuedEventItem q;
        q.logRec.event = ev;
        q.logRec.program = program;
        q.logRec.trigger = trigger ? trigger : "";
        q.logRec.text = text;

        q.feedState = true;
        q.stateEv.event = ev;
        q.stateEv.program = program;
        q.stateEv.text = text;
        enqueueWorkerEvent(std::move(q));
    }

    void enqueueContextAnchorEvent(const std::string &program,
                                   const char *eventName, bool surroundingValid,
                                   int cursor, const std::string &prefix,
                                   const std::string &suffix) {
        std::string ev = eventName ? eventName : "";
        std::string pTail = utf8TailBytes(prefix, 384);
        std::string sHead = utf8HeadBytes(suffix, 384);

        QueuedEventItem q;
        q.logRec.event = ev;
        q.logRec.program = program;
        q.logRec.surroundingValid = surroundingValid;
        q.logRec.cursor = cursor;
        if (!pTail.empty()) {
            q.logRec.prefixLen = utf8CharCount(pTail);
            q.logRec.prefixTail = PredictionLogger::previewTail(pTail);
        }
        if (!sHead.empty()) {
            q.logRec.suffixLen = utf8CharCount(sHead);
            q.logRec.suffixHead = PredictionLogger::previewHead(sHead);
        }

        q.feedState = true;
        q.stateEv.event = ev;
        q.stateEv.program = program;
        q.stateEv.prefix = std::move(pTail);
        q.stateEv.suffix = std::move(sHead);
        enqueueWorkerEvent(std::move(q));
    }

    void triggerRequest() {
        if (!engineReady_.load(std::memory_order_acquire)) {
            return;
        }
        auto *ic = activeIC_;
        if (!isEligible(ic)) {
            return;
        }
        std::string prefix, suffix, pendingCommit;
        bool surroundingValid = false;
        int cursor = -1;
        extractContext(ic, prefix, suffix, surroundingValid, cursor,
                       pendingCommit);
        if (prefix.empty()) {
            return;
        }

        uint64_t reqID = ++nextReqID_;
        inflightReqID_ = reqID;
        acceptedStreamReqID_ = 0;
        acceptedStreamPrefix_.clear();
        activeReqId_.store(reqID, std::memory_order_relaxed);

        PendingSuggestTask task;
        task.reqId = reqID;
        task.program = ic->program();
        task.trigger = pendingTriggerReason_;
        task.surroundingValid = surroundingValid;
        task.cursor = cursor;
        task.pendingCommit = std::move(pendingCommit);
        task.prefix = std::move(prefix);
        task.suffix = std::move(suffix);

        {
            std::lock_guard<std::mutex> lk(workerMutex_);
            pendingSuggest_ = std::move(task);
        }
        workerCv_.notify_one();
    }

    void acceptChunkCount(InputContext *ic, size_t count) {
        if (!hasProposal_ || activeIC_ != ic || count == 0) {
            return;
        }
        deferTrigger_.reset();

        uint64_t reqID = proposalReqID_;
        bool streamStillInflight =
            (inflightReqID_ != 0 && inflightReqID_ == reqID);

        std::string accepted;
        std::string remaining;
        std::vector<std::string> remainingChunks;

        if (!proposalChunks_.empty()) {
            size_t take = std::min(count, proposalChunks_.size());
            for (size_t i = 0; i < take; ++i) {
                accepted += proposalChunks_[i];
            }
            remainingChunks.assign(proposalChunks_.begin() +
                                       static_cast<ptrdiff_t>(take),
                                   proposalChunks_.end());
            for (const auto &c : remainingChunks) {
                remaining += c;
            }
        } else {
            size_t remStart = 0;
            for (size_t i = 0; i < count && remStart < proposalText_.size();
                 ++i) {
                remStart += firstTokenByteLen(proposalText_.substr(remStart));
            }
            accepted = proposalText_.substr(0, remStart);
            remaining = proposalText_.substr(remStart);
        }

        if (accepted.empty()) {
            return;
        }

        if (streamStillInflight) {
            if (acceptedStreamReqID_ != reqID) {
                acceptedStreamReqID_ = reqID;
                acceptedStreamPrefix_.clear();
            }
            acceptedStreamPrefix_ += accepted;
        } else {
            cancelPending();
        }

        int64_t shownMs = currentShownMs();
        clearProposal(ic);

        if (!remaining.empty()) {
            selfCommitIC_ = ic;
            selfCommitProtectUntilUs_ =
                now(CLOCK_MONOTONIC) + kSelfCommitResetGuardUs;
        } else {
            clearSelfCommitGuard();
        }

        suppressNextCommit_ = true;
        ic->commitString(accepted);
        suppressNextCommit_ = false;

        recordCommittedText(ic, accepted);
        if (isRecordable(ic)) {
            enqueueProposalEvent(reqID, ic->program(), "accept_chunk", accepted,
                                 remaining, shownMs);
        }

        if (!remaining.empty()) {
            showProposal(ic, remaining, remainingChunks, reqID);
        } else if (!streamStillInflight) {
            scheduleTrigger(ic, "chain_accept");
        }
    }

#ifdef __ANDROID__
    class ProposalCandidateWord : public CandidateWord {
    public:
        ProposalCandidateWord(LlmCompletion *owner, std::string label,
                              size_t chunkCount)
            : CandidateWord(Text(std::move(label))), owner_(owner),
              chunkCount_(chunkCount) {}

        void select(InputContext *ic) const override {
            owner_->acceptChunkCount(ic, chunkCount_);
        }

    private:
        LlmCompletion *owner_;
        size_t chunkCount_;
    };
#endif

    void showProposal(InputContext *ic, const std::string &text,
                      const std::vector<std::string> &chunks, uint64_t reqID) {
        if (text.empty() || !isEligible(ic, hasProposal_ && activeIC_ == ic)) {
            return;
        }
        statusTimer_.reset();
        if (!hasProposal_ || proposalReqID_ != reqID) {
            proposalShownAtUs_ = now(CLOCK_MONOTONIC);
        }
        hasProposal_ = true;
        proposalReqID_ = reqID;
        proposalText_ = text;
        proposalChunks_ = chunks;

#ifdef __ANDROID__
        auto candList = std::make_unique<CommonCandidateList>();
        candList->setPageSize(10);
        if (!chunks.empty()) {
            for (size_t i = 0; i < chunks.size(); ++i) {
                candList->append<ProposalCandidateWord>(this, chunks[i], i + 1);
            }
            if (chunks.size() > 1) {
                candList->append<ProposalCandidateWord>(this, "✓ " + text,
                                                        chunks.size());
            }
        } else {
            candList->append<ProposalCandidateWord>(this, text, 1);
        }
        ic->inputPanel().setCandidateList(std::move(candList));
#else
        ic->inputPanel().setAuxUp(makeProposalAuxText(text));
#endif
        updatingOwnUI_ = true;
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        updatingOwnUI_ = false;
    }

    void clearProposal(InputContext *ic) {
        if (!hasProposal_) {
            return;
        }
        hasProposal_ = false;
        proposalReqID_ = 0;
        proposalShownAtUs_ = 0;
        proposalText_.clear();
        proposalChunks_.clear();
        if (ic) {
#ifdef __ANDROID__
            ic->inputPanel().setCandidateList(nullptr);
#else
            ic->inputPanel().setAuxUp(Text());
#endif
            updatingOwnUI_ = true;
            ic->updateUserInterface(UserInterfaceComponent::InputPanel);
            updatingOwnUI_ = false;
        }
    }

    void flashStatus(InputContext *ic, const std::string &msg) {
        if (!ic) {
            return;
        }
#ifdef __ANDROID__
        ic->inputPanel().setAuxDown(Text(msg));
#else
        ic->inputPanel().setAuxUp(Text(msg));
#endif
        updatingOwnUI_ = true;
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        updatingOwnUI_ = false;
        statusTimer_ = instance_->eventLoop().addTimeEvent(
            CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 1200000, 50000,
            [this, ic](EventSourceTime *, uint64_t) {
                statusTimer_.reset();
                if (!hasProposal_ && ic &&
                    contexts_.find(ic) != contexts_.end() && ic->hasFocus()) {
#ifdef __ANDROID__
                    ic->inputPanel().setAuxDown(Text());
#else
                    ic->inputPanel().setAuxUp(Text());
#endif
                    updatingOwnUI_ = true;
                    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
                    updatingOwnUI_ = false;
                }
                return true;
            });
    }

    Instance *instance_;
    RuntimePaths paths_;
    AppBlacklist blacklist_;
    PredictionLogger logger_;

    SimpleAction toggleAction_;
    ScopedConnection toggleConn_;
    std::vector<std::unique_ptr<HandlerTableEntry<EventHandler>>> handlers_;
    std::unordered_map<InputContext *, ContextState> contexts_;

    bool enabled_ = true;
    bool hasProposal_ = false;
    bool suppressNextCommit_ = false;
    bool updatingOwnUI_ = false;
    uint64_t proposalReqID_ = 0;
    uint64_t proposalShownAtUs_ = 0;
    std::string proposalText_;
    std::vector<std::string> proposalChunks_;
    InputContext *activeIC_ = nullptr;

    InputContext *selfCommitIC_ = nullptr;
    uint64_t selfCommitProtectUntilUs_ = 0;
    uint64_t acceptedStreamReqID_ = 0;
    std::string acceptedStreamPrefix_;
    std::string pendingTriggerReason_ = "unknown";

    std::unique_ptr<EventSource> deferTrigger_;
    std::unique_ptr<EventSource> reassertDefer_;
    std::unique_ptr<EventSourceTime> statusTimer_;
    std::unique_ptr<EventSourceIO> wakeupIoEvent_;
    int wakeupPipe_[2] = {-1, -1};

    uint64_t nextReqID_ = 0;
    uint64_t inflightReqID_ = 0;
    std::atomic<uint64_t> activeReqId_{0};
    std::atomic<bool> engineReady_{false};
    std::atomic<bool> stopRequested_{false};

    std::mutex workerMutex_;
    std::condition_variable workerCv_;
    std::deque<QueuedEventItem> eventQueue_;
    std::optional<PendingSuggestTask> pendingSuggest_;

    std::mutex resultMutex_;
    std::deque<CompletedProposal> completedQueue_;
    std::thread workerThread_;
};

class LlmCompletionFactory : public AddonFactory {
    AddonInstance *create(AddonManager *manager) override {
        return new LlmCompletion(manager->instance());
    }
};

} // namespace fcitx

FCITX_ADDON_FACTORY_V2_BACKWARDS(llmcompletion, fcitx::LlmCompletionFactory);
