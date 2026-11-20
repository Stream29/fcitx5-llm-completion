#include <fcitx-utils/event.h>
#include <fcitx-utils/eventloopinterface.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/utf8.h>
#include <fcitx/action.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/statusarea.h>
#include <fcitx/surroundingtext.h>
#include <fcitx/text.h>
#include <fcitx/userinterfacemanager.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace fcitx {

namespace {

constexpr size_t kMaxRingBytes = 6144; // ~2048 CJK chars

std::string defaultSocketPath() {
    if (const char *xdg = std::getenv("XDG_RUNTIME_DIR"); xdg && *xdg) {
        return std::string(xdg) + "/fcitx5-llm-completion.sock";
    }
    return "/tmp/fcitx5-llm-completion-" + std::to_string(getuid()) + ".sock";
}

std::string jsonEscape(const std::string &in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    return out;
}

bool extractJsonUint64(const std::string &line, const std::string &key,
                       uint64_t &out) {
    std::string needle = "\"" + key + "\":";
    auto pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    if (pos >= line.size() || line[pos] < '0' || line[pos] > '9') {
        return false;
    }
    uint64_t val = 0;
    while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
        val = val * 10 + static_cast<uint64_t>(line[pos] - '0');
        ++pos;
    }
    out = val;
    return true;
}

bool parseJsonStringAt(const std::string &line, size_t &pos, std::string &out) {
    if (pos >= line.size() || line[pos] != '"') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < line.size()) {
        char c = line[pos++];
        if (c == '"') {
            return true;
        }
        if (c == '\\' && pos < line.size()) {
            char esc = line[pos++];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
                if (pos + 4 <= line.size()) {
                    unsigned int cp = 0;
                    bool ok = true;
                    for (int i = 0; i < 4; ++i) {
                        char h = line[pos + i];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') {
                            cp |= static_cast<unsigned int>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            cp |= static_cast<unsigned int>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            cp |= static_cast<unsigned int>(h - 'A' + 10);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    pos += 4;
                    if (ok) {
                        out += utf8::UCS4ToUTF8(cp);
                    }
                }
                break;
            default:
                out.push_back(esc);
                break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

bool extractJsonString(const std::string &line, const std::string &key,
                       std::string &out) {
    std::string needle = "\"" + key + "\":";
    auto pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    return parseJsonStringAt(line, pos, out);
}

bool extractJsonStringArray(const std::string &line, const std::string &key,
                            std::vector<std::string> &out) {
    std::string needle = "\"" + key + "\":";
    auto pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    if (pos >= line.size() || line[pos] != '[') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < line.size()) {
        while (pos < line.size() &&
               (line[pos] == ' ' || line[pos] == '\t' || line[pos] == ',')) {
            ++pos;
        }
        if (pos < line.size() && line[pos] == ']') {
            return true;
        }
        std::string item;
        if (!parseJsonStringAt(line, pos, item)) {
            return false;
        }
        if (!item.empty()) {
            out.push_back(std::move(item));
        }
    }
    return false;
}

void popLastUtf8Char(std::string &s) {
    while (!s.empty()) {
        unsigned char c = static_cast<unsigned char>(s.back());
        s.pop_back();
        if ((c & 0xC0) != 0x80) {
            break;
        }
    }
}

void trimRingBuffer(std::string &s) {
    if (s.size() <= kMaxRingBytes) {
        return;
    }
    size_t cut = s.size() - kMaxRingBytes;
    while (cut < s.size() &&
           (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        ++cut;
    }
    s.erase(0, cut);
}

bool isAsciiWordChar(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

// Extracts one logical token/unit from the front of s:
// optional leading ASCII spaces + either one ASCII word or one UTF-8 codepoint.
size_t firstTokenByteLen(const std::string &s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
        ++i;
    }
    if (i >= s.size()) {
        return s.size();
    }
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (isAsciiWordChar(c)) {
        while (i < s.size() &&
               isAsciiWordChar(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        return i;
    }
    // Advance by one full UTF-8 codepoint.
    size_t charLen = 1;
    if ((c & 0xE0) == 0xC0) {
        charLen = 2;
    } else if ((c & 0xF0) == 0xE0) {
        charLen = 3;
    } else if ((c & 0xF8) == 0xF0) {
        charLen = 4;
    }
    return std::min(s.size(), i + charLen);
}

} // namespace

class LlmCompletion : public AddonInstance {
public:
    explicit LlmCompletion(Instance *instance)
        : instance_(instance), socketPath_(defaultSocketPath()) {
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

                // Ctrl+Alt+I toggles inline proposal on/off.
                if (keyEvent.key().check(
                        Key(FcitxKey_i, KeyStates{KeyState::Ctrl, KeyState::Alt})) ||
                    keyEvent.key().check(
                        Key(FcitxKey_I, KeyStates{KeyState::Ctrl, KeyState::Alt}))) {
                    toggleEnabled(ic);
                    keyEvent.filterAndAccept();
                    return;
                }

                // Never dismiss or cancel on pure modifier key presses (e.g. holding Ctrl for Ctrl+Tab).
                if (keyEvent.key().isModifier()) {
                    return;
                }

                // If we currently display a proposal on this IC, handle accept/dismiss.
                if (hasProposal_ && activeIC_ == ic) {
                    // Ctrl+Tab: accept one BPE token/word chunk from the model stream.
                    if (keyEvent.key().check(Key(FcitxKey_Tab, KeyState::Ctrl))) {
                        cancelPending();
                        std::string chunk;
                        std::string remaining;
                        std::vector<std::string> remainingChunks;

                        if (!proposalChunks_.empty()) {
                            chunk = proposalChunks_.front();
                            remainingChunks.assign(proposalChunks_.begin() + 1,
                                                   proposalChunks_.end());
                            for (const auto &c : remainingChunks) {
                                remaining += c;
                            }
                        } else {
                            size_t n = firstTokenByteLen(proposalText_);
                            chunk = proposalText_.substr(0, n);
                            remaining = proposalText_.substr(n);
                        }

                        clearProposal(ic);
                        suppressNextCommit_ = true;
                        ic->commitString(chunk);
                        suppressNextCommit_ = false;

                        auto &buf = ringBuffers_[ic];
                        buf += chunk;
                        trimRingBuffer(buf);

                        if (!remaining.empty()) {
                            showProposal(ic, remaining, remainingChunks);
                        } else {
                            scheduleTrigger(ic);
                        }
                        keyEvent.filterAndAccept();
                        return;
                    }
                    // Ctrl+Shift+Tab: accept the entire remaining proposal.
                    if (keyEvent.key().check(Key(
                            FcitxKey_ISO_Left_Tab,
                            KeyStates{KeyState::Ctrl, KeyState::Shift})) ||
                        keyEvent.key().check(Key(
                            FcitxKey_Tab,
                            KeyStates{KeyState::Ctrl, KeyState::Shift}))) {
                        std::string accepted = proposalText_;
                        clearProposal(ic);
                        cancelPending();
                        suppressNextCommit_ = true;
                        ic->commitString(accepted);
                        suppressNextCommit_ = false;
                        auto &buf = ringBuffers_[ic];
                        buf += accepted;
                        trimRingBuffer(buf);
                        scheduleTrigger(ic);
                        keyEvent.filterAndAccept();
                        return;
                    }
                    if (keyEvent.key().check(Key(FcitxKey_Escape))) {
                        clearProposal(ic);
                        cancelPending();
                        keyEvent.filterAndAccept();
                        return;
                    }
                    // Any other key clears our proposal and passes through to the main IM.
                    clearProposal(ic);
                }

                // Cancel any pending trigger or inflight request on non-modifier key press.
                cancelPending();
            }));

        // Post-IM: track plain ASCII typing and trigger debounce when main IM is idle.
        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextKeyEvent, EventWatcherPhase::PostInputMethod,
            [this](Event &event) {
                auto &keyEvent = static_cast<KeyEvent &>(event);
                if (keyEvent.isRelease() || keyEvent.filtered()) {
                    return;
                }
                auto *ic = keyEvent.inputContext();
                const Key &k = keyEvent.key();

                if (k.check(Key(FcitxKey_BackSpace))) {
                    popLastUtf8Char(ringBuffers_[ic]);
                    return;
                }
                if (k.check(Key(FcitxKey_Return)) ||
                    k.check(Key(FcitxKey_KP_Enter)) ||
                    k.check(Key(FcitxKey_Left)) ||
                    k.check(Key(FcitxKey_Right)) ||
                    k.check(Key(FcitxKey_Up)) ||
                    k.check(Key(FcitxKey_Down)) ||
                    k.check(Key(FcitxKey_Home)) ||
                    k.check(Key(FcitxKey_End))) {
                    ringBuffers_[ic].clear();
                    return;
                }

                // Ignore pure modifier presses or Ctrl/Alt shortcuts.
                if (k.isModifier() ||
                    k.states().testAny(KeyStates{KeyState::Ctrl, KeyState::Alt,
                                                 KeyState::Super})) {
                    return;
                }

                auto chr = Key::keySymToUnicode(k.sym());
                if (chr >= 0x20 && chr != 0x7f) {
                    auto &buf = ringBuffers_[ic];
                    buf += utf8::UCS4ToUTF8(chr);
                    trimRingBuffer(buf);
                    scheduleTrigger(ic);
                }
            }));

        // Observe commits from main IMs (e.g. Rime / Mozc).
        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextCommitString,
            EventWatcherPhase::PostInputMethod, [this](Event &event) {
                if (suppressNextCommit_) {
                    return;
                }
                auto &ce = static_cast<CommitStringEvent &>(event);
                auto *ic = ce.inputContext();
                if (!ce.text().empty()) {
                    auto &buf = ringBuffers_[ic];
                    buf += ce.text();
                    trimRingBuffer(buf);
                    scheduleTrigger(ic);
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextFocusIn, EventWatcherPhase::PostInputMethod,
            [this](Event &event) {
                auto *ic = static_cast<InputContextEvent &>(event).inputContext();
                ic->statusArea().addAction(StatusGroup::AfterInputMethod,
                                           &toggleAction_);
                updateActionState(ic);
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextInputMethodActivated,
            EventWatcherPhase::PostInputMethod, [this](Event &event) {
                auto *ic = static_cast<InputContextEvent &>(event).inputContext();
                ic->statusArea().addAction(StatusGroup::AfterInputMethod,
                                           &toggleAction_);
                updateActionState(ic);
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextFocusOut, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic = static_cast<InputContextEvent &>(event).inputContext();
                if (activeIC_ == ic) {
                    clearProposal(ic);
                    cancelPending();
                    activeIC_ = nullptr;
                }
                ringBuffers_[ic].clear();
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextReset, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic = static_cast<InputContextEvent &>(event).inputContext();
                // Some apps emit Reset right after CommitString; only clear an
                // already displayed proposal without killing a newly scheduled trigger.
                if (activeIC_ == ic && hasProposal_) {
                    clearProposal(ic);
                }
            }));

        handlers_.emplace_back(instance_->watchEvent(
            EventType::InputContextDestroyed, EventWatcherPhase::PreInputMethod,
            [this](Event &event) {
                auto *ic = static_cast<InputContextEvent &>(event).inputContext();
                if (activeIC_ == ic) {
                    cancelPending();
                    hasProposal_ = false;
                    proposalText_.clear();
                    activeIC_ = nullptr;
                }
                ringBuffers_.erase(ic);
            }));
    }

    ~LlmCompletion() override {
        closeSocket();
    }

private:
    bool isEligible(InputContext *ic, bool allowOwnProposal = false) const {
        if (!enabled_ || !ic || !ic->hasFocus()) {
            return false;
        }
        auto caps = ic->capabilityFlags();
        if (caps.test(CapabilityFlag::Password)) {
            return false;
        }
        if (!caps.test(CapabilityFlag::Preedit)) {
            return false;
        }
        // Never collide with an active main IM composition or candidate list.
        if (!allowOwnProposal && !ic->inputPanel().clientPreedit().empty()) {
            return false;
        }
        if (!ic->inputPanel().preedit().empty()) {
            return false;
        }
        if (auto cand = ic->inputPanel().candidateList();
            cand && !cand->empty()) {
            return false;
        }
        return true;
    }

    void extractContext(InputContext *ic, std::string &prefix,
                        std::string &suffix) {
        prefix.clear();
        suffix.clear();

        if (ic->capabilityFlags().test(CapabilityFlag::SurroundingText)) {
            const auto &st = ic->surroundingText();
            if (st.isValid() && !st.text().empty()) {
                const auto &text = st.text();
                auto cursor = st.cursor();
                auto byteOff = utf8::ncharByteLength(text.begin(), cursor);
                if (byteOff >= 0 && static_cast<size_t>(byteOff) <= text.size()) {
                    prefix = text.substr(0, static_cast<size_t>(byteOff));
                    suffix = text.substr(static_cast<size_t>(byteOff));
                }
            }
        }
        // If the app hasn't pushed updated SurroundingText yet after CommitString,
        // prefer the local ring buffer when it has newer text.
        if (auto it = ringBuffers_.find(ic);
            it != ringBuffers_.end() && !it->second.empty()) {
            if (prefix.empty() || !prefix.ends_with(it->second)) {
                if (prefix.empty()) {
                    prefix = it->second;
                } else if (it->second.size() <= 64 &&
                           !prefix.ends_with(it->second)) {
                    // Check if the tail of ringBuffer was just committed and not yet in SurroundingText.
                    size_t overlap = 0;
                    size_t maxCheck = std::min(prefix.size(), it->second.size());
                    for (size_t n = maxCheck; n > 0; --n) {
                        if (prefix.compare(prefix.size() - n, n, it->second, 0,
                                           n) == 0) {
                            overlap = n;
                            break;
                        }
                    }
                    if (overlap > 0) {
                        prefix.append(it->second, overlap, std::string::npos);
                    }
                }
            }
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
        clearProposal(ic);
        cancelPending();
        updateActionState(ic);
        flashStatus(ic, enabled_ ? "[LLM Completion: ON]"
                                 : "[LLM Completion: OFF]");
    }

    void scheduleTrigger(InputContext *ic) {
        // Do NOT call isEligible(ic) synchronously here: during CommitString,
        // main IMs like Rime have not cleared their preedit/candidates yet.
        if (!enabled_ || !ic) {
            return;
        }
        activeIC_ = ic;
        deferTrigger_ = instance_->eventLoop().addDeferEvent(
            [this](EventSource *) {
                deferTrigger_.reset();
                triggerRequest();
                return true;
            });
    }

    void cancelPending() {
        deferTrigger_.reset();
        if (inflightReqID_ != 0) {
            sendRaw("{\"type\":\"cancel\",\"req_id\":" +
                    std::to_string(inflightReqID_) + "}\n");
            inflightReqID_ = 0;
        }
    }

    void triggerRequest() {
        auto *ic = activeIC_;
        if (!isEligible(ic)) {
            return;
        }
        std::string prefix, suffix;
        extractContext(ic, prefix, suffix);
        if (prefix.empty()) {
            return;
        }
        if (!ensureConnected()) {
            return;
        }

        uint64_t reqID = ++nextReqID_;
        inflightReqID_ = reqID;

        std::string msg = "{\"type\":\"suggest\",\"req_id\":" +
                          std::to_string(reqID) + ",\"program\":\"" +
                          jsonEscape(ic->program()) + "\",\"prefix\":\"" +
                          jsonEscape(prefix) + "\",\"suffix\":\"" +
                          jsonEscape(suffix) + "\"}\n";
        if (!sendRaw(msg)) {
            inflightReqID_ = 0;
        }
    }

    void showProposal(InputContext *ic, const std::string &text,
                      const std::vector<std::string> &chunks) {
        if (text.empty() || !isEligible(ic, hasProposal_ && activeIC_ == ic)) {
            return;
        }
        Text preedit;
        preedit.append(text, TextFormatFlag::Underline);
        preedit.setCursor(0);
        ic->inputPanel().setClientPreedit(preedit);
        ic->updatePreedit();
        hasProposal_ = true;
        proposalText_ = text;
        proposalChunks_ = chunks;
    }

    void clearProposal(InputContext *ic) {
        if (!hasProposal_) {
            return;
        }
        hasProposal_ = false;
        proposalText_.clear();
        proposalChunks_.clear();
        if (ic) {
            ic->inputPanel().setClientPreedit(Text());
            ic->updatePreedit();
        }
    }

    void flashStatus(InputContext *ic, const std::string &msg) {
        if (!ic) {
            return;
        }
        ic->inputPanel().setAuxDown(Text(msg));
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        statusTimer_ = instance_->eventLoop().addTimeEvent(
            CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 1200000, 50000,
            [this, ic](EventSourceTime *, uint64_t) {
                statusTimer_.reset();
                if (ic && ic->hasFocus()) {
                    ic->inputPanel().setAuxDown(Text());
                    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
                }
                return true;
            });
    }

    bool ensureConnected() {
        if (sockFd_ >= 0) {
            return true;
        }
        int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) {
            return false;
        }
        struct sockaddr_un addr {};
        addr.sun_family = AF_UNIX;
        if (socketPath_.size() >= sizeof(addr.sun_path)) {
            ::close(fd);
            return false;
        }
        std::strncpy(addr.sun_path, socketPath_.c_str(),
                     sizeof(addr.sun_path) - 1);
        if (::connect(fd, reinterpret_cast<struct sockaddr *>(&addr),
                      sizeof(addr)) < 0) {
            if (errno != EINPROGRESS) {
                ::close(fd);
                return false;
            }
        }

        sockFd_ = fd;
        readBuf_.clear();
        ioEvent_ = instance_->eventLoop().addIOEvent(
            sockFd_, IOEventFlags{IOEventFlag::In, IOEventFlag::Err,
                                  IOEventFlag::Hup},
            [this](EventSourceIO *, int, IOEventFlags flags) {
                if (flags.testAny(IOEventFlags{IOEventFlag::Err,
                                               IOEventFlag::Hup})) {
                    closeSocket();
                    return true;
                }
                onSocketReadable();
                return true;
            });
        return true;
    }

    void closeSocket() {
        ioEvent_.reset();
        if (sockFd_ >= 0) {
            ::close(sockFd_);
            sockFd_ = -1;
        }
        readBuf_.clear();
        inflightReqID_ = 0;
    }

    bool sendRaw(const std::string &data) {
        if (sockFd_ < 0) {
            return false;
        }
        ssize_t n = ::send(sockFd_, data.data(), data.size(), MSG_NOSIGNAL);
        if (n < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                closeSocket();
            }
            return false;
        }
        return static_cast<size_t>(n) == data.size();
    }

    void onSocketReadable() {
        char buf[4096];
        while (sockFd_ >= 0) {
            ssize_t n = ::recv(sockFd_, buf, sizeof(buf), 0);
            if (n > 0) {
                readBuf_.append(buf, static_cast<size_t>(n));
            } else if (n == 0) {
                closeSocket();
                return;
            } else {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    break;
                }
                closeSocket();
                return;
            }
        }

        size_t pos = 0;
        while ((pos = readBuf_.find('\n')) != std::string::npos) {
            std::string line = readBuf_.substr(0, pos);
            readBuf_.erase(0, pos + 1);
            handleDaemonLine(line);
        }
    }

    void handleDaemonLine(const std::string &line) {
        std::string type;
        if (!extractJsonString(line, "type", type) || type != "proposal") {
            return;
        }
        uint64_t reqID = 0;
        if (!extractJsonUint64(line, "req_id", reqID) ||
            reqID != inflightReqID_) {
            return;
        }
        if (line.find("\"done\":true") != std::string::npos) {
            inflightReqID_ = 0;
        }

        std::string text;
        if (!extractJsonString(line, "text", text) || text.empty()) {
            return;
        }
        std::vector<std::string> chunks;
        extractJsonStringArray(line, "chunks", chunks);
        if (activeIC_) {
            showProposal(activeIC_, text, chunks);
        }
    }

    Instance *instance_;
    std::string socketPath_;
    SimpleAction toggleAction_;
    ScopedConnection toggleConn_;
    std::vector<std::unique_ptr<HandlerTableEntry<EventHandler>>> handlers_;
    std::unordered_map<InputContext *, std::string> ringBuffers_;

    bool enabled_ = true;
    bool hasProposal_ = false;
    bool suppressNextCommit_ = false;
    std::string proposalText_;
    std::vector<std::string> proposalChunks_;
    InputContext *activeIC_ = nullptr;

    std::unique_ptr<EventSource> deferTrigger_;
    std::unique_ptr<EventSourceTime> statusTimer_;
    std::unique_ptr<EventSourceIO> ioEvent_;

    int sockFd_ = -1;
    std::string readBuf_;
    uint64_t nextReqID_ = 0;
    uint64_t inflightReqID_ = 0;
};

class LlmCompletionFactory : public AddonFactory {
    AddonInstance *create(AddonManager *manager) override {
        return new LlmCompletion(manager->instance());
    }
};

} // namespace fcitx

FCITX_ADDON_FACTORY_V2(llmcompletion, fcitx::LlmCompletionFactory);
