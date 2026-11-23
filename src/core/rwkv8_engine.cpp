#include "rwkv8_engine.hpp"
#include "text_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace fcitx5_llm {

namespace {

constexpr size_t kSyncPrefixChars = 64;
constexpr size_t kTrackedPrefixChars = 96;
constexpr size_t kSyncSuffixChars = 40;

std::vector<std::string_view> splitUtf8Codepoints(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char b = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((b & 0xE0) == 0xC0) {
            len = 2;
        } else if ((b & 0xF0) == 0xE0) {
            len = 3;
        } else if ((b & 0xF8) == 0xF0) {
            len = 4;
        }
        len = std::min(len, s.size() - i);
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::string extractCanonicalFimSuffixHead(std::string_view rawSuffix) {
    std::string clean = stripInvisibleFormattingUtf8(rawSuffix);
    std::string head = utf8HeadChars(clean, kSyncSuffixChars);
    if (trimWhitespace(head).empty()) {
        return {};
    }
    return head;
}

bool hasNgramRepeat(std::span<const int32_t> ids) {
    size_t n = ids.size();
    if (n >= 3 && ids[n - 1] == ids[n - 2] && ids[n - 2] == ids[n - 3]) {
        return true;
    }
    for (size_t k : {size_t(2), size_t(3), size_t(4)}) {
        if (n >= 2 * k &&
            std::equal(ids.end() - static_cast<ptrdiff_t>(k), ids.end(),
                       ids.end() - static_cast<ptrdiff_t>(2 * k))) {
            return true;
        }
    }
    return false;
}

bool endsWithClausePunctuation(std::string_view s) {
    if (s.empty()) {
        return false;
    }
    static constexpr std::string_view kClauseEnds[] = {
        "，", "。", "？", "！", "；", "：", "、",
        ",",  ".",  "?",  "!",  ";",  ":"};
    for (auto p : kClauseEnds) {
        if (s.size() >= p.size() &&
            s.substr(s.size() - p.size()) == p) {
            return true;
        }
    }
    return false;
}

std::string lowerAsciiCopy(std::string s) {
    for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s;
}

std::filesystem::path
resolveAdapterStatePath(const std::filesystem::path &warmupLogPath,
                        const std::filesystem::path &explicitPath) {
    if (!explicitPath.empty()) {
        return explicitPath;
    }
    if (!warmupLogPath.empty()) {
        return warmupLogPath.parent_path() / "online_adapter.bin";
    }
    return {};
}

std::filesystem::path
resolveSuffixStatePath(const std::filesystem::path &warmupLogPath,
                       const std::filesystem::path &explicitAdapterPath) {
    if (!explicitAdapterPath.empty()) {
        return std::filesystem::path(explicitAdapterPath.string() + ".suffix");
    }
    if (!warmupLogPath.empty()) {
        return warmupLogPath.parent_path() / "online_suffix.bin";
    }
    return {};
}

} // namespace

Rwkv8CompletionEngine::Rwkv8CompletionEngine(
    std::unique_ptr<IRwkvStepRunner> runner,
    const std::filesystem::path &modelDir,
    const std::filesystem::path &warmupLogPath,
    const std::filesystem::path &adapterStatePath)
    : runner_(std::move(runner)),
      codec_(modelDir / "unicode_vocab.json", runner_->config().vocabSize),
      sam_(98304),
      dynSuffix_(32768, runner_->config().vocabSize,
                 resolveSuffixStatePath(warmupLogPath, adapterStatePath)),
      personalIndex_(modelDir / "personal_suffix_index.npz", 2.0f),
      onlineAdapter_(std::make_unique<OnlineAdapterManager>(
          runner_.get(),
          resolveAdapterStatePath(warmupLogPath, adapterStatePath))),
      mainState_(RwkvState::createZero(runner_->config())) {
    const auto &cfg = runner_->config();
    tmpXIn_.resize(static_cast<size_t>(cfg.nEmbd));
    tmpEnn_.resize(static_cast<size_t>(cfg.nLayer) * cfg.nEmbd);
    tmpLogits_.resize(static_cast<size_t>(cfg.vocabSize));
    tmpXOut_.resize(static_cast<size_t>(cfg.nEmbd));

    dynSuffix_.seedBackgroundUnigrams(personalIndex_);

    if (!warmupLogPath.empty()) {
        warmedChars_ = sam_.warmupFromLog(warmupLogPath, codec_, 48000);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(
                resolveSuffixStatePath(warmupLogPath, adapterStatePath), ec)) {
            dynSuffix_.warmupFromLog(warmupLogPath, codec_, 48000);
        }
    }
}

void Rwkv8CompletionEngine::prepareXIn(int32_t tokId, int32_t rosaNt,
                                       int32_t rosaMl, float *xOut,
                                       float *ennOut) const {
    const auto &cfg = runner_->config();
    const auto &tb = runner_->tables();
    size_t d = static_cast<size_t>(cfg.nEmbd);
    int32_t safeTok = std::clamp(tokId, 0, cfg.vocabSize - 1);

    std::memcpy(xOut, tb.emb.data() + static_cast<size_t>(safeTok) * d,
                d * sizeof(float));

    if (rosaNt >= BYTE_OFFSET && rosaMl >= 2 && rosaNt < cfg.vocabSize) {
        const float *mEmb = tb.emb.data() + static_cast<size_t>(rosaNt) * d;
        int32_t lb = std::clamp(rosaMl, 0, tb.maxLenBucket);
        const float *lenRow = tb.rosaLenEmb.data() + static_cast<size_t>(lb) * d;
        for (size_t r = 0; r < d; ++r) {
            const float *projRow = tb.rosaEmbProj.data() + r * d;
            float dot = 0.0f;
            for (size_t c = 0; c < d; ++c) {
                dot += projRow[c] * mEmb[c];
            }
            xOut[r] += dot + lenRow[r];
        }
    }

    size_t nLayer = static_cast<size_t>(cfg.nLayer);
    if (!tb.ennTable.empty()) {
        size_t v = static_cast<size_t>(cfg.vocabSize);
        for (size_t l = 0; l < nLayer; ++l) {
            const float *src =
                tb.ennTable.data() + (l * v + static_cast<size_t>(safeTok)) * d;
            std::memcpy(ennOut + l * d, src, d * sizeof(float));
        }
    } else {
        size_t rDim = static_cast<size_t>(cfg.deRank);
        size_t v = static_cast<size_t>(cfg.vocabSize);
        for (size_t l = 0; l < nLayer; ++l) {
            const float *uVec =
                tb.ennU.data() +
                (l * v + static_cast<size_t>(safeTok)) * rDim;
            const float *vMat = tb.ennV.data() + l * rDim * d;
            float *dst = ennOut + l * d;
            for (size_t j = 0; j < d; ++j) {
                dst[j] = 1.0f;
            }
            for (size_t k = 0; k < rDim; ++k) {
                float uk = uVec[k];
                const float *vRow = vMat + k * d;
                for (size_t j = 0; j < d; ++j) {
                    dst[j] += uk * vRow[j];
                }
            }
        }
    }
}

void Rwkv8CompletionEngine::stepMainToken(int32_t tokId, int32_t rosaNt,
                                          int32_t rosaMl) {
    prepareXIn(tokId, rosaNt, rosaMl, tmpXIn_.data(), tmpEnn_.data());
    runner_->inferStep(tmpXIn_.data(), tmpEnn_.data(), mainState_,
                       tmpLogits_.data(), tmpXOut_.data());
    prevRosaNt_ = rosaNt;
    prevRosaMl_ = rosaMl;
}

void Rwkv8CompletionEngine::appendContinuousChars(std::string_view text,
                                                  float trainLrScale,
                                                  float dynWeight) {
    if (text.empty()) {
        return;
    }
    if (!streamPrimed_) {
        syncContextWindow("", "", {});
    }

    auto pushCheckpoint = [&]() {
        if (checkpoints_.size() >= kMaxCheckpoints) {
            checkpoints_.pop_front();
        }
        checkpoints_.push_back(TokenCheckpoint{
            mainState_, tmpLogits_, tmpXOut_, prevRosaNt_, prevRosaMl_,
            hasPrevPredState_, syncedPrefix_});
    };

    for (std::string_view ch : splitUtf8Codepoints(text)) {
        auto chToks = codec_.encode(ch);
        if (ch == "\n") {
            onlineAdapter_->flushAllPending();
            sam_.boundary();
            dynSuffix_.boundary();
            for (int32_t t : chToks) {
                if (t < BYTE_OFFSET) {
                    continue;
                }
                pushCheckpoint();
                stepMainToken(t, -1, 0);
                hasPrevPredState_ = true;
            }
            syncedPrefix_ =
                utf8TailChars(syncedPrefix_ + "\n", kTrackedPrefixChars);
            continue;
        }

        for (int32_t t : chToks) {
            if (t < BYTE_OFFSET) {
                continue;
            }
            pushCheckpoint();
            if (trainLrScale > 0.0f && hasPrevPredState_) {
                // Prevent ROSA Shortcut Starvation: discount ephemeral SAM injection
                // so the persistent NPU neural adapter continues learning!
                if (prevRosaNt_ == t && prevRosaMl_ >= 2 &&
                    t < static_cast<int32_t>(tmpLogits_.size())) {
                    tmpLogits_[static_cast<size_t>(t)] -= 9.0f;
                }
                onlineAdapter_->enqueueTokenSample(
                    tmpXOut_.data(), tmpLogits_.data(), t, trainLrScale);
            }
            if (dynWeight > 0.0f) {
                dynSuffix_.pushToken(t, dynWeight);
            }
            auto m = sam_.pushCharTok(t);
            stepMainToken(t, m.nextTok, m.matchLen);
            hasPrevPredState_ = true;
        }
        syncedPrefix_ = utf8TailChars(syncedPrefix_ + std::string(ch),
                                      kTrackedPrefixChars);
    }
    hasSyncedTail_ = true;
}

void Rwkv8CompletionEngine::feedRejectedProposal(
    std::string_view rejectedText) {
    if (rejectedText.empty()) {
        return;
    }
    std::string head = utf8HeadChars(rejectedText, 8);
    if (std::find(recentlyRejected_.begin(), recentlyRejected_.end(), head) ==
        recentlyRejected_.end()) {
        if (recentlyRejected_.size() >= 16) {
            recentlyRejected_.erase(recentlyRejected_.begin());
        }
        recentlyRejected_.push_back(std::move(head));
    }

    auto enc = codec_.encode(utf8HeadChars(rejectedText, 16));
    if (!enc.empty() && hasLastSuggestStep0_) {
        onlineAdapter_->penalizeRejectedToken(lastSuggestStep0XOut_.data(),
                                              lastSuggestStep0BaseLogits_.data(),
                                              enc[0]);
    }
    // Intentionally keep mainState_ 100% pure continuous document text without
    // injecting <PROPOSE>...<REJECT> tokens into the recurrent sliding window.
}

bool Rwkv8CompletionEngine::syncContextWindow(
    std::string_view prefix, std::string_view sufHead,
    const std::function<bool()> &shouldAbort) {
    std::string tailCheck = utf8TailChars(prefix, 16);
    if (streamPrimed_ && hasSyncedTail_ && hasSyncedSuffix_ &&
        tailCheck == utf8TailChars(syncedPrefix_, 16) &&
        sufHead == lastSyncedSuffixHead_) {
        hasPendingAnchor_ = false;
        return true;
    }

    const auto &cfg = runner_->config();
    RwkvState nextState = RwkvState::createZero(cfg);
    std::vector<float> nextLogits(static_cast<size_t>(cfg.vocabSize), 0.0f);
    std::vector<float> nextXOut(static_cast<size_t>(cfg.nEmbd), 0.0f);
    int32_t nextPrevNt = -1;
    int32_t nextPrevMl = 0;
    bool nextHasPrevPred = false;
    std::deque<TokenCheckpoint> nextCheckpoints;

    auto stepTx = [&](int32_t tokId, int32_t rosaNt, int32_t rosaMl) -> bool {
        if (shouldAbort && shouldAbort()) {
            return false;
        }
        prepareXIn(tokId, rosaNt, rosaMl, tmpXIn_.data(), tmpEnn_.data());
        runner_->inferStep(tmpXIn_.data(), tmpEnn_.data(), nextState,
                           nextLogits.data(), nextXOut.data());
        nextPrevNt = rosaNt;
        nextPrevMl = rosaMl;
        return true;
    };

    if (!lastProgram_.empty()) {
        if (!stepTx(TOK_FOCUS, -1, 0)) {
            return false;
        }
        for (int32_t ft : codec_.encode(lastProgram_ + "\n")) {
            if (!stepTx(ft, -1, 0)) {
                return false;
            }
        }
    }

    sam_.boundary();
    dynSuffix_.boundary();

    if (!sufHead.empty()) {
        if (!stepTx(TOK_AFTER, -1, 0)) {
            sam_.boundary();
            return false;
        }
        for (int32_t st : codec_.encode(sufHead)) {
            if (!stepTx(st, -1, 0)) {
                sam_.boundary();
                return false;
            }
        }
        if (!stepTx(TOK_BEFORE, -1, 0)) {
            sam_.boundary();
            return false;
        }
        nextHasPrevPred = true;
    } else {
        auto m0 = sam_.queryCurrent();
        if (!stepTx(TOK_TYPE, m0.nextTok, m0.matchLen)) {
            sam_.boundary();
            return false;
        }
        nextHasPrevPred = true;
    }

    std::string preWin = utf8TailChars(prefix, kSyncPrefixChars);
    std::string runningPre =
        (prefix.size() > preWin.size())
            ? utf8TailChars(prefix.substr(0, prefix.size() - preWin.size()),
                            kTrackedPrefixChars)
            : std::string{};
    auto preChars = splitUtf8Codepoints(preWin);
    for (size_t idx = 0; idx < preChars.size(); ++idx) {
        std::string_view ch = preChars[idx];
        bool saveCkpt = (idx + kMaxCheckpoints >= preChars.size());
        auto chToks = codec_.encode(ch);
        if (ch == "\n") {
            sam_.boundary();
            for (int32_t t : chToks) {
                if (t < BYTE_OFFSET) {
                    continue;
                }
                if (saveCkpt) {
                    if (nextCheckpoints.size() >= kMaxCheckpoints) {
                        nextCheckpoints.pop_front();
                    }
                    nextCheckpoints.push_back(TokenCheckpoint{
                        nextState, nextLogits, nextXOut, nextPrevNt, nextPrevMl,
                        nextHasPrevPred, runningPre});
                }
                if (!stepTx(t, -1, 0)) {
                    sam_.boundary();
                    return false;
                }
                nextHasPrevPred = true;
            }
        } else {
            for (int32_t t : chToks) {
                if (t < BYTE_OFFSET) {
                    continue;
                }
                if (saveCkpt) {
                    if (nextCheckpoints.size() >= kMaxCheckpoints) {
                        nextCheckpoints.pop_front();
                    }
                    nextCheckpoints.push_back(TokenCheckpoint{
                        nextState, nextLogits, nextXOut, nextPrevNt, nextPrevMl,
                        nextHasPrevPred, runningPre});
                }
                if (shouldAbort && shouldAbort()) {
                    sam_.boundary();
                    return false;
                }
                auto m = sam_.pushCharTok(t);
                if (!stepTx(t, m.nextTok, m.matchLen)) {
                    sam_.boundary();
                    return false;
                }
                nextHasPrevPred = true;
            }
        }
        runningPre =
            utf8TailChars(runningPre + std::string(ch), kTrackedPrefixChars);
    }

    mainState_ = std::move(nextState);
    checkpoints_ = std::move(nextCheckpoints);
    tmpLogits_ = std::move(nextLogits);
    tmpXOut_ = std::move(nextXOut);
    prevRosaNt_ = nextPrevNt;
    prevRosaMl_ = nextPrevMl;
    hasPrevPredState_ = nextHasPrevPred;
    streamPrimed_ = true;
    syncedPrefix_ = utf8TailChars(prefix, kTrackedPrefixChars);
    lastSyncedSuffixHead_ = std::string(sufHead);
    hasSyncedTail_ = true;
    hasSyncedSuffix_ = true;
    hasPendingAnchor_ = false;
    return true;
}

bool Rwkv8CompletionEngine::prewarmPendingContext(
    const std::function<bool()> &shouldAbort) {
    if (!hasPendingAnchor_) {
        return true;
    }
    std::string pre = pendingAnchorPrefix_;
    std::string suf = pendingAnchorSuffix_;
    return syncContextWindow(pre, suf, shouldAbort);
}

void Rwkv8CompletionEngine::handleEvent(
    const InteractionEvent &ev, const std::function<bool()> &shouldAbort) {
    if (!ev.program.empty() && ev.program != lastProgram_) {
        onlineAdapter_->flushAllPending();
        lastProgram_ = ev.program;
        streamPrimed_ = false;
        hasSyncedTail_ = false;
        hasSyncedSuffix_ = false;
        syncedPrefix_.clear();
        lastSyncedSuffixHead_.clear();
        checkpoints_.clear();
        hasPendingAnchor_ = false;
    }

    if (ev.event == "focus_in" || ev.event == "cursor_jump") {
        onlineAdapter_->flushAllPending();
        std::string sufHead = extractCanonicalFimSuffixHead(ev.suffix);
        std::string cleanPre = stripInvisibleFormattingUtf8(ev.prefix);
        std::string tailCheck = utf8TailChars(cleanPre, 16);
        if (!hasPendingAnchor_ && streamPrimed_ && hasSyncedTail_ &&
            hasSyncedSuffix_ &&
            tailCheck == utf8TailChars(syncedPrefix_, 16) &&
            sufHead == lastSyncedSuffixHead_) {
            return;
        }
        sam_.boundary();
        dynSuffix_.boundary();
        streamPrimed_ = false;
        hasPrevPredState_ = false;
        hasSyncedTail_ = false;
        hasSyncedSuffix_ = false;
        checkpoints_.clear();
        pendingAnchorPrefix_ = utf8TailChars(cleanPre, kTrackedPrefixChars);
        pendingAnchorSuffix_ = std::move(sufHead);
        hasPendingAnchor_ = true;
    } else if (ev.event == "backspace") {
        std::string del = stripInvisibleFormattingUtf8(ev.text);
        int nDel = del.empty()
                       ? 1
                       : std::max(1, static_cast<int>(codec_.encode(del).size()));
        sam_.tombstoneBackspace(nDel);
        dynSuffix_.rollbackBackspace(nDel);
        onlineAdapter_->rollbackPending(nDel);

        if (hasPendingAnchor_) {
            popLastUtf8Char(pendingAnchorPrefix_);
        } else if (streamPrimed_ && hasSyncedTail_ &&
                   checkpoints_.size() >= static_cast<size_t>(nDel)) {
            for (int k = 0; k + 1 < nDel; ++k) {
                checkpoints_.pop_back();
            }
            auto ckpt = std::move(checkpoints_.back());
            checkpoints_.pop_back();
            mainState_ = std::move(ckpt.state);
            tmpLogits_ = std::move(ckpt.logits);
            tmpXOut_ = std::move(ckpt.xOut);
            prevRosaNt_ = ckpt.prevRosaNt;
            prevRosaMl_ = ckpt.prevRosaMl;
            hasPrevPredState_ = ckpt.hasPrevPredState;
            syncedPrefix_ = std::move(ckpt.syncedPrefix);
            hasSyncedTail_ = true;
        } else {
            checkpoints_.clear();
            streamPrimed_ = false;
            hasPrevPredState_ = false;
            hasSyncedTail_ = false;
            syncedPrefix_.clear();
        }
    } else if (ev.event == "user_type") {
        std::string txt = stripInvisibleFormattingUtf8(ev.text);
        if (!txt.empty()) {
            if (hasPendingAnchor_) {
                if (!prewarmPendingContext(shouldAbort)) {
                    return;
                }
            }
            appendContinuousChars(txt, 1.0f, 1.0f);
            recentlyRejected_.clear();
        }
    } else if (ev.event == "accept_chunk" || ev.event == "accept_word" ||
               ev.event == "accept_all") {
        std::string acc = stripInvisibleFormattingUtf8(ev.accepted);
        if (!acc.empty()) {
            if (hasPendingAnchor_) {
                if (!prewarmPendingContext(shouldAbort)) {
                    return;
                }
            }
            // Feed accepted characters as pure continuous Unicode text (zero <ACCEPT>
            // fragmentation), boosting online learning weights!
            appendContinuousChars(acc, 1.9f, 2.5f);
            onlineAdapter_->flushAllPending();
            recentlyRejected_.clear();
        }
    } else if (ev.event == "dismiss_esc" || ev.event == "dismiss_escape") {
        std::string rem = stripInvisibleFormattingUtf8(ev.remaining);
        if (!rem.empty()) {
            if (hasPendingAnchor_) {
                if (!prewarmPendingContext(shouldAbort)) {
                    return;
                }
            }
            feedRejectedProposal(rem);
        }
    }
}

SuggestOutput Rwkv8CompletionEngine::suggest(
    const SuggestParams &params, const std::function<bool()> &shouldAbort) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    SuggestOutput out;
    out.model = runner_->modelLabel();

    std::string prefix = stripInvisibleFormattingUtf8(params.prefix);
    std::string sufHead = extractCanonicalFimSuffixHead(params.suffix);
    bool hasFimSuffix = !sufHead.empty();
    std::string suffix = hasFimSuffix
                             ? stripInvisibleFormattingUtf8(params.suffix)
                             : std::string{};
    out.mode = hasFimSuffix ? "fim" : "tail";
    int32_t maxChars = std::clamp(params.maxTokens, 4, 14);

    if (prefix.empty()) {
        return out;
    }
    if (shouldAbort && shouldAbort()) {
        out.cancelled = true;
        return out;
    }

    if (!params.program.empty() && params.program != lastProgram_) {
        onlineAdapter_->flushAllPending();
        lastProgram_ = params.program;
        streamPrimed_ = false;
        hasSyncedTail_ = false;
        hasSyncedSuffix_ = false;
        syncedPrefix_.clear();
        lastSyncedSuffixHead_.clear();
        checkpoints_.clear();
        hasPendingAnchor_ = false;
    }

    hasPendingAnchor_ = false;
    std::string tailCheck = utf8TailChars(prefix, 16);
    if (!streamPrimed_ || !hasSyncedTail_ ||
        tailCheck != utf8TailChars(syncedPrefix_, 16) || !hasSyncedSuffix_ ||
        sufHead != lastSyncedSuffixHead_) {
        if (!syncContextWindow(prefix, sufHead, shouldAbort)) {
            out.cancelled = true;
            return out;
        }
    }

    if (shouldAbort && shouldAbort()) {
        out.cancelled = true;
        return out;
    }

    // Copy-on-Write speculative snapshot S_spec <- S_main (single-token <PREDICT> for both tail & FIM)
    RwkvState spState = mainState_;
    int32_t sIdx = sam_.lastState();
    int32_t sLen = sam_.lastLength();

    const auto &cfg = runner_->config();
    const auto &tb = runner_->tables();
    std::vector<float> logits(static_cast<size_t>(cfg.vocabSize));
    std::vector<float> xOut(static_cast<size_t>(cfg.nEmbd));
    // Capture character-stream hidden state right BEFORE TOK_PREDICT so step 0
    // matches the exact hidden subspace trained during feedCommittedTokens!
    std::vector<float> prePredictXOut = tmpXOut_;
    std::vector<float> stepXIn(static_cast<size_t>(cfg.nEmbd));
    std::vector<float> stepEnn(static_cast<size_t>(cfg.nLayer) * cfg.nEmbd);

    SamMatch predMatch = sam_.queryFrom(sIdx, sLen);
    prepareXIn(TOK_PREDICT, predMatch.nextTok, predMatch.matchLen,
               stepXIn.data(), stepEnn.data());
    runner_->inferStep(stepXIn.data(), stepEnn.data(), spState, logits.data(),
                       xOut.data());
    int32_t lastNt = predMatch.nextTok;
    int32_t lastMl = predMatch.matchLen;

    lastSuggestStep0XOut_ = prePredictXOut;
    lastSuggestStep0BaseLogits_ = logits;
    hasLastSuggestStep0_ = true;

    int32_t minMl = minRosaMatchLenForPrefix(prefix);
    auto nlLastPos = prefix.rfind('\n');
    std::string_view curLineView =
        (nlLastPos == std::string::npos)
            ? std::string_view(prefix)
            : std::string_view(prefix).substr(nlLastPos + 1);
    std::vector<int32_t> ctxLineToks =
        codec_.encode(utf8TailChars(curLineView, 16));

    int32_t initialRosaMatchLen = (lastMl >= minMl) ? lastMl : 0;
    std::vector<int32_t> genIds;

    std::vector<int32_t> rejectedFirstToks;
    for (const auto &rHead : recentlyRejected_) {
        auto encR = codec_.encode(rHead);
        if (!encR.empty()) {
            rejectedFirstToks.push_back(encR[0]);
        }
    }

    for (int32_t stepI = 0; stepI < maxChars; ++stepI) {
        if (shouldAbort && shouldAbort()) {
            out.cancelled = true;
            return out;
        }

        std::vector<float> lg = logits;
        // 1. Apply NPU-resident Online SGD Adapter delta logits (subspace-aligned)
        const float *adapterH =
            (stepI == 0) ? prePredictXOut.data() : xOut.data();
        onlineAdapter_->applyLogits(adapterH, lg.data());

        for (int32_t c = 1; c < BYTE_OFFSET; ++c) {
            lg[static_cast<size_t>(c)] = -1e9f;
        }

        std::vector<int32_t> fullLineCtx = ctxLineToks;
        fullLineCtx.insert(fullLineCtx.end(), genIds.begin(), genIds.end());

        // 2. Universal Information-Theoretic SAM (ROSA) copy gate:
        // Modulates copy strength by empirical Dirichlet posterior (self-information
        // + branching entropy across any language, zero hardcoded word lists!)
        int32_t effMl = (stepI == 0) ? minMl : 2;
        if (lastNt >= BYTE_OFFSET && lastMl >= effMl &&
            lastNt < cfg.vocabSize) {
            float rawGate = tb.copyGateB;
            for (int32_t j = 0; j < cfg.nEmbd; ++j) {
                rawGate += xOut[static_cast<size_t>(j)] *
                           tb.copyGateW[static_cast<size_t>(j)];
            }
            float gate = std::log1pf(std::exp(std::min(rawGate, 20.0f)));
            gate = std::max(lastMl >= 3 ? 0.95f : 0.35f, gate);
            float lenFactor =
                std::log1pf(static_cast<float>(lastMl)) +
                0.65f * static_cast<float>(std::max(0, lastMl - 2));
            if (stepI == 0 &&
                fullLineCtx.size() >= static_cast<size_t>(lastMl)) {
                size_t spanLen = std::min<size_t>(8, static_cast<size_t>(lastMl));
                std::span<const int32_t> tailSpan(
                    fullLineCtx.data() + fullLineCtx.size() - spanLen, spanLen);
                float pEmp = dynSuffix_.empiricalPosterior(
                    tailSpan, lastNt, &personalIndex_);
                float confScale = std::clamp(pEmp * 1.85f, 0.12f, 1.0f);
                lenFactor *= confScale;
                if (pEmp < 0.25f) {
                    initialRosaMatchLen = 0;
                }
            }
            lg[static_cast<size_t>(lastNt)] += gate * lenFactor;
        }

        // 3. Live DynamicSuffixMemory (universal empirical-Bayes suffix boost)
        int32_t dynEffMl = (stepI == 0) ? effMl : 2;
        auto [dynDist, dynMl] = dynSuffix_.queryDistribution(
            fullLineCtx, &personalIndex_, dynEffMl);
        for (auto [dTid, dBoost] : dynDist) {
            if (dTid >= 0 && dTid < cfg.vocabSize) {
                lg[static_cast<size_t>(dTid)] += dBoost;
            }
        }
        if (stepI == 0 && dynMl > initialRosaMatchLen) {
            initialRosaMatchLen = dynMl;
        }

        // 4. Static PersonalSuffixIndex (corpus & Rime lexicon)
        auto [persDist, persMl] =
            personalIndex_.queryDistribution(fullLineCtx, effMl);
        float persScale = (persMl == 2) ? 0.65f : 1.0f;
        for (auto [pTid, pBoost] : persDist) {
            if (pTid >= 0 && pTid < cfg.vocabSize) {
                lg[static_cast<size_t>(pTid)] += pBoost * persScale;
            }
        }
        if (stepI == 0 && persMl >= 3 && persMl > initialRosaMatchLen) {
            initialRosaMatchLen = persMl;
        }

        if (stepI == 0) {
            lg[TOK_END] = -1e9f;
            for (int32_t nlT : codec_.newlineTokenIds()) {
                if (nlT >= 0 && nlT < cfg.vocabSize) {
                    lg[static_cast<size_t>(nlT)] = -1e9f;
                }
            }
            for (int32_t rt : rejectedFirstToks) {
                if (rt >= 0 && rt < cfg.vocabSize) {
                    lg[static_cast<size_t>(rt)] = -1e9f;
                    if (rt == lastNt) {
                        initialRosaMatchLen = 0;
                    }
                }
            }
        }

        auto maxIt = std::max_element(lg.begin(), lg.end());
        int32_t nxt = static_cast<int32_t>(maxIt - lg.begin());
        float maxLg = *maxIt;
        double sumExp = 0.0;
        for (float v : lg) {
            float diff = std::clamp(v - maxLg, -30.0f, 0.0f);
            sumExp += std::exp(static_cast<double>(diff));
        }
        float topProb = static_cast<float>(1.0 / std::max(1.0, sumExp));

        if (stepI == 0) {
            out.ttftMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             Clock::now() - t0)
                             .count();
        } else if (nxt == TOK_END) {
            break;
        } else if (lastMl < 2 && dynMl < 2 && stepI >= 2 && topProb < 0.05f) {
            break;
        }

        std::vector<int32_t> candIds = genIds;
        candIds.push_back(nxt);
        if (hasNgramRepeat(candIds)) {
            break;
        }
        std::string candStr = codec_.decode(candIds);
        auto nlInCand = candStr.find('\n');
        if (nlInCand != std::string::npos) {
            candStr.erase(nlInCand);
            genIds = codec_.encode(candStr);
            break;
        }
        genIds = std::move(candIds);

        if (lastMl < 2 && dynMl < 2 && utf8CharCount(candStr) >= 2 &&
            endsWithClausePunctuation(candStr)) {
            break;
        }
        if (stepI + 1 >= maxChars) {
            break;
        }

        auto st = sam_.stepSpeculative(sIdx, sLen, nxt);
        sIdx = st.stateIdx;
        sLen = st.curLen;
        lastNt = st.match.nextTok;
        lastMl = st.match.matchLen;
        prepareXIn(nxt, lastNt, lastMl, stepXIn.data(), stepEnn.data());
        runner_->inferStep(stepXIn.data(), stepEnn.data(), spState,
                           logits.data(), xOut.data());
    }

    std::string rawText = codec_.decode(genIds);
    if (utf8CharCount(rawText) >= 2 &&
        (curLineView == rawText || curLineView.starts_with(rawText))) {
        rawText.clear();
        initialRosaMatchLen = 0;
    }

    std::string trimmed =
        hasFimSuffix ? trimSuffixOverlap(rawText, suffix) : rawText;
    out.rawText = std::move(rawText);
    out.text = std::move(trimmed);
    out.chunks = segmentCompletionChunks(out.text);
    out.rosaMatchLen = initialRosaMatchLen;
    out.latencyMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now() - t0)
                        .count();

    std::string devLower = lowerAsciiCopy(runner_->deviceName());
    if (out.text.empty()) {
        out.finishReason = "below_confidence";
    } else if (initialRosaMatchLen > 0) {
        out.finishReason =
            devLower + "_rosa_l" + std::to_string(initialRosaMatchLen);
    } else {
        out.finishReason = devLower + "_neural";
    }
    return out;
}

} // namespace fcitx5_llm
