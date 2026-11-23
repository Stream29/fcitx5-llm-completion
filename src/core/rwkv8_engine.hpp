#pragma once

#include "../backends/step_runner.hpp"
#include "online_adapter.hpp"
#include "online_sam.hpp"
#include "suffix_index.hpp"
#include "unicode_codec.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace fcitx5_llm {

struct InteractionEvent {
    std::string event;
    std::string program;
    std::string prefix;
    std::string suffix;
    std::string text;
    std::string accepted;
    std::string remaining;
};

struct SuggestParams {
    std::string program;
    std::string prefix;
    std::string suffix;
    int32_t maxTokens = 10;
};

struct SuggestOutput {
    bool cancelled = false;
    std::string text;
    std::string rawText;
    std::vector<std::string> chunks;
    std::string model;
    std::string mode;
    int64_t ttftMs = 0;
    int64_t latencyMs = 0;
    int32_t rosaMatchLen = 0;
    std::string finishReason;
};

class Rwkv8CompletionEngine {
public:
    Rwkv8CompletionEngine(std::unique_ptr<IRwkvStepRunner> runner,
                          const std::filesystem::path &modelDir,
                          const std::filesystem::path &warmupLogPath = {},
                          const std::filesystem::path &adapterStatePath = {});

    std::string modelLabel() const { return runner_->modelLabel(); }
    const std::string &deviceName() const { return runner_->deviceName(); }
    size_t warmedChars() const { return warmedChars_; }
    uint64_t onlineUpdates() const { return onlineAdapter_->totalUpdates(); }
    bool hasHardwareOnlineAdapter() const {
        return runner_->hasHardwareOnlineAdapter();
    }
    void flushOnlineLearning() {
        onlineAdapter_->flushAllPending();
        onlineAdapter_->saveIfDirty();
        dynSuffix_.saveIfDirty();
    }

    void handleEvent(const InteractionEvent &ev,
                     const std::function<bool()> &shouldAbort = {});
    bool hasPendingContext() const { return hasPendingAnchor_; }
    bool prewarmPendingContext(const std::function<bool()> &shouldAbort = {});
    SuggestOutput suggest(const SuggestParams &params,
                          const std::function<bool()> &shouldAbort = {});

private:
    bool syncContextWindow(std::string_view prefix, std::string_view sufHead,
                           const std::function<bool()> &shouldAbort);
    void prepareXIn(int32_t tokId, int32_t rosaNt, int32_t rosaMl, float *xOut,
                    float *ennOut) const;
    void stepMainToken(int32_t tokId, int32_t rosaNt, int32_t rosaMl);
    void feedCommittedTokens(std::span<const int32_t> toks,
                             float trainLrScale = 0.0f,
                             float dynWeight = 0.0f);
    void feedRejectedProposal(std::string_view rejectedText);

    std::unique_ptr<IRwkvStepRunner> runner_;
    PureUnicodeCodec codec_;
    EventAwareOnlineSAM sam_;
    DynamicSuffixMemory dynSuffix_;
    PersonalSuffixIndex personalIndex_;
    std::unique_ptr<OnlineAdapterManager> onlineAdapter_;
    RwkvState mainState_;

    std::string lastProgram_;
    std::string lastSyncedTail_;
    std::string lastSyncedSuffixHead_;
    bool hasSyncedTail_ = false;
    bool hasSyncedSuffix_ = false;
    std::string pendingAnchorPrefix_;
    std::string pendingAnchorSuffix_;
    bool hasPendingAnchor_ = false;
    bool inTypingRun_ = false;
    bool hasPrevPredState_ = false;
    int32_t prevRosaNt_ = -1;
    int32_t prevRosaMl_ = 0;
    bool hasLastSuggestStep0_ = false;
    std::vector<float> lastSuggestStep0XOut_;
    std::vector<float> lastSuggestStep0BaseLogits_;
    std::vector<std::string> recentlyRejected_;
    size_t warmedChars_ = 0;

    mutable std::vector<float> tmpXIn_;
    mutable std::vector<float> tmpEnn_;
    mutable std::vector<float> tmpLogits_;
    mutable std::vector<float> tmpXOut_;
};

} // namespace fcitx5_llm
