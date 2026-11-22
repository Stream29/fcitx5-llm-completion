#pragma once

#include "../backends/step_runner.hpp"
#include "suffix_index.hpp"
#include "unicode_codec.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fcitx5_llm {

class OnlineAdapterManager {
public:
    OnlineAdapterManager(IRwkvStepRunner *runner,
                         std::filesystem::path stateFilePath);
    ~OnlineAdapterManager();

    OnlineAdapterManager(const OnlineAdapterManager &) = delete;
    OnlineAdapterManager &operator=(const OnlineAdapterManager &) = delete;

    void applyLogits(const float *xOut, float *logitsInOut);

    void enqueueTokenSample(const float *prevXOut, const float *prevBaseLogits,
                            int32_t targetTok, float lrScale = 1.0f);

    void rollbackPending(int nTokens);
    void flushAllPending();

    void penalizeRejectedToken(const float *xOut, const float *baseLogits,
                               int32_t rejectedTok);

    void saveIfDirty();
    uint64_t totalUpdates() const { return totalUpdates_; }
    float baseLearningRate() const { return baseLr_; }

private:
    struct PendingSample {
        std::vector<float> xOut;
        std::vector<float> baseLogits;
        int32_t targetTok = -1;
        float lrScale = 1.0f;
    };

    void commitSingleSample(const PendingSample &s);
    void cpuApplyLogits(const float *xOut, float *logitsInOut) const;
    void cpuUpdateStep(const float *xOut, const float *baseLogits,
                       int32_t tokenIdx, float lr, float weightDecay,
                       bool isNegativePush);
    bool loadFromDisk();

    IRwkvStepRunner *runner_ = nullptr;
    std::filesystem::path statePath_;
    int32_t vocabSize_ = 0;
    int32_t nEmbd_ = 0;

    float baseLr_ = 1.45f;
    float rejectLr_ = 1.60f;
    float weightDecay_ = 0.99995f;

    static constexpr size_t kMaturityHorizon = 4;
    std::deque<PendingSample> pending_;

    // Used only when runner_->hasHardwareOnlineAdapter() is false
    std::vector<float> cpuWeights_;
    bool cpuNonZero_ = false;

    uint64_t totalUpdates_ = 0;
    uint64_t unsavedUpdates_ = 0;
};

class DynamicSuffixMemory {
public:
    explicit DynamicSuffixMemory(size_t maxKeys = 32768,
                                 int32_t vocabSize = 4096,
                                 std::filesystem::path stateFilePath = {});
    ~DynamicSuffixMemory();

    void seedBackgroundUnigrams(const PersonalSuffixIndex &bgIndex);
    void setTokenUnigramPrior(int32_t tok, float count);

    void boundary();
    void pushToken(int32_t tok, float weight = 1.0f);
    void rollbackBackspace(int nToks);
    void saveIfDirty();

    size_t warmupFromLog(const std::filesystem::path &logPath,
                         const PureUnicodeCodec &codec,
                         size_t maxChars = 48000);

    float spanSelfInformation(std::span<const int32_t> span) const;
    float empiricalPosterior(std::span<const int32_t> tail, int32_t nextTok,
                             const PersonalSuffixIndex *bgIndex = nullptr) const;
    float countForSuffix2(int32_t t0, int32_t t1, int32_t nextTok) const;

    std::pair<std::vector<std::pair<int32_t, float>>, int32_t>
    queryDistribution(std::span<const int32_t> contextToks,
                      const PersonalSuffixIndex *bgIndex = nullptr,
                      int32_t minMatchLen = 2) const;

private:
    struct NextEntry {
        int32_t tok = -1;
        float count = 0.0f;
    };
    struct AddedRecord {
        uint64_t key = 0;
        int32_t nextTok = -1;
        float weight = 0.0f;
    };
    struct LiveStep {
        int32_t tok = -1;
        float unigramWeight = 0.0f;
        std::vector<AddedRecord> updates;
    };

    static uint64_t hashSpan(std::span<const int32_t> s);
    static float dirichletPriorStrength(float selfInfoNats);
    bool loadFromDisk();

    size_t maxKeys_ = 32768;
    std::filesystem::path statePath_;
    std::vector<float> unigramCounts_;
    double totalUnigramMass_ = 4096.0;
    std::unordered_map<uint64_t, std::vector<NextEntry>> table_;
    std::vector<LiveStep> liveStack_;
    uint64_t unsavedOps_ = 0;
};

} // namespace fcitx5_llm
