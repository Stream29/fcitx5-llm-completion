#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace fcitx5_llm {

uint64_t hashTokSuffix(std::span<const int32_t> toks);

struct CorpusSuffixStats {
    float totalCount = 0.0f;
    float nextTokCount = 0.0f;
};

class PersonalSuffixIndex {
public:
    PersonalSuffixIndex() = default;
    explicit PersonalSuffixIndex(const std::filesystem::path &npzPath,
                                 float defaultScale = 2.0f);

    bool enabled() const { return enabled_; }
    size_t size() const { return keys_.size(); }

    CorpusSuffixStats lookupSuffixStats(std::span<const int32_t> suffix,
                                        int32_t candidateNextTok) const;

    void accumulateUnigramCounts(std::span<float> unigramOut) const;

    std::pair<std::vector<std::pair<int32_t, float>>, int32_t>
    queryDistribution(std::span<const int32_t> recentCharToks,
                      int32_t minLen = 2) const;

private:
    bool enabled_ = false;
    float scale_ = 2.0f;
    std::vector<uint64_t> keys_;
    std::vector<uint16_t> topToks_;  // flattened [N * 3]
    std::vector<uint8_t> topProbs_;  // flattened [N * 3]
    std::vector<uint16_t> counts_;   // [N]
};

} // namespace fcitx5_llm
