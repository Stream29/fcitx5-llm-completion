#include "suffix_index.hpp"
#include "npz_reader.hpp"
#include "unicode_codec.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace fcitx5_llm {

uint64_t hashTokSuffix(std::span<const int32_t> toks) {
    uint64_t h = 14695981039346656037ULL;
    for (int32_t t : toks) {
        uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(t) + 0x9E37LL);
        h = (h ^ v) * 1099511628211ULL;
    }
    return h;
}

PersonalSuffixIndex::PersonalSuffixIndex(const std::filesystem::path &npzPath,
                                         float defaultScale)
    : scale_(defaultScale) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(npzPath, ec)) {
        return;
    }
    auto archive = loadNpzArchive(npzPath);
    auto itKeys = archive.find("keys");
    auto itToks = archive.find("top_toks");
    auto itProbs = archive.find("top_probs");
    auto itCounts = archive.find("counts");
    if (itKeys == archive.end() || itToks == archive.end() ||
        itProbs == archive.end() || itCounts == archive.end()) {
        return;
    }

    keys_ = itKeys->second.asUint64();
    topToks_ = itToks->second.asUint16();
    topProbs_ = itProbs->second.asUint8();
    counts_ = itCounts->second.asUint16();
    if (auto itSc = archive.find("fusion_scale"); itSc != archive.end()) {
        auto sc = itSc->second.asFloat32();
        if (!sc.empty()) {
            scale_ = sc[0];
        }
    }

    size_t n = keys_.size();
    enabled_ = n > 0 && topToks_.size() >= n * 3 && topProbs_.size() >= n * 3 &&
               counts_.size() >= n;
}

CorpusSuffixStats
PersonalSuffixIndex::lookupSuffixStats(std::span<const int32_t> suffix,
                                       int32_t candidateNextTok) const {
    CorpusSuffixStats stats{};
    if (!enabled_ || suffix.empty() || suffix.size() > 8) {
        return stats;
    }
    int32_t buf[9];
    int32_t length = static_cast<int32_t>(suffix.size());
    buf[0] = length;
    for (int32_t i = 0; i < length; ++i) {
        buf[i + 1] = suffix[static_cast<size_t>(i)];
    }
    uint64_t h = hashTokSuffix(
        std::span<const int32_t>(buf, static_cast<size_t>(length + 1)));
    auto it = std::lower_bound(keys_.begin(), keys_.end(), h);
    if (it == keys_.end() || *it != h) {
        return stats;
    }
    size_t idx = static_cast<size_t>(it - keys_.begin());
    stats.totalCount = static_cast<float>(counts_[idx]);
    for (size_t k = 0; k < 3; ++k) {
        uint8_t qp = topProbs_[idx * 3 + k];
        if (qp > 0 &&
            static_cast<int32_t>(topToks_[idx * 3 + k]) == candidateNextTok) {
            stats.nextTokCount =
                stats.totalCount * (static_cast<float>(qp) / 255.0f);
            break;
        }
    }
    return stats;
}

void PersonalSuffixIndex::accumulateUnigramCounts(
    std::span<float> unigramOut) const {
    if (!enabled_ || unigramOut.empty()) {
        return;
    }
    size_t n = keys_.size();
    for (size_t i = 0; i < n; ++i) {
        float c = static_cast<float>(counts_[i]);
        // Only accumulate lowest-order entries lightly so counts reflect marginal token distribution
        for (size_t k = 0; k < 3; ++k) {
            uint8_t qp = topProbs_[i * 3 + k];
            if (qp == 0) {
                continue;
            }
            uint16_t tid = topToks_[i * 3 + k];
            if (tid < unigramOut.size()) {
                unigramOut[tid] +=
                    0.25f * c * (static_cast<float>(qp) / 255.0f);
            }
        }
    }
}

std::pair<std::vector<std::pair<int32_t, float>>, int32_t>
PersonalSuffixIndex::queryDistribution(std::span<const int32_t> recentCharToks,
                                       int32_t minLen) const {
    if (!enabled_ || static_cast<int32_t>(recentCharToks.size()) < minLen) {
        return {{}, 0};
    }
    int32_t maxL = std::min<int32_t>(
        8, static_cast<int32_t>(recentCharToks.size()));
    int32_t bestL = 0;
    int32_t matchedOrders = 0;
    std::unordered_map<int32_t, float> accum;
    int32_t buf[9];

    for (int32_t length = maxL; length >= minLen; --length) {
        buf[0] = length;
        auto sub = recentCharToks.subspan(recentCharToks.size() - length,
                                          static_cast<size_t>(length));
        for (int32_t i = 0; i < length; ++i) {
            buf[i + 1] = sub[static_cast<size_t>(i)];
        }
        uint64_t h = hashTokSuffix(
            std::span<const int32_t>(buf, static_cast<size_t>(length + 1)));
        auto it = std::lower_bound(keys_.begin(), keys_.end(), h);
        if (it != keys_.end() && *it == h) {
            size_t idx = static_cast<size_t>(it - keys_.begin());
            float cnt = static_cast<float>(counts_[idx]);
            float prior = (length >= 4) ? 0.35f : (length == 3 ? 0.9f : 2.2f);
            float conf = cnt / (cnt + prior);
            float orderWeight =
                (0.75f + 0.60f * static_cast<float>(length - 1)) *
                std::pow(0.45f, static_cast<float>(matchedOrders));
            for (size_t k = 0; k < 3; ++k) {
                uint8_t qp = topProbs_[idx * 3 + k];
                if (qp == 0) {
                    continue;
                }
                int32_t tid = static_cast<int32_t>(topToks_[idx * 3 + k]);
                if (tid >= BYTE_OFFSET) {
                    accum[tid] +=
                        scale_ * orderWeight * conf * (static_cast<float>(qp) / 255.0f);
                }
            }
            if (bestL == 0) {
                bestL = length;
            }
            ++matchedOrders;
            if (matchedOrders >= 2) {
                break;
            }
        }
    }

    std::vector<std::pair<int32_t, float>> out;
    out.reserve(accum.size());
    for (const auto &kv : accum) {
        out.push_back(kv);
    }
    return {std::move(out), bestL};
}

} // namespace fcitx5_llm
