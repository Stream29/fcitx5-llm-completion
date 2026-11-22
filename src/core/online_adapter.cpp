#include "online_adapter.hpp"
#include "text_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>

namespace fcitx5_llm {

namespace {

constexpr uint32_t kAdapterMagic = 0x414F3852u; // "R8OA"
constexpr uint32_t kAdapterVersion = 1u;

struct AdapterFileHeader {
    uint32_t magic = kAdapterMagic;
    uint32_t version = kAdapterVersion;
    int32_t vocabSize = 0;
    int32_t nEmbd = 0;
    uint64_t totalUpdates = 0;
};

uint16_t float32ToFp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(uint32_t));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7BFFu);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                 ((mant + 0x1000u) >> 13));
}

float fp16ToFloat32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

void normalizeVector(const float *src, size_t n, std::vector<float> &dst) {
    dst.resize(n);
    double sumSq = 1e-12;
    for (size_t i = 0; i < n; ++i) {
        sumSq += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    float inv = static_cast<float>(1.0 / std::sqrt(sumSq));
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] * inv;
    }
}

} // namespace

OnlineAdapterManager::OnlineAdapterManager(IRwkvStepRunner *runner,
                                           std::filesystem::path stateFilePath)
    : runner_(runner), statePath_(std::move(stateFilePath)) {
    if (runner_) {
        vocabSize_ = runner_->config().vocabSize;
        nEmbd_ = runner_->config().nEmbd;
    }
    if (const char *envLr = std::getenv("FCITX5_LLM_ONLINE_LR");
        envLr && *envLr) {
        float parsed = std::strtof(envLr, nullptr);
        if (parsed >= 0.0f && parsed <= 5.0f) {
            baseLr_ = parsed;
        }
    }
    if (runner_ && !runner_->hasHardwareOnlineAdapter() && vocabSize_ > 0 &&
        nEmbd_ > 0) {
        cpuWeights_.assign(
            static_cast<size_t>(vocabSize_) * static_cast<size_t>(nEmbd_),
            0.0f);
    }
    loadFromDisk();
}

OnlineAdapterManager::~OnlineAdapterManager() {
    flushAllPending();
    saveIfDirty();
}

void OnlineAdapterManager::applyLogits(const float *xOut, float *logitsInOut) {
    if (!runner_ || vocabSize_ <= 0 || nEmbd_ <= 0) {
        return;
    }
    if (runner_->hasHardwareOnlineAdapter()) {
        runner_->applyAdapterLogits(xOut, logitsInOut);
    } else {
        cpuApplyLogits(xOut, logitsInOut);
    }
}

void OnlineAdapterManager::enqueueTokenSample(const float *prevXOut,
                                              const float *prevBaseLogits,
                                              int32_t targetTok,
                                              float lrScale) {
    if (!runner_ || targetTok < BYTE_OFFSET || targetTok >= vocabSize_ ||
        baseLr_ <= 0.0f) {
        return;
    }
    PendingSample s;
    s.xOut.assign(prevXOut, prevXOut + nEmbd_);
    s.baseLogits.assign(prevBaseLogits, prevBaseLogits + vocabSize_);
    s.targetTok = targetTok;
    s.lrScale = lrScale;
    pending_.push_back(std::move(s));

    while (pending_.size() > kMaturityHorizon) {
        commitSingleSample(pending_.front());
        pending_.pop_front();
    }
}

void OnlineAdapterManager::rollbackPending(int nTokens) {
    int cnt = std::max(1, nTokens);
    while (cnt > 0 && !pending_.empty()) {
        pending_.pop_back();
        --cnt;
    }
}

void OnlineAdapterManager::flushAllPending() {
    while (!pending_.empty()) {
        commitSingleSample(pending_.front());
        pending_.pop_front();
    }
}

void OnlineAdapterManager::penalizeRejectedToken(const float *xOut,
                                                 const float *baseLogits,
                                                 int32_t rejectedTok) {
    if (!runner_ || rejectedTok < BYTE_OFFSET || rejectedTok >= vocabSize_ ||
        rejectLr_ <= 0.0f) {
        return;
    }
    if (runner_->hasHardwareOnlineAdapter()) {
        runner_->updateAdapterStep(xOut, baseLogits, rejectedTok, rejectLr_,
                                   1.0f, true);
    } else {
        cpuUpdateStep(xOut, baseLogits, rejectedTok, rejectLr_, 1.0f, true);
    }
    ++totalUpdates_;
    ++unsavedUpdates_;
    if (unsavedUpdates_ >= 40) {
        saveIfDirty();
    }
}

void OnlineAdapterManager::commitSingleSample(const PendingSample &s) {
    float lr = std::clamp(baseLr_ * s.lrScale, 0.0f, 4.0f);
    if (lr <= 0.0f) {
        return;
    }
    if (runner_->hasHardwareOnlineAdapter()) {
        runner_->updateAdapterStep(s.xOut.data(), s.baseLogits.data(),
                                   s.targetTok, lr, weightDecay_, false);
    } else {
        cpuUpdateStep(s.xOut.data(), s.baseLogits.data(), s.targetTok, lr,
                      weightDecay_, false);
    }
    ++totalUpdates_;
    ++unsavedUpdates_;
    if (unsavedUpdates_ >= 40) {
        saveIfDirty();
    }
}

void OnlineAdapterManager::cpuApplyLogits(const float *xOut,
                                          float *logitsInOut) const {
    if (!cpuNonZero_ || cpuWeights_.empty()) {
        return;
    }
    size_t v = static_cast<size_t>(vocabSize_);
    size_t d = static_cast<size_t>(nEmbd_);
    std::vector<float> hUnit;
    normalizeVector(xOut, d, hUnit);

    for (size_t i = 9; i < v; ++i) {
        const float *row = cpuWeights_.data() + i * d;
        float dot = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            dot += row[j] * hUnit[j];
        }
        logitsInOut[i] += dot;
    }
}

void OnlineAdapterManager::cpuUpdateStep(const float *xOut,
                                         const float *baseLogits,
                                         int32_t tokenIdx, float lr,
                                         float weightDecay,
                                         bool isNegativePush) {
    if (cpuWeights_.empty() || tokenIdx < 9 || tokenIdx >= vocabSize_) {
        return;
    }
    size_t v = static_cast<size_t>(vocabSize_);
    size_t d = static_cast<size_t>(nEmbd_);
    std::vector<float> hUnit;
    normalizeVector(xOut, d, hUnit);

    auto clipRow = [&](float *row) {
        float sq = 1e-12f;
        for (size_t j = 0; j < d; ++j) {
            sq += row[j] * row[j];
        }
        float norm = std::sqrt(sq);
        if (norm > 14.0f) {
            float sc = 14.0f / norm;
            for (size_t j = 0; j < d; ++j) {
                row[j] *= sc;
            }
        }
    };

    if (isNegativePush) {
        float *row = cpuWeights_.data() + static_cast<size_t>(tokenIdx) * d;
        for (size_t j = 0; j < d; ++j) {
            row[j] -= lr * hUnit[j];
        }
        clipRow(row);
        cpuNonZero_ = true;
        return;
    }

    std::vector<float> totalLg(v, -10000.0f);
    float maxLg = -10000.0f;
    for (size_t i = 9; i < v; ++i) {
        float val = std::clamp(baseLogits[i], -10000.0f, 10000.0f);
        if (cpuNonZero_) {
            const float *row = cpuWeights_.data() + i * d;
            float dot = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                dot += row[j] * hUnit[j];
            }
            val += dot;
        }
        totalLg[i] = val;
        if (val > maxLg) {
            maxLg = val;
        }
    }

    std::vector<float> probs(v, 0.0f);
    double sumExp = 0.0;
    for (size_t i = 9; i < v; ++i) {
        float e = std::exp(std::clamp(totalLg[i] - maxLg, -30.0f, 0.0f));
        probs[i] = e;
        sumExp += e;
    }
    float invSum = static_cast<float>(1.0 / std::max(1e-12, sumExp));
    for (size_t i = 9; i < v; ++i) {
        probs[i] *= invSum;
    }

    for (size_t i = 9; i < v; ++i) {
        float target = (static_cast<int32_t>(i) == tokenIdx) ? 1.0f : 0.0f;
        float err = probs[i] - target;
        float *row = cpuWeights_.data() + i * d;
        if (std::fabs(err) < 5e-4f) {
            if (weightDecay < 1.0f && cpuNonZero_) {
                for (size_t j = 0; j < d; ++j) {
                    row[j] *= weightDecay;
                }
            }
            continue;
        }
        float step = lr * err;
        for (size_t j = 0; j < d; ++j) {
            row[j] = row[j] * weightDecay - step * hUnit[j];
        }
        clipRow(row);
    }
    cpuNonZero_ = true;
}

bool OnlineAdapterManager::loadFromDisk() {
    if (statePath_.empty() || !runner_ || vocabSize_ <= 0 || nEmbd_ <= 0) {
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(statePath_, ec)) {
        return false;
    }
    std::ifstream ifs(statePath_, std::ios::binary);
    if (!ifs) {
        return false;
    }
    AdapterFileHeader hdr{};
    ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    if (!ifs || hdr.magic != kAdapterMagic || hdr.version != kAdapterVersion ||
        hdr.vocabSize != vocabSize_ || hdr.nEmbd != nEmbd_) {
        return false;
    }
    size_t count =
        static_cast<size_t>(vocabSize_) * static_cast<size_t>(nEmbd_);
    std::vector<uint16_t> fp16(count);
    ifs.read(reinterpret_cast<char *>(fp16.data()),
             static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!ifs) {
        return false;
    }
    totalUpdates_ = hdr.totalUpdates;
    if (runner_->hasHardwareOnlineAdapter()) {
        return runner_->loadAdapterFp16(fp16.data(), count);
    }
    cpuWeights_.resize(count);
    cpuNonZero_ = false;
    for (size_t i = 0; i < count; ++i) {
        float val = fp16ToFloat32(fp16[i]);
        cpuWeights_[i] = val;
        if (val != 0.0f) {
            cpuNonZero_ = true;
        }
    }
    return true;
}

void OnlineAdapterManager::saveIfDirty() {
    if (unsavedUpdates_ == 0 || statePath_.empty() || !runner_ ||
        vocabSize_ <= 0 || nEmbd_ <= 0) {
        return;
    }
    size_t count =
        static_cast<size_t>(vocabSize_) * static_cast<size_t>(nEmbd_);
    std::vector<uint16_t> fp16(count, 0);
    if (runner_->hasHardwareOnlineAdapter()) {
        if (!runner_->saveAdapterFp16(fp16.data(), count)) {
            return;
        }
    } else {
        if (cpuWeights_.size() != count) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            fp16[i] = float32ToFp16(cpuWeights_[i]);
        }
    }

    std::error_code ec;
    auto parent = statePath_.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        ::chmod(parent.c_str(), 0700);
    }
    std::filesystem::path tmpPath = statePath_.string() + ".tmp";
    FILE *fp = std::fopen(tmpPath.c_str(), "wb");
    if (!fp) {
        return;
    }
    ::chmod(tmpPath.c_str(), 0600);
    AdapterFileHeader hdr{};
    hdr.vocabSize = vocabSize_;
    hdr.nEmbd = nEmbd_;
    hdr.totalUpdates = totalUpdates_;
    bool ok = (std::fwrite(&hdr, sizeof(hdr), 1, fp) == 1) &&
              (std::fwrite(fp16.data(), sizeof(uint16_t), count, fp) == count);
    std::fflush(fp);
    std::fclose(fp);
    if (ok) {
        std::filesystem::rename(tmpPath, statePath_, ec);
        ::chmod(statePath_.c_str(), 0600);
        unsavedUpdates_ = 0;
    } else {
        std::filesystem::remove(tmpPath, ec);
    }
}

// ============================================================================
// DynamicSuffixMemory
// ============================================================================

DynamicSuffixMemory::DynamicSuffixMemory(size_t maxKeys, int32_t vocabSize,
                                         std::filesystem::path stateFilePath)
    : maxKeys_(maxKeys), statePath_(std::move(stateFilePath)),
      unigramCounts_(static_cast<size_t>(std::max(256, vocabSize)), 1.0f),
      totalUnigramMass_(static_cast<double>(std::max(256, vocabSize))) {
    table_.reserve(4096);
    loadFromDisk();
}

DynamicSuffixMemory::~DynamicSuffixMemory() { saveIfDirty(); }

void DynamicSuffixMemory::seedBackgroundUnigrams(
    const PersonalSuffixIndex &bgIndex) {
    bgIndex.accumulateUnigramCounts(unigramCounts_);
    double sum = 0.0;
    for (float c : unigramCounts_) {
        sum += static_cast<double>(std::max(1.0f, c));
    }
    totalUnigramMass_ = std::max(1.0, sum);
}

void DynamicSuffixMemory::setTokenUnigramPrior(int32_t tok, float count) {
    if (tok < 0 || static_cast<size_t>(tok) >= unigramCounts_.size()) {
        return;
    }
    float clamped = std::max(1.0f, count);
    totalUnigramMass_ +=
        static_cast<double>(clamped - unigramCounts_[static_cast<size_t>(tok)]);
    unigramCounts_[static_cast<size_t>(tok)] = clamped;
}

float DynamicSuffixMemory::dirichletPriorStrength(float selfInfoNats) {
    float excess = std::max(0.0f, selfInfoNats - 8.5f);
    return std::max(0.30f, 3.5f * std::exp(-0.20f * excess));
}

float DynamicSuffixMemory::spanSelfInformation(
    std::span<const int32_t> span) const {
    if (span.empty() || unigramCounts_.empty()) {
        return 0.0f;
    }
    double info = 0.0;
    double denom = std::max(1.0, totalUnigramMass_);
    for (int32_t t : span) {
        double c = (t >= 0 && static_cast<size_t>(t) < unigramCounts_.size())
                       ? static_cast<double>(unigramCounts_[static_cast<size_t>(t)])
                       : 1.0;
        double p = std::clamp(c / denom, 1e-9, 0.99);
        info += -std::log(p);
    }
    return static_cast<float>(info);
}

uint64_t DynamicSuffixMemory::hashSpan(std::span<const int32_t> s) {
    uint64_t h = 14695981039346656037ull;
    h ^= static_cast<uint64_t>(s.size());
    h *= 1099511628211ull;
    for (int32_t tok : s) {
        h ^= static_cast<uint64_t>(static_cast<uint32_t>(tok) + 0x9e3779b9u);
        h *= 1099511628211ull;
    }
    return h;
}

void DynamicSuffixMemory::boundary() { liveStack_.clear(); }

void DynamicSuffixMemory::pushToken(int32_t tok, float weight) {
    if (tok < BYTE_OFFSET || weight <= 0.0f) {
        return;
    }
    if (table_.size() >= maxKeys_) {
        for (auto it = table_.begin(); it != table_.end();) {
            for (auto &e : it->second) {
                e.count *= 0.5f;
            }
            std::erase_if(it->second,
                          [](const NextEntry &e) { return e.count < 0.4f; });
            if (it->second.empty()) {
                it = table_.erase(it);
            } else {
                ++it;
            }
        }
    }

    LiveStep step;
    step.tok = tok;
    if (static_cast<size_t>(tok) < unigramCounts_.size()) {
        unigramCounts_[static_cast<size_t>(tok)] += weight;
        totalUnigramMass_ += static_cast<double>(weight);
        step.unigramWeight = weight;
    }

    size_t nPrev = liveStack_.size();
    size_t maxOrder = std::min<size_t>(8, nPrev);
    if (maxOrder >= 2) {
        std::vector<int32_t> ctx(maxOrder);
        for (size_t i = 0; i < maxOrder; ++i) {
            ctx[i] = liveStack_[nPrev - maxOrder + i].tok;
        }
        for (size_t k = 2; k <= maxOrder; ++k) {
            std::span<const int32_t> tail(ctx.data() + (maxOrder - k), k);
            uint64_t key = hashSpan(tail);
            auto &vec = table_[key];
            bool found = false;
            for (auto &e : vec) {
                if (e.tok == tok) {
                    e.count += weight;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (vec.size() >= 6) {
                    auto minIt = std::min_element(
                        vec.begin(), vec.end(),
                        [](const NextEntry &a, const NextEntry &b) {
                            return a.count < b.count;
                        });
                    *minIt = NextEntry{tok, weight};
                } else {
                    vec.push_back(NextEntry{tok, weight});
                }
            }
            step.updates.push_back(AddedRecord{key, tok, weight});
        }
    }

    liveStack_.push_back(std::move(step));
    if (liveStack_.size() > 96) {
        liveStack_.erase(liveStack_.begin(), liveStack_.begin() + 32);
    }
    ++unsavedOps_;
    if (unsavedOps_ >= 120) {
        saveIfDirty();
    }
}

void DynamicSuffixMemory::rollbackBackspace(int nToks) {
    int cnt = std::max(1, nToks);
    while (cnt > 0 && !liveStack_.empty()) {
        const auto &last = liveStack_.back();
        if (last.unigramWeight > 0.0f && last.tok >= 0 &&
            static_cast<size_t>(last.tok) < unigramCounts_.size()) {
            float prev = unigramCounts_[static_cast<size_t>(last.tok)];
            float next = std::max(1.0f, prev - last.unigramWeight);
            totalUnigramMass_ -= static_cast<double>(prev - next);
            unigramCounts_[static_cast<size_t>(last.tok)] = next;
        }
        for (const auto &u : last.updates) {
            auto it = table_.find(u.key);
            if (it == table_.end()) {
                continue;
            }
            for (auto &e : it->second) {
                if (e.tok == u.nextTok) {
                    e.count -= u.weight;
                    break;
                }
            }
            std::erase_if(it->second,
                          [](const NextEntry &e) { return e.count <= 0.05f; });
            if (it->second.empty()) {
                table_.erase(it);
            }
        }
        liveStack_.pop_back();
        --cnt;
    }
    ++unsavedOps_;
}

bool DynamicSuffixMemory::loadFromDisk() {
    if (statePath_.empty()) {
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(statePath_, ec)) {
        return false;
    }
    std::ifstream ifs(statePath_, std::ios::binary);
    if (!ifs) {
        return false;
    }
    uint32_t magic = 0;
    uint32_t ver = 0;
    uint32_t numKeys = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&ver), sizeof(ver));
    ifs.read(reinterpret_cast<char *>(&numKeys), sizeof(numKeys));
    if (!ifs || magic != 0x4D533852u || ver != 1u || numKeys > maxKeys_ * 2) {
        return false;
    }
    table_.clear();
    for (uint32_t i = 0; i < numKeys; ++i) {
        uint64_t key = 0;
        uint16_t len = 0;
        ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
        ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
        if (!ifs || len > 16) {
            table_.clear();
            return false;
        }
        std::vector<NextEntry> entries(len);
        for (uint16_t j = 0; j < len; ++j) {
            ifs.read(reinterpret_cast<char *>(&entries[j].tok), sizeof(int32_t));
            ifs.read(reinterpret_cast<char *>(&entries[j].count), sizeof(float));
        }
        if (!ifs) {
            table_.clear();
            return false;
        }
        table_.emplace(key, std::move(entries));
    }
    unsavedOps_ = 0;
    return true;
}

void DynamicSuffixMemory::saveIfDirty() {
    if (unsavedOps_ == 0 || statePath_.empty()) {
        return;
    }
    std::error_code ec;
    auto parent = statePath_.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        ::chmod(parent.c_str(), 0700);
    }
    std::filesystem::path tmpPath = statePath_.string() + ".tmp";
    FILE *fp = std::fopen(tmpPath.c_str(), "wb");
    if (!fp) {
        return;
    }
    ::chmod(tmpPath.c_str(), 0600);
    uint32_t magic = 0x4D533852u; // "R8SM"
    uint32_t ver = 1u;
    uint32_t numKeys = static_cast<uint32_t>(table_.size());
    bool ok = (std::fwrite(&magic, sizeof(magic), 1, fp) == 1) &&
              (std::fwrite(&ver, sizeof(ver), 1, fp) == 1) &&
              (std::fwrite(&numKeys, sizeof(numKeys), 1, fp) == 1);
    for (const auto &[key, vec] : table_) {
        if (!ok) {
            break;
        }
        uint16_t len = static_cast<uint16_t>(std::min<size_t>(vec.size(), 16));
        ok = ok && (std::fwrite(&key, sizeof(key), 1, fp) == 1) &&
             (std::fwrite(&len, sizeof(len), 1, fp) == 1);
        for (uint16_t j = 0; j < len && ok; ++j) {
            ok = ok &&
                 (std::fwrite(&vec[j].tok, sizeof(int32_t), 1, fp) == 1) &&
                 (std::fwrite(&vec[j].count, sizeof(float), 1, fp) == 1);
        }
    }
    std::fflush(fp);
    std::fclose(fp);
    if (ok) {
        std::filesystem::rename(tmpPath, statePath_, ec);
        ::chmod(statePath_.c_str(), 0600);
        unsavedOps_ = 0;
    } else {
        std::filesystem::remove(tmpPath, ec);
    }
}

float DynamicSuffixMemory::countForSuffix2(int32_t t0, int32_t t1,
                                           int32_t nextTok) const {
    const int32_t arr[2] = {t0, t1};
    auto it = table_.find(hashSpan(arr));
    if (it == table_.end()) {
        return 0.0f;
    }
    for (const auto &e : it->second) {
        if (e.tok == nextTok) {
            return e.count;
        }
    }
    return 0.0f;
}

float DynamicSuffixMemory::empiricalPosterior(
    std::span<const int32_t> tail, int32_t nextTok,
    const PersonalSuffixIndex *bgIndex) const {
    if (tail.empty()) {
        return 0.0f;
    }
    float liveTotal = 0.0f;
    float liveNext = 0.0f;
    size_t fanout = 0;
    auto it = table_.find(hashSpan(tail));
    if (it != table_.end()) {
        fanout = it->second.size();
        for (const auto &e : it->second) {
            liveTotal += e.count;
            if (e.tok == nextTok) {
                liveNext = e.count;
            }
        }
    }
    // If SAM matched this transition from <BEFORE> context before live table recorded it,
    // treat as a single observed count (1.0).
    if (liveNext <= 0.0f) {
        liveNext = 1.0f;
        liveTotal = std::max(1.0f, liveTotal + 1.0f);
        fanout = std::max<size_t>(1, fanout);
    }

    float bgEffTotal = 0.0f;
    float bgEffNext = 0.0f;
    if (bgIndex && bgIndex->enabled()) {
        auto st = bgIndex->lookupSuffixStats(tail, nextTok);
        if (st.totalCount > 0.0f) {
            bgEffTotal = 2.5f * std::log1pf(st.totalCount);
            bgEffNext = bgEffTotal * (st.nextTokCount / st.totalCount);
        }
    }

    float selfInfo = spanSelfInformation(tail);
    float tau = dirichletPriorStrength(selfInfo);
    float fanoutPenalty =
        0.35f * static_cast<float>(fanout > 1 ? (fanout - 1) : 0);
    float denom = liveTotal + bgEffTotal + tau + fanoutPenalty;
    return denom > 0.0f ? (liveNext + bgEffNext) / denom : 0.0f;
}

std::pair<std::vector<std::pair<int32_t, float>>, int32_t>
DynamicSuffixMemory::queryDistribution(std::span<const int32_t> contextToks,
                                       const PersonalSuffixIndex *bgIndex,
                                       int32_t minMatchLen) const {
    if (contextToks.size() < 2) {
        return {{}, 0};
    }
    size_t maxK = std::min<size_t>(8, contextToks.size());
    size_t minK = static_cast<size_t>(std::max(2, minMatchLen));
    if (maxK < minK) {
        return {{}, 0};
    }

    for (size_t k = maxK; k >= minK; --k) {
        std::span<const int32_t> tail =
            contextToks.subspan(contextToks.size() - k, k);
        auto it = table_.find(hashSpan(tail));
        if (it == table_.end() || it->second.empty()) {
            continue;
        }

        float liveTotal = 0.0f;
        for (const auto &e : it->second) {
            liveTotal += e.count;
        }
        if (liveTotal <= 0.0f) {
            continue;
        }

        float selfInfo = spanSelfInformation(tail);
        float tau = dirichletPriorStrength(selfInfo);
        float fanoutPenalty = 0.35f * static_cast<float>(
                                          it->second.size() > 1
                                              ? (it->second.size() - 1)
                                              : 0);

        float bgRawTotal = 0.0f;
        float bgEffTotal = 0.0f;
        if (bgIndex && bgIndex->enabled()) {
            auto st0 = bgIndex->lookupSuffixStats(tail, -1);
            bgRawTotal = st0.totalCount;
            if (bgRawTotal > 0.0f) {
                bgEffTotal = 2.5f * std::log1pf(bgRawTotal);
            }
        }
        float denom = liveTotal + bgEffTotal + tau + fanoutPenalty;

        std::vector<std::pair<int32_t, float>> boosts;
        boosts.reserve(it->second.size());
        for (const auto &e : it->second) {
            float bgEffNext = 0.0f;
            if (bgRawTotal > 0.0f && bgIndex) {
                auto stTok = bgIndex->lookupSuffixStats(tail, e.tok);
                bgEffNext = bgEffTotal * (stTok.nextTokCount / bgRawTotal);
            }
            float posterior = (e.count + bgEffNext) / std::max(1e-6f, denom);
            // Universal information-theoretic posterior gate: filters out high-entropy
            // low-information prefix collisions in ANY language without word lists!
            if (posterior < 0.28f) {
                continue;
            }
            float lenGain = 1.0f + 0.85f * static_cast<float>(k - 2);
            float freqGain = std::log1pf(e.count * 2.0f);
            float boost = std::min(14.0f, 2.85f * lenGain * freqGain * posterior);
            boosts.emplace_back(e.tok, boost);
        }
        if (!boosts.empty()) {
            return {std::move(boosts), static_cast<int32_t>(k)};
        }
    }
    return {{}, 0};
}

size_t DynamicSuffixMemory::warmupFromLog(const std::filesystem::path &logPath,
                                          const PureUnicodeCodec &codec,
                                          size_t maxChars) {
    std::vector<std::filesystem::path> files;
    std::filesystem::path rot1 = logPath.string() + ".1";
    std::error_code ec;
    if (std::filesystem::is_regular_file(rot1, ec)) {
        files.push_back(rot1);
    }
    if (std::filesystem::is_regular_file(logPath, ec)) {
        files.push_back(logPath);
    }
    if (files.empty()) {
        return 0;
    }

    std::vector<std::string> segments;
    std::string curBuf;
    for (const auto &p : files) {
        std::ifstream ifs(p);
        if (!ifs) {
            continue;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        size_t startIdx = lines.size() > 12000 ? lines.size() - 12000 : 0;
        for (size_t i = startIdx; i < lines.size(); ++i) {
            const std::string &ln = lines[i];
            if (ln.find("\"seq\":") == std::string::npos) {
                continue;
            }
            std::string et;
            extractJsonStringField(ln, "event", et);
            if (et == "focus_in" || et == "cursor_jump") {
                if (utf8CharCount(curBuf) >= 2) {
                    segments.push_back(curBuf);
                }
                curBuf.clear();
            } else if (et == "user_type") {
                std::string txtRaw;
                if (extractJsonStringField(ln, "text", txtRaw)) {
                    std::string txt = stripInvisibleFormattingUtf8(txtRaw);
                    if (txt == "\n") {
                        if (utf8CharCount(curBuf) >= 2) {
                            segments.push_back(curBuf);
                        }
                        curBuf.clear();
                    } else {
                        curBuf += txt;
                    }
                }
            } else if (et == "accept_chunk" || et == "accept_word" ||
                       et == "accept_all") {
                std::string accRaw;
                if (extractJsonStringField(ln, "accepted", accRaw)) {
                    curBuf += stripInvisibleFormattingUtf8(accRaw);
                }
            } else if (et == "backspace") {
                popLastUtf8Char(curBuf);
            }
        }
    }
    if (utf8CharCount(curBuf) >= 2) {
        segments.push_back(curBuf);
    }

    size_t fedChars = 0;
    size_t startSeg = 0;
    size_t sumChars = 0;
    for (size_t i = segments.size(); i > 0; --i) {
        sumChars += utf8CharCount(segments[i - 1]);
        if (sumChars >= maxChars) {
            startSeg = i - 1;
            break;
        }
    }
    for (size_t i = startSeg; i < segments.size(); ++i) {
        boundary();
        for (int32_t t : codec.encode(segments[i])) {
            if (t >= BYTE_OFFSET) {
                pushToken(t, 1.0f);
            }
        }
        fedChars += utf8CharCount(segments[i]);
    }
    boundary();
    return fedChars;
}

} // namespace fcitx5_llm
