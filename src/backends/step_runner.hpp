#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fcitx5_llm {

struct RwkvModelConfig {
    std::string modelName = "rwkv8-unicode-37m-int8";
    int32_t vocabSize = 4096;
    int32_t nLayer = 10;
    int32_t nEmbd = 512;
    int32_t headSize = 64;
    int32_t rosaDim = 192;
    int32_t deRank = 48;

    int32_t nHead() const { return nEmbd / headSize; }
    int32_t rosaHeads() const { return rosaDim / 4; }
};

struct RwkvHostTables {
    std::vector<float> emb;          // [V * D]
    std::vector<float> rosaEmbProj;  // [D * D]
    std::vector<float> rosaLenEmb;   // [maxBuckets * D]
    int32_t maxLenBucket = 24;
    std::vector<float> copyGateW;    // [D]
    float copyGateB = 0.0f;

    // Either factorized DeepEmbed (ennU [L * V * R], ennV [L * R * D]) or full ennTable [L * V * D]
    std::vector<float> ennU;
    std::vector<float> ennV;
    std::vector<float> ennTable;
};

struct RwkvState {
    std::vector<float> ap;  // [L * D]
    std::vector<float> fp;  // [L * D]
    std::vector<float> rp;  // [L * D]
    std::vector<float> st;  // [L * H * HS * HS]
    std::vector<float> kr;  // [L * RH * 8 * 4]
    std::vector<float> vr;  // [L * RH * 8 * 4]

    static RwkvState createZero(const RwkvModelConfig &cfg) {
        RwkvState s;
        size_t ld = static_cast<size_t>(cfg.nLayer) * cfg.nEmbd;
        size_t stSz = static_cast<size_t>(cfg.nLayer) * cfg.nHead() *
                      cfg.headSize * cfg.headSize;
        size_t ringSz =
            static_cast<size_t>(cfg.nLayer) * cfg.rosaHeads() * 8 * 4;
        s.ap.assign(ld, 0.0f);
        s.fp.assign(ld, 0.0f);
        s.rp.assign(ld, 0.0f);
        s.st.assign(stSz, 0.0f);
        s.kr.assign(ringSz, 1.0f);
        s.vr.assign(ringSz, 1.0f);
        return s;
    }
};

RwkvModelConfig loadModelConfigJson(const std::filesystem::path &jsonPath);
RwkvHostTables loadHostTablesNpz(const std::filesystem::path &npzPath,
                                 RwkvModelConfig &cfg);

class IRwkvStepRunner {
public:
    virtual ~IRwkvStepRunner() = default;

    virtual const RwkvModelConfig &config() const = 0;
    virtual const RwkvHostTables &tables() const = 0;
    virtual const std::string &deviceName() const = 0;
    virtual std::string modelLabel() const = 0;

    virtual void inferStep(const float *xIn, const float *ennTok,
                           RwkvState &state, float *logitsOut,
                           float *xOut) = 0;

    // Optional NPU-resident full-rank FP16 online Test-Time Training adapter
    virtual bool hasHardwareOnlineAdapter() const { return false; }
    virtual void applyAdapterLogits(const float *, float *) {}
    virtual void updateAdapterStep(const float *, const float *, int32_t, float,
                                   float, bool) {}
    virtual bool loadAdapterFp16(const uint16_t *, size_t) { return false; }
    virtual bool saveAdapterFp16(uint16_t *, size_t) const { return false; }
};

std::unique_ptr<IRwkvStepRunner>
createPlatformNpuRunner(const std::filesystem::path &modelDir,
                        const std::filesystem::path &cacheDir,
                        const std::string &preferredDevice = "NPU");

} // namespace fcitx5_llm
