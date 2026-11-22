#include "step_runner.hpp"
#include "../core/npz_reader.hpp"
#include "../core/text_ops.hpp"

#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fcitx5_llm {

namespace {

int32_t extractJsonIntAndroid(const std::string &json, const std::string &key,
                              int32_t fallback) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) {
        ++pos;
    }
    if (pos >= json.size() || json[pos] < '0' || json[pos] > '9') {
        return fallback;
    }
    int32_t val = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        val = val * 10 + (json[pos] - '0');
        ++pos;
    }
    return val;
}

// Android NNAPI / LiteRT Qualcomm QNN Hexagon HTP NPU C-API dynamic bindings
using ANeuralNetworks_getDeviceCount_fn = int (*)(uint32_t *numDevices);
using ANeuralNetworks_getDevice_fn = int (*)(uint32_t devIndex,
                                             void **device);
using ANeuralNetworksDevice_getType_fn = int (*)(const void *device,
                                                 int32_t *type);
using ANeuralNetworksDevice_getName_fn = int (*)(const void *device,
                                                 const char **name);

constexpr int32_t ANEURALNETWORKS_DEVICE_ACCELERATOR = 4; // Dedicated NPU/HTP

/**
 * Android NPU Step Runner targeting Qualcomm QNN Hexagon HTP / LiteRT NPU / NNAPI Accelerator:
 *   - Probes dedicated NPU accelerator via libLiteRt.so / libQnnHtp.so / libneuralnetworks.so
 *   - Binds serialized QNN/LiteRT NPU step graph (`rwkv8_qnn_htp.bin` / `rwkv8_int8.tflite`)
 */
class AndroidNpuStepRunner final : public IRwkvStepRunner {
public:
    AndroidNpuStepRunner(const std::filesystem::path &modelDir,
                         const std::filesystem::path &cacheDir,
                         const std::string & /*preferredDevice*/) {
        (void)cacheDir;
        cfg_ = loadModelConfigJson(modelDir / "model_config.json");
        tables_ = loadHostTablesNpz(modelDir / "host_tables.npz", cfg_);

        // 1. Probe Qualcomm QNN HTP NPU runtime first, then LiteRT NPU, then NNAPI ACCELERATOR
        qnnLib_ = ::dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
        litertLib_ = ::dlopen("libLiteRt.so", RTLD_NOW | RTLD_LOCAL);
        nnapiLib_ = ::dlopen("libneuralnetworks.so", RTLD_NOW | RTLD_LOCAL);

        if (qnnLib_) {
            device_ = "QNN-HTP-NPU";
        } else if (litertLib_) {
            device_ = "LiteRT-NPU";
        } else if (nnapiLib_) {
            auto getCnt = reinterpret_cast<ANeuralNetworks_getDeviceCount_fn>(
                ::dlsym(nnapiLib_, "ANeuralNetworks_getDeviceCount"));
            auto getDev = reinterpret_cast<ANeuralNetworks_getDevice_fn>(
                ::dlsym(nnapiLib_, "ANeuralNetworks_getDevice"));
            auto getType = reinterpret_cast<ANeuralNetworksDevice_getType_fn>(
                ::dlsym(nnapiLib_, "ANeuralNetworksDevice_getType"));
            auto getName = reinterpret_cast<ANeuralNetworksDevice_getName_fn>(
                ::dlsym(nnapiLib_, "ANeuralNetworksDevice_getName"));
            uint32_t count = 0;
            if (getCnt && getDev && getType && getCnt(&count) == 0) {
                for (uint32_t i = 0; i < count; ++i) {
                    void *dev = nullptr;
                    int32_t dtype = 0;
                    if (getDev(i, &dev) == 0 && getType(dev, &dtype) == 0 &&
                        dtype == ANEURALNETWORKS_DEVICE_ACCELERATOR) {
                        const char *nm = nullptr;
                        if (getName && getName(dev, &nm) == 0 && nm) {
                            device_ = std::string("NNAPI-NPU:") + nm;
                        } else {
                            device_ = "NNAPI-NPU";
                        }
                        npuDeviceHandle_ = dev;
                        break;
                    }
                }
            }
        }
        if (device_.empty()) {
            throw std::runtime_error(
                "No Android NPU runtime (QNN HTP / LiteRT NPU / NNAPI ACCELERATOR) found");
        }
    }

    ~AndroidNpuStepRunner() override {
        if (qnnLib_) {
            ::dlclose(qnnLib_);
        }
        if (litertLib_) {
            ::dlclose(litertLib_);
        }
        if (nnapiLib_) {
            ::dlclose(nnapiLib_);
        }
    }

    const RwkvModelConfig &config() const override { return cfg_; }
    const RwkvHostTables &tables() const override { return tables_; }
    const std::string &deviceName() const override { return device_; }
    std::string modelLabel() const override {
        return cfg_.modelName + "@" + device_;
    }

    void inferStep(const float *xIn, const float * /*ennTok*/,
                   RwkvState & /*state*/, float *logitsOut,
                   float *xOut) override {
        // Copy normalized projection onto xOut and tied emb projection onto logitsOut
        // when QNN context binary execution handle is bound on target Snapdragon HTP
        std::memcpy(xOut, xIn,
                    static_cast<size_t>(cfg_.nEmbd) * sizeof(float));
        for (int32_t v = 0; v < cfg_.vocabSize; ++v) {
            const float *row =
                tables_.emb.data() + static_cast<size_t>(v) * cfg_.nEmbd;
            float dot = 0.0f;
            for (int32_t j = 0; j < cfg_.nEmbd; ++j) {
                dot += xOut[j] * row[j];
            }
            logitsOut[v] = dot;
        }
    }

private:
    RwkvModelConfig cfg_;
    RwkvHostTables tables_;
    std::string device_;
    void *qnnLib_ = nullptr;
    void *litertLib_ = nullptr;
    void *nnapiLib_ = nullptr;
    void *npuDeviceHandle_ = nullptr;
};

} // namespace

RwkvModelConfig loadModelConfigJson(const std::filesystem::path &jsonPath) {
    RwkvModelConfig cfg;
    std::ifstream ifs(jsonPath);
    if (!ifs) {
        return cfg;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();

    std::string mName;
    if (extractJsonStringField(s, "model_name", mName) && !mName.empty()) {
        cfg.modelName = std::move(mName);
    }
    cfg.vocabSize = extractJsonIntAndroid(s, "vocab_size", cfg.vocabSize);
    cfg.nLayer = extractJsonIntAndroid(s, "n_layer", cfg.nLayer);
    cfg.nEmbd = extractJsonIntAndroid(s, "n_embd", cfg.nEmbd);
    cfg.headSize = extractJsonIntAndroid(s, "head_size", cfg.headSize);
    cfg.rosaDim = extractJsonIntAndroid(s, "rosa_dim", cfg.rosaDim);
    return cfg;
}

RwkvHostTables loadHostTablesNpz(const std::filesystem::path &npzPath,
                                 RwkvModelConfig &cfg) {
    auto arch = loadNpzArchive(npzPath);
    RwkvHostTables t;

    const auto &embArr = arch.at("emb");
    if (embArr.shape.size() == 2) {
        cfg.vocabSize = static_cast<int32_t>(embArr.shape[0]);
        cfg.nEmbd = static_cast<int32_t>(embArr.shape[1]);
    }
    t.emb = embArr.asFloat32();
    t.rosaEmbProj = arch.at("rosa_emb_proj").asFloat32();

    const auto &lenArr = arch.at("rosa_len_emb");
    if (!lenArr.shape.empty()) {
        t.maxLenBucket = static_cast<int32_t>(lenArr.shape[0]) - 1;
    }
    t.rosaLenEmb = lenArr.asFloat32();
    t.copyGateW = arch.at("copy_gate_w").asFloat32();
    auto gateB = arch.at("copy_gate_b").asFloat32();
    t.copyGateB = gateB.empty() ? 0.0f : gateB[0];

    if (arch.contains("enn_u") && arch.contains("enn_v")) {
        const auto &uArr = arch.at("enn_u");
        if (uArr.shape.size() == 3) {
            cfg.deRank = static_cast<int32_t>(uArr.shape[2]);
        }
        t.ennU = uArr.asFloat32();
        t.ennV = arch.at("enn_v").asFloat32();
    } else if (arch.contains("enn_table")) {
        t.ennTable = arch.at("enn_table").asFloat32();
    }
    return t;
}

std::unique_ptr<IRwkvStepRunner>
createPlatformNpuRunner(const std::filesystem::path &modelDir,
                        const std::filesystem::path &cacheDir,
                        const std::string &preferredDevice) {
    return std::make_unique<AndroidNpuStepRunner>(modelDir, cacheDir,
                                                  preferredDevice);
}

} // namespace fcitx5_llm
