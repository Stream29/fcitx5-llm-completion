#include "step_runner.hpp"
#include "../core/npz_reader.hpp"
#include "../core/text_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__ANDROID__)
#include <android/log.h>
#define APU_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "fcitx5-llm-apu", __VA_ARGS__)
#define APU_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "fcitx5-llm-apu", __VA_ARGS__)
#else
#include <cstdio>
#define APU_LOGI(...) do { std::fprintf(stderr, "[APU INFO] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#define APU_LOGW(...) do { std::fprintf(stderr, "[APU WARN] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

namespace fcitx5_llm {

namespace {

int32_t extractJsonIntMobile(const std::string &json, const std::string &key,
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

inline float fastSigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -30.0f, 30.0f)));
}

struct QuantMatI8 {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<int8_t> q;       // [rows * cols], row-major
    std::vector<float> rowScale; // [rows], strictly positive
};

QuantMatI8 loadQuantMatFromNpz(const NpzArchive &arch,
                               const std::string &baseName) {
    const auto &qArr = arch.at(baseName + "__q8");
    const auto &sArr = arch.at(baseName + "__sc");
    if (qArr.shape.size() != 2) {
        throw std::runtime_error("Expected 2D int8 tensor: " + baseName);
    }
    QuantMatI8 m;
    m.rows = static_cast<int32_t>(qArr.shape[0]);
    m.cols = static_cast<int32_t>(qArr.shape[1]);
    size_t n = static_cast<size_t>(m.rows) * static_cast<size_t>(m.cols);
    if (qArr.rawBytes.size() < n) {
        throw std::runtime_error("Truncated int8 tensor: " + baseName);
    }
    m.q.resize(n);
    std::memcpy(m.q.data(), qArr.rawBytes.data(), n);
    m.rowScale = sArr.asFloat32();
    if (static_cast<int32_t>(m.rowScale.size()) != m.rows) {
        throw std::runtime_error("Scale row mismatch for: " + baseName);
    }
    for (float &sc : m.rowScale) {
        if (!(sc > 1e-8f)) {
            sc = 1e-8f;
        }
    }
    return m;
}

QuantMatI8 quantizeFloatMatrixPerRow(const std::vector<float> &src,
                                     int32_t rows, int32_t cols) {
    QuantMatI8 m;
    m.rows = rows;
    m.cols = cols;
    m.q.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    m.rowScale.resize(static_cast<size_t>(rows));
    for (int32_t r = 0; r < rows; ++r) {
        const float *rowPtr = src.data() + static_cast<size_t>(r) * cols;
        float maxAbs = 1e-6f;
        for (int32_t c = 0; c < cols; ++c) {
            maxAbs = std::max(maxAbs, std::fabs(rowPtr[c]));
        }
        float sc = std::max(maxAbs / 127.0f, 1e-8f);
        float inv = 127.0f / maxAbs;
        m.rowScale[static_cast<size_t>(r)] = sc;
        int8_t *dstPtr = m.q.data() + static_cast<size_t>(r) * cols;
        for (int32_t c = 0; c < cols; ++c) {
            int32_t qi = static_cast<int32_t>(std::nearbyintf(rowPtr[c] * inv));
            dstPtr[c] = static_cast<int8_t>(std::clamp(qi, -127, 127));
        }
    }
    return m;
}

std::vector<float> loadTransposed2D(const NpzArchive &arch,
                                    const std::string &name) {
    const auto &arr = arch.at(name);
    if (arr.shape.size() != 2) {
        throw std::runtime_error("Expected 2D float tensor: " + name);
    }
    size_t rows = arr.shape[0]; // inDim
    size_t cols = arr.shape[1]; // outDim
    auto src = arr.asFloat32();
    std::vector<float> dst(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

struct HeronLayerWeights {
    std::vector<float> ln0W, ln0B; // layer 0 only
    std::vector<float> ln1W, ln1B;
    std::vector<float> ln2W, ln2B;
    std::vector<float> lnxW, lnxB;

    std::vector<float> x_r, x_w, x_k, x_v, x_a, x_g;
    std::vector<float> w0, w1T, w2T;
    int32_t loRank = 48;
    std::vector<float> a0, a1T, a2T;
    std::vector<float> v0, v1T, v2T; // layer > 0
    std::vector<float> g1T, g2T;
    int32_t gRank = 96;
    std::vector<float> k_k, k_a, r_k;

    QuantMatI8 receptance;
    QuantMatI8 key;
    QuantMatI8 value;
    QuantMatI8 output;

    std::vector<float> rosa_xq, rosa_xk, rosa_xv, rosa_scale;
    QuantMatI8 rosa_q, rosa_k, rosa_v, rosa_o;

    std::vector<float> ffn_x_k;
    QuantMatI8 ffn_key, ffn_value;
};

struct LoadedMobileWeights {
    std::filesystem::path blocksPath;
    std::vector<HeronLayerWeights> layers;
    std::vector<float> lnOutW, lnOutB;
    QuantMatI8 head;
    int32_t ffnDim = 1536;
    int32_t maxCols = 384;
};

LoadedMobileWeights loadMobileWeightsBundle(const std::filesystem::path &modelDir,
                                            RwkvModelConfig &cfg,
                                            const RwkvHostTables &tables) {
    LoadedMobileWeights w;
    w.blocksPath = modelDir / "rwkv8_blocks_int8.npz";
    if (!std::filesystem::is_regular_file(w.blocksPath)) {
        w.blocksPath = modelDir / "unicode_37m_int8pack.npz";
    }
    if (!std::filesystem::is_regular_file(w.blocksPath)) {
        throw std::runtime_error(
            "Missing mobile INT8 weights (rwkv8_blocks_int8.npz) in " +
            modelDir.string());
    }

    auto arch = loadNpzArchive(w.blocksPath);
    w.lnOutW = arch.at("ln_out.weight").asFloat32();
    w.lnOutB = arch.at("ln_out.bias").asFloat32();

    w.layers.resize(static_cast<size_t>(cfg.nLayer));
    w.maxCols = cfg.nEmbd;

    for (int32_t l = 0; l < cfg.nLayer; ++l) {
        std::string pre = "blocks." + std::to_string(l) + ".";
        auto &blk = w.layers[static_cast<size_t>(l)];
        if (l == 0) {
            blk.ln0W = arch.at(pre + "ln0.weight").asFloat32();
            blk.ln0B = arch.at(pre + "ln0.bias").asFloat32();
        }
        blk.ln1W = arch.at(pre + "ln1.weight").asFloat32();
        blk.ln1B = arch.at(pre + "ln1.bias").asFloat32();
        blk.ln2W = arch.at(pre + "ln2.weight").asFloat32();
        blk.ln2B = arch.at(pre + "ln2.bias").asFloat32();
        blk.lnxW = arch.at(pre + "ln_x.weight").asFloat32();
        blk.lnxB = arch.at(pre + "ln_x.bias").asFloat32();

        blk.x_r = arch.at(pre + "x_r").asFloat32();
        blk.x_w = arch.at(pre + "x_w").asFloat32();
        blk.x_k = arch.at(pre + "x_k").asFloat32();
        blk.x_v = arch.at(pre + "x_v").asFloat32();
        blk.x_a = arch.at(pre + "x_a").asFloat32();
        blk.x_g = arch.at(pre + "x_g").asFloat32();

        blk.loRank = static_cast<int32_t>(arch.at(pre + "w1").shape[1]);
        blk.w0 = arch.at(pre + "w0").asFloat32();
        blk.w1T = loadTransposed2D(arch, pre + "w1");
        blk.w2T = loadTransposed2D(arch, pre + "w2");

        blk.a0 = arch.at(pre + "a0").asFloat32();
        blk.a1T = loadTransposed2D(arch, pre + "a1");
        blk.a2T = loadTransposed2D(arch, pre + "a2");

        if (l > 0) {
            blk.v0 = arch.at(pre + "v0").asFloat32();
            blk.v1T = loadTransposed2D(arch, pre + "v1");
            blk.v2T = loadTransposed2D(arch, pre + "v2");
        }

        blk.gRank = static_cast<int32_t>(arch.at(pre + "g1").shape[1]);
        blk.g1T = loadTransposed2D(arch, pre + "g1");
        blk.g2T = loadTransposed2D(arch, pre + "g2");

        blk.k_k = arch.at(pre + "k_k").asFloat32();
        blk.k_a = arch.at(pre + "k_a").asFloat32();
        blk.r_k = arch.at(pre + "r_k").asFloat32();

        blk.receptance = loadQuantMatFromNpz(arch, pre + "receptance.weight");
        blk.key = loadQuantMatFromNpz(arch, pre + "key.weight");
        blk.value = loadQuantMatFromNpz(arch, pre + "value.weight");
        blk.output = loadQuantMatFromNpz(arch, pre + "output.weight");

        std::string rpre = pre + "rosa.";
        blk.rosa_xq = arch.at(rpre + "x_q").asFloat32();
        blk.rosa_xk = arch.at(rpre + "x_k").asFloat32();
        blk.rosa_xv = arch.at(rpre + "x_v").asFloat32();
        blk.rosa_scale = arch.at(rpre + "scale").asFloat32();
        blk.rosa_q = loadQuantMatFromNpz(arch, rpre + "q.weight");
        blk.rosa_k = loadQuantMatFromNpz(arch, rpre + "k.weight");
        blk.rosa_v = loadQuantMatFromNpz(arch, rpre + "v.weight");
        blk.rosa_o = loadQuantMatFromNpz(arch, rpre + "o.weight");

        blk.ffn_x_k = arch.at(pre + "ffn_x_k").asFloat32();
        blk.ffn_key = loadQuantMatFromNpz(arch, pre + "ffn_key.weight");
        blk.ffn_value = loadQuantMatFromNpz(arch, pre + "ffn_value.weight");
        w.maxCols = std::max(w.maxCols, blk.ffn_value.cols);
    }

    w.head = quantizeFloatMatrixPerRow(tables.emb, cfg.vocabSize, cfg.nEmbd);
    cfg.rosaDim = w.layers[0].rosa_q.rows;
    w.ffnDim = w.layers[0].ffn_key.rows;
    return w;
}

// ============================================================================
// Tier 1: Direct In-Process MediaTek DLA Runner (`libneuron_runtime.so`)
// ============================================================================

class MediaTekDlaDirectRunner final : public IRwkvStepRunner {
public:
    struct EnvOptions {
        uint32_t deviceKind = 4; // kEnvOptHardware
        int32_t mdlaCoreOption = 0;
        uint8_t cpuThreadNum = 1;
        bool suppressInputConversion = false;
        bool suppressOutputConversion = false;
    };
    struct BufferAttribute {
        int32_t ionFd = -1;
    };
    struct QoSOptions {
        int32_t preference = 2; // SUSTAINED_SPEED
        int32_t priority = 2;   // HIGH
        uint8_t boostValue = 100;
        uint8_t maxBoostValue = 100;
        uint8_t minBoostValue = 100;
        uint16_t deadline = 0;
        uint16_t abortTime = 0;
        int32_t delayedPowerOffTime = 500;
        int32_t powerPolicy = 0;
        int32_t applicationType = 0;
        void *profiledQoSData = nullptr;
    };

    MediaTekDlaDirectRunner(const std::filesystem::path &modelDir,
                            const std::filesystem::path &dlaPath) {
        cfg_ = loadModelConfigJson(modelDir / "model_config.json");
        tables_ = loadHostTablesNpz(modelDir / "host_tables.npz", cfg_);
        nextState_ = RwkvState::createZero(cfg_);

        for (const char *cand : {"libneuron_runtime.so",
                                 "/vendor/lib64/libneuron_runtime.so"}) {
            lib_ = dlopen(cand, RTLD_NOW | RTLD_LOCAL);
            if (lib_) {
                break;
            }
        }
        if (!lib_) {
            throw std::runtime_error("dlopen(libneuron_runtime.so) failed");
        }

        auto fnCreate = reinterpret_cast<int (*)(const EnvOptions *, void **)>(
            dlsym(lib_, "NeuronRuntime_create"));
        auto fnLoad = reinterpret_cast<int (*)(void *, const char *)>(
            dlsym(lib_, "NeuronRuntime_loadNetworkFromFile"));
        auto fnInNum = reinterpret_cast<int (*)(void *, size_t *)>(
            dlsym(lib_, "NeuronRuntime_getInputNumber"));
        auto fnOutNum = reinterpret_cast<int (*)(void *, size_t *)>(
            dlsym(lib_, "NeuronRuntime_getOutputNumber"));
        fnSetIn_ = reinterpret_cast<SetInFn>(dlsym(lib_, "NeuronRuntime_setInput"));
        fnSetOut_ = reinterpret_cast<SetOutFn>(dlsym(lib_, "NeuronRuntime_setOutput"));
        fnInfer_ = reinterpret_cast<InferFn>(dlsym(lib_, "NeuronRuntime_inference"));
        fnRelease_ = reinterpret_cast<ReleaseFn>(dlsym(lib_, "NeuronRuntime_release"));

        if (!fnCreate || !fnLoad || !fnInNum || !fnOutNum || !fnSetIn_ ||
            !fnSetOut_ || !fnInfer_ || !fnRelease_) {
            dlclose(lib_);
            lib_ = nullptr;
            throw std::runtime_error("Missing symbols in libneuron_runtime.so");
        }

        EnvOptions opts{};
        if (fnCreate(&opts, &rt_) != 0 || !rt_) {
            dlclose(lib_);
            lib_ = nullptr;
            throw std::runtime_error("NeuronRuntime_create failed");
        }
        if (fnLoad(rt_, dlaPath.c_str()) != 0) {
            fnRelease_(rt_);
            rt_ = nullptr;
            dlclose(lib_);
            lib_ = nullptr;
            throw std::runtime_error("NeuronRuntime_loadNetworkFromFile failed: " +
                                     dlaPath.string());
        }
        size_t nIn = 0, nOut = 0;
        fnInNum(rt_, &nIn);
        fnOutNum(rt_, &nOut);
        if (nIn != 8 || nOut != 8) {
            fnRelease_(rt_);
            rt_ = nullptr;
            dlclose(lib_);
            lib_ = nullptr;
            throw std::runtime_error("Expected 8 inputs and 8 outputs in DLA");
        }

        // Configure APU boost gear (100) + 500ms burst power hold
        if (auto fnSetQoS =
                reinterpret_cast<int (*)(void *, const QoSOptions *)>(
                    dlsym(lib_, "NeuronRuntime_setQoSOption"))) {
            QoSOptions qos{};
            int qrc = fnSetQoS(rt_, &qos);
            APU_LOGI("NeuronRuntime_setQoSOption(boost=100) -> rc=%d", qrc);
        }

        // Run a 1-step finite sanity check on actual /dev/apusys hardware
        try {
            std::vector<float> testX(static_cast<size_t>(cfg_.nEmbd), 0.01f);
            std::vector<float> testEnn(
                static_cast<size_t>(cfg_.nLayer) * cfg_.nEmbd, 1.0f);
            RwkvState testState = RwkvState::createZero(cfg_);
            std::vector<float> testLogits(static_cast<size_t>(cfg_.vocabSize),
                                          0.0f);
            std::vector<float> testXOut(static_cast<size_t>(cfg_.nEmbd), 0.0f);
            inferStep(testX.data(), testEnn.data(), testState, testLogits.data(),
                      testXOut.data());
            for (int32_t i = 0; i < std::min<int32_t>(64, cfg_.vocabSize); ++i) {
                if (!std::isfinite(testLogits[static_cast<size_t>(i)])) {
                    throw std::runtime_error(
                        "Non-finite output during DLA sanity check");
                }
            }
        } catch (...) {
            fnRelease_(rt_);
            rt_ = nullptr;
            dlclose(lib_);
            lib_ = nullptr;
            throw;
        }
        device_ = "MTK-APU-DLA";
    }

    ~MediaTekDlaDirectRunner() override {
        if (rt_ && fnRelease_) {
            fnRelease_(rt_);
        }
        if (lib_) {
            dlclose(lib_);
        }
    }

    const RwkvModelConfig &config() const override { return cfg_; }
    const RwkvHostTables &tables() const override { return tables_; }
    const std::string &deviceName() const override { return device_; }
    std::string modelLabel() const override {
        return cfg_.modelName + "@" + device_;
    }

    void inferStep(const float *xIn, const float *ennTok, RwkvState &state,
                   float *logitsOut, float *xOut) override {
        const size_t dBytes = static_cast<size_t>(cfg_.nEmbd) * sizeof(float);
        const size_t ldBytes = static_cast<size_t>(cfg_.nLayer) * dBytes;
        const size_t vBytes =
            static_cast<size_t>(cfg_.vocabSize) * sizeof(float);
        BufferAttribute attr{-1};

        fnSetIn_(rt_, 0, xIn, dBytes, attr);
        fnSetIn_(rt_, 1, ennTok, ldBytes, attr);
        fnSetIn_(rt_, 2, state.ap.data(), state.ap.size() * sizeof(float), attr);
        fnSetIn_(rt_, 3, state.fp.data(), state.fp.size() * sizeof(float), attr);
        fnSetIn_(rt_, 4, state.rp.data(), state.rp.size() * sizeof(float), attr);
        fnSetIn_(rt_, 5, state.st.data(), state.st.size() * sizeof(float), attr);
        fnSetIn_(rt_, 6, state.kr.data(), state.kr.size() * sizeof(float), attr);
        fnSetIn_(rt_, 7, state.vr.data(), state.vr.size() * sizeof(float), attr);

        fnSetOut_(rt_, 0, logitsOut, vBytes, attr);
        fnSetOut_(rt_, 1, xOut, dBytes, attr);
        fnSetOut_(rt_, 2, nextState_.ap.data(), nextState_.ap.size() * sizeof(float), attr);
        fnSetOut_(rt_, 3, nextState_.fp.data(), nextState_.fp.size() * sizeof(float), attr);
        fnSetOut_(rt_, 4, nextState_.rp.data(), nextState_.rp.size() * sizeof(float), attr);
        fnSetOut_(rt_, 5, nextState_.st.data(), nextState_.st.size() * sizeof(float), attr);
        fnSetOut_(rt_, 6, nextState_.kr.data(), nextState_.kr.size() * sizeof(float), attr);
        fnSetOut_(rt_, 7, nextState_.vr.data(), nextState_.vr.size() * sizeof(float), attr);

        if (fnInfer_(rt_) != 0) {
            throw std::runtime_error("NeuronRuntime_inference failed");
        }
        state = nextState_;
    }

private:
    using SetInFn = int (*)(void *, uint64_t, const void *, size_t, BufferAttribute);
    using SetOutFn = int (*)(void *, uint64_t, void *, size_t, BufferAttribute);
    using InferFn = int (*)(void *);
    using ReleaseFn = void (*)(void *);

    RwkvModelConfig cfg_;
    RwkvHostTables tables_;
    RwkvState nextState_;
    std::string device_;
    void *lib_ = nullptr;
    void *rt_ = nullptr;
    SetInFn fnSetIn_ = nullptr;
    SetOutFn fnSetOut_ = nullptr;
    InferFn fnInfer_ = nullptr;
    ReleaseFn fnRelease_ = nullptr;
};

// ============================================================================
// Tier 2: On-Device MediaTek APU (`MDLA_6_1` / Neuron v9) Compiler + Disk Cache
//         (`libneuronservice_adapter.mtk.so`, 100% MDLA-native `16A8W` INT8)
// ============================================================================

namespace neuron_abi {

enum OperandCode : int32_t {
    OPD_FLOAT32 = 0,
    OPD_INT32 = 1,
    OPD_TENSOR_FLOAT32 = 3,
    OPD_TENSOR_INT32 = 4,
    OPD_BOOL = 6,
    OPD_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
};

enum OperationCode : int32_t {
    OP_ADD = 0,
    OP_CONCAT = 2,
    OP_DEQUANTIZE = 6,
    OP_FULLY_CONNECTED = 9,
    OP_LOGISTIC = 14,
    OP_MUL = 18,
    OP_RELU = 19,
    OP_RESHAPE = 22,
    OP_TANH = 28,
    OP_DIV = 30,
    OP_MEAN = 31,
    OP_SUB = 36,
    OP_EXP = 49,
    OP_REDUCE_MAX = 77,
    OP_REDUCE_SUM = 80,
    OP_RSQRT = 83,
    OP_SLICE = 86,
    OP_BATCH_MATMUL = 102,
};

struct OperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t *dimensions;
    float scale;
    int32_t zeroPoint;
};

struct SymmPerChannelQuantParams {
    uint32_t channelDim;
    uint32_t scaleCount;
    const float *scales;
};

struct Api {
    void *handle = nullptr;
    int (*getDeviceCount)(uint32_t *) = nullptr;
    int (*getDevice)(uint32_t, void **) = nullptr;
    int (*getDeviceName)(const void *, const char **) = nullptr;
    int (*getDeviceDesc)(const void *, const char **) = nullptr;

    int (*modelCreate)(void **) = nullptr;
    void (*modelFree)(void *) = nullptr;
    int (*addOperand)(void *, const OperandType *) = nullptr;
    int (*setOperandValue)(void *, int32_t, const void *, size_t) = nullptr;
    int (*setPerChannelQuant)(void *, int32_t,
                              const SymmPerChannelQuantParams *) = nullptr;
    int (*addOperation)(void *, int32_t, uint32_t, const uint32_t *, uint32_t,
                        const uint32_t *) = nullptr;
    int (*identifyIO)(void *, uint32_t, const uint32_t *, uint32_t,
                      const uint32_t *) = nullptr;
    int (*relaxFp32ToFp16)(void *, bool) = nullptr;
    int (*modelFinish)(void *) = nullptr;

    int (*compCreate)(void *, void **) = nullptr;
    int (*compCreateForDevices)(void *, const void *const *, uint32_t,
                                void **) = nullptr;
    void (*compFree)(void *) = nullptr;
    int (*compSetPreference)(void *, int32_t) = nullptr;
    int (*compSetOptHint)(void *, uint32_t) = nullptr;
    int (*compSetCaching)(void *, const char *, const uint8_t *) = nullptr;
    int (*compGetSupportedOps)(void *, uint32_t, bool *) = nullptr;
    int (*compGetSize)(void *, size_t *) = nullptr;
    int (*compStore)(void *, void *, size_t) = nullptr;
    int (*compFinish)(void *) = nullptr;

    int (*execCreate)(void *, void **) = nullptr;
    void (*execFree)(void *) = nullptr;
    int (*execSetInput)(void *, int32_t, const OperandType *, const void *,
                        size_t) = nullptr;
    int (*execSetOutput)(void *, int32_t, const OperandType *, void *,
                         size_t) = nullptr;
    int (*execSetBoostHint)(void *, uint8_t) = nullptr;
    int (*execCompute)(void *) = nullptr;

    bool load() {
        std::vector<std::string> candidates;
        if (const char *envLib = std::getenv("FCITX5_LLM_NEURON_LIB");
            envLib && *envLib) {
            candidates.emplace_back(envLib);
        }
        candidates.emplace_back("libneuronservice_adapter.mtk.so");
        candidates.emplace_back("libneuron_adapter.mtk.so");

        for (const auto &name : candidates) {
            handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle) {
                break;
            }
        }
        if (!handle) {
            return false;
        }

        auto sym = [&](auto &fnPtr, const char *symName) {
            fnPtr = reinterpret_cast<std::decay_t<decltype(fnPtr)>>(
                dlsym(handle, symName));
            return fnPtr != nullptr;
        };

        bool ok = sym(getDeviceCount, "Neuron_getDeviceCount") &&
                  sym(getDevice, "Neuron_getDevice") &&
                  sym(getDeviceName, "NeuronDevice_getName") &&
                  sym(modelCreate, "NeuronModel_create") &&
                  sym(modelFree, "NeuronModel_free") &&
                  sym(addOperand, "NeuronModel_addOperand") &&
                  sym(setOperandValue, "NeuronModel_setOperandValue") &&
                  sym(setPerChannelQuant,
                      "NeuronModel_setOperandSymmPerChannelQuantParams") &&
                  sym(addOperation, "NeuronModel_addOperation") &&
                  sym(identifyIO, "NeuronModel_identifyInputsAndOutputs") &&
                  sym(relaxFp32ToFp16,
                      "NeuronModel_relaxComputationFloat32toFloat16") &&
                  sym(modelFinish, "NeuronModel_finish") &&
                  sym(compCreate, "NeuronCompilation_create") &&
                  sym(compCreateForDevices,
                      "NeuronCompilation_createForDevices") &&
                  sym(compFree, "NeuronCompilation_free") &&
                  sym(compSetPreference, "NeuronCompilation_setPreference") &&
                  sym(compFinish, "NeuronCompilation_finish") &&
                  sym(execCreate, "NeuronExecution_create") &&
                  sym(execFree, "NeuronExecution_free") &&
                  sym(execSetInput, "NeuronExecution_setInput") &&
                  sym(execSetOutput, "NeuronExecution_setOutput") &&
                  sym(execCompute, "NeuronExecution_compute");

        // Optional helpers
        sym(getDeviceDesc, "NeuronDevice_getDescription");
        sym(compSetOptHint, "NeuronCompilation_setOptimizationHint");
        sym(compSetCaching, "NeuronCompilation_setCaching");
        sym(compGetSupportedOps, "NeuronCompilation_getSupportedOperations");
        sym(compGetSize, "NeuronCompilation_getCompiledNetworkSize");
        sym(compStore, "NeuronCompilation_storeCompiledNetwork");
        sym(execSetBoostHint, "NeuronExecution_setBoostHint");

        if (!ok) {
            dlclose(handle);
            handle = nullptr;
            return false;
        }
        return true;
    }

    ~Api() {
        if (handle) {
            dlclose(handle);
        }
    }
};

struct GraphBuilder {
    Api &api;
    void *model = nullptr;
    uint32_t nextOpd = 0;
    uint32_t numOps = 0;

    std::vector<std::vector<float>> f32Pool;
    std::vector<std::vector<int32_t>> i32Pool;
    std::vector<std::vector<uint8_t>> u8Pool;

    uint32_t fuseNone = 0;
    uint32_t fuseRelu = 0;
    uint32_t axis0 = 0;
    uint32_t axis1 = 0;
    uint32_t axisLast = 0;
    uint32_t keepDimsI32 = 0;
    uint32_t keepDimsBoolTrue = 0;
    uint32_t keepDimsBoolFalse = 0;
    uint32_t boolFalse = 0;

    explicit GraphBuilder(Api &a) : api(a) {
        if (api.modelCreate(&model) != 0 || !model) {
            throw std::runtime_error("NeuronModel_create failed");
        }
        f32Pool.reserve(512);
        i32Pool.reserve(512);
        u8Pool.reserve(16);

        fuseNone = addScalarI32(0);
        fuseRelu = addScalarI32(1);
        axis0 = addScalarI32(0);
        axis1 = addScalarI32(1);
        axisLast = addConstI32({1}, {-1});
        keepDimsI32 = addScalarI32(1);
        keepDimsBoolTrue = addScalarBool(1);
        keepDimsBoolFalse = addScalarBool(0);
        boolFalse = addScalarBool(0);
    }

    ~GraphBuilder() {
        if (model) {
            api.modelFree(model);
        }
    }

    uint32_t addTensorF32(const std::vector<uint32_t> &dims) {
        OperandType ot{OPD_TENSOR_FLOAT32, static_cast<uint32_t>(dims.size()),
                       dims.data(), 0.0f, 0};
        if (api.addOperand(model, &ot) != 0) {
            throw std::runtime_error("NeuronModel_addOperand(F32) failed");
        }
        return nextOpd++;
    }

    uint32_t addConstF32(const std::vector<uint32_t> &dims,
                         std::vector<float> vals) {
        uint32_t id = addTensorF32(dims);
        f32Pool.push_back(std::move(vals));
        const auto &buf = f32Pool.back();
        if (api.setOperandValue(model, static_cast<int32_t>(id), buf.data(),
                                buf.size() * sizeof(float)) != 0) {
            throw std::runtime_error("NeuronModel_setOperandValue(F32) failed");
        }
        return id;
    }

    uint32_t addConstF32Ref(const std::vector<uint32_t> &dims,
                            const std::vector<float> &vals) {
        uint32_t id = addTensorF32(dims);
        if (api.setOperandValue(model, static_cast<int32_t>(id), vals.data(),
                                vals.size() * sizeof(float)) != 0) {
            throw std::runtime_error(
                "NeuronModel_setOperandValue(F32Ref) failed");
        }
        return id;
    }

    uint32_t addConstI32(const std::vector<uint32_t> &dims,
                         std::vector<int32_t> vals) {
        OperandType ot{OPD_TENSOR_INT32, static_cast<uint32_t>(dims.size()),
                       dims.data(), 0.0f, 0};
        if (api.addOperand(model, &ot) != 0) {
            throw std::runtime_error("NeuronModel_addOperand(I32) failed");
        }
        uint32_t id = nextOpd++;
        i32Pool.push_back(std::move(vals));
        const auto &buf = i32Pool.back();
        if (api.setOperandValue(model, static_cast<int32_t>(id), buf.data(),
                                buf.size() * sizeof(int32_t)) != 0) {
            throw std::runtime_error("NeuronModel_setOperandValue(I32) failed");
        }
        return id;
    }

    uint32_t addScalarI32(int32_t val) {
        OperandType ot{OPD_INT32, 0, nullptr, 0.0f, 0};
        if (api.addOperand(model, &ot) != 0) {
            throw std::runtime_error("NeuronModel_addOperand(sI32) failed");
        }
        uint32_t id = nextOpd++;
        i32Pool.push_back({val});
        if (api.setOperandValue(model, static_cast<int32_t>(id),
                                i32Pool.back().data(), sizeof(int32_t)) != 0) {
            throw std::runtime_error("NeuronModel_setOperandValue(sI32) failed");
        }
        return id;
    }

    uint32_t addScalarBool(uint8_t val) {
        OperandType ot{OPD_BOOL, 0, nullptr, 0.0f, 0};
        if (api.addOperand(model, &ot) != 0) {
            throw std::runtime_error("NeuronModel_addOperand(sBool) failed");
        }
        uint32_t id = nextOpd++;
        u8Pool.push_back({val});
        if (api.setOperandValue(model, static_cast<int32_t>(id),
                                u8Pool.back().data(), sizeof(uint8_t)) != 0) {
            throw std::runtime_error(
                "NeuronModel_setOperandValue(sBool) failed");
        }
        return id;
    }

    void emitOp(int32_t opCode, const std::vector<uint32_t> &ins,
                const std::vector<uint32_t> &outs) {
        if (api.addOperation(model, opCode, static_cast<uint32_t>(ins.size()),
                             ins.data(), static_cast<uint32_t>(outs.size()),
                             outs.data()) != 0) {
            throw std::runtime_error("NeuronModel_addOperation failed for op=" +
                                     std::to_string(opCode));
        }
        ++numOps;
    }

    uint32_t binaryOp(int32_t opCode, uint32_t a, uint32_t b,
                      const std::vector<uint32_t> &outDims,
                      uint32_t fuse = 0) {
        uint32_t y = addTensorF32(outDims);
        emitOp(opCode, {a, b, fuse == 0 ? fuseNone : fuse}, {y});
        return y;
    }

    uint32_t add(uint32_t a, uint32_t b, const std::vector<uint32_t> &dims) {
        return binaryOp(OP_ADD, a, b, dims);
    }
    uint32_t sub(uint32_t a, uint32_t b, const std::vector<uint32_t> &dims) {
        return binaryOp(OP_SUB, a, b, dims);
    }
    uint32_t mul(uint32_t a, uint32_t b, const std::vector<uint32_t> &dims) {
        return binaryOp(OP_MUL, a, b, dims);
    }
    uint32_t div(uint32_t a, uint32_t b, const std::vector<uint32_t> &dims) {
        return binaryOp(OP_DIV, a, b, dims);
    }

    uint32_t unaryOp(int32_t opCode, uint32_t x,
                     const std::vector<uint32_t> &dims) {
        uint32_t y = addTensorF32(dims);
        emitOp(opCode, {x}, {y});
        return y;
    }

    uint32_t reshape(uint32_t x, const std::vector<int32_t> &targetDims) {
        std::vector<uint32_t> uDims(targetDims.size());
        for (size_t i = 0; i < targetDims.size(); ++i) {
            uDims[i] = static_cast<uint32_t>(targetDims[i]);
        }
        uint32_t sh = addConstI32({static_cast<uint32_t>(targetDims.size())},
                                  targetDims);
        uint32_t y = addTensorF32(uDims);
        emitOp(OP_RESHAPE, {x, sh}, {y});
        return y;
    }

    uint32_t slice(uint32_t x, const std::vector<int32_t> &begin,
                   const std::vector<int32_t> &size) {
        std::vector<uint32_t> uDims(size.size());
        for (size_t i = 0; i < size.size(); ++i) {
            uDims[i] = static_cast<uint32_t>(size[i]);
        }
        uint32_t begId =
            addConstI32({static_cast<uint32_t>(begin.size())}, begin);
        uint32_t szId = addConstI32({static_cast<uint32_t>(size.size())}, size);
        uint32_t y = addTensorF32(uDims);
        emitOp(OP_SLICE, {x, begId, szId}, {y});
        return y;
    }

    uint32_t layerNorm2D(uint32_t x, uint32_t rows, uint32_t cols,
                         const std::vector<float> &w,
                         const std::vector<float> &b, float epsVal,
                         uint32_t zeroBiasCol) {
        (void)zeroBiasCol;
        uint32_t mean = addTensorF32({rows, 1});
        emitOp(OP_MEAN, {x, axisLast, keepDimsI32}, {mean});
        uint32_t diff = sub(x, mean, {rows, cols});
        uint32_t sq = mul(diff, diff, {rows, cols});
        uint32_t var = addTensorF32({rows, 1});
        emitOp(OP_MEAN, {sq, axisLast, keepDimsI32}, {var});
        uint32_t epsC = addConstF32({1, 1}, {epsVal});
        uint32_t varEps = add(var, epsC, {rows, 1});
        uint32_t invStd = unaryOp(OP_RSQRT, varEps, {rows, 1});
        uint32_t normed = mul(diff, invStd, {rows, cols});
        uint32_t wId = addConstF32Ref({rows, cols}, w);
        uint32_t bId = addConstF32Ref({rows, cols}, b);
        return add(mul(normed, wId, {rows, cols}), bId, {rows, cols});
    }

    uint32_t linearInt8(uint32_t x, const QuantMatI8 &qm, uint32_t zeroBiasId,
                        bool reluFuse = false) {
        uint32_t outDim = static_cast<uint32_t>(qm.rows);
        uint32_t inDim = static_cast<uint32_t>(qm.cols);
        uint32_t dims[2] = {outDim, inDim};
        OperandType qOt{OPD_TENSOR_QUANT8_SYMM_PER_CHANNEL, 2, dims, 0.0f, 0};
        if (api.addOperand(model, &qOt) != 0) {
            throw std::runtime_error("addOperand(QUANT8_PER_CH) failed");
        }
        uint32_t wQId = nextOpd++;
        if (api.setOperandValue(model, static_cast<int32_t>(wQId), qm.q.data(),
                                qm.q.size()) != 0) {
            throw std::runtime_error("setOperandValue(QUANT8_PER_CH) failed");
        }
        SymmPerChannelQuantParams qp{0, outDim, qm.rowScale.data()};
        if (api.setPerChannelQuant(model, static_cast<int32_t>(wQId), &qp) !=
            0) {
            throw std::runtime_error("setPerChannelQuant failed");
        }
        uint32_t wFId = unaryOp(OP_DEQUANTIZE, wQId, {outDim, inDim});
        uint32_t y = addTensorF32({1, outDim});
        emitOp(OP_FULLY_CONNECTED,
               {x, wFId, zeroBiasId, reluFuse ? fuseRelu : fuseNone}, {y});
        return y;
    }

    uint32_t linearFp32(uint32_t x, const std::vector<float> &wRowMajor,
                        uint32_t outDim, uint32_t inDim, uint32_t biasId) {
        uint32_t wId = addConstF32Ref({outDim, inDim}, wRowMajor);
        uint32_t y = addTensorF32({1, outDim});
        emitOp(OP_FULLY_CONNECTED, {x, wId, biasId, fuseNone}, {y});
        return y;
    }
};

void computeModelCacheToken(const std::filesystem::path &blocksPath,
                            const RwkvModelConfig &cfg, uint8_t outToken[32]) {
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL;
    auto mix = [&](uint64_t v) {
        h1 = (h1 ^ v) * 1099511628211ULL;
        h2 = (h2 ^ (v + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2))) *
             14695981039346656037ULL;
    };
    mix(0x52574B56384D444CULL); // "RWKV8MDL" schema v1
    mix(static_cast<uint64_t>(cfg.vocabSize));
    mix(static_cast<uint64_t>(cfg.nLayer));
    mix(static_cast<uint64_t>(cfg.nEmbd));
    mix(static_cast<uint64_t>(cfg.headSize));
    mix(static_cast<uint64_t>(cfg.rosaDim));
    std::error_code ec;
    mix(static_cast<uint64_t>(std::filesystem::file_size(blocksPath, ec)));
    auto mtime = std::filesystem::last_write_time(blocksPath, ec);
    mix(static_cast<uint64_t>(mtime.time_since_epoch().count()));

    for (int i = 0; i < 4; ++i) {
        mix(static_cast<uint64_t>(i * 0x1000193));
        std::memcpy(outToken + i * 8, (i % 2 == 0) ? &h1 : &h2, 8);
    }
}

} // namespace neuron_abi

class MediaTekNeuronServiceRunner final : public IRwkvStepRunner {
public:
    MediaTekNeuronServiceRunner(const std::filesystem::path &modelDir,
                                const std::filesystem::path &cacheDir,
                                const std::filesystem::path &exportDlaPath = {}) {
        if (!api_.load()) {
            throw std::runtime_error("MediaTek Neuron service adapter not available");
        }

        uint32_t devCnt = 0;
        if (api_.getDeviceCount(&devCnt) != 0 || devCnt == 0) {
            throw std::runtime_error("No MediaTek Neuron devices found");
        }
        void *mdlaDev = nullptr;
        std::string mdlaDesc = "APU";
        for (uint32_t i = 0; i < devCnt; ++i) {
            void *dev = nullptr;
            const char *nm = nullptr;
            if (api_.getDevice(i, &dev) == 0 && dev &&
                api_.getDeviceName(dev, &nm) == 0 && nm) {
                if (std::string_view(nm).find("mdla") !=
                    std::string_view::npos) {
                    mdlaDev = dev;
                    const char *ds = nullptr;
                    if (api_.getDeviceDesc &&
                        api_.getDeviceDesc(dev, &ds) == 0 && ds && *ds) {
                        mdlaDesc = ds;
                    }
                    break;
                }
            }
        }
        if (!mdlaDev) {
            throw std::runtime_error("No mtk-mdla hardware accelerator found");
        }

        cfg_ = loadModelConfigJson(modelDir / "model_config.json");
        tables_ = loadHostTablesNpz(modelDir / "host_tables.npz", cfg_);
        auto weights = loadMobileWeightsBundle(modelDir, cfg_, tables_);
        nextState_ = RwkvState::createZero(cfg_);

        try {
            buildAndCompileStepModel(weights, mdlaDev, cacheDir, exportDlaPath);
            if (!exportDlaPath.empty()) {
                device_ = "MTK-APU-EXPORT(" + mdlaDesc + ")";
                return;
            }

            // Run a 1-step sanity verification to ensure finite APU outputs
            std::vector<float> testX(static_cast<size_t>(cfg_.nEmbd), 0.01f);
            std::vector<float> testEnn(
                static_cast<size_t>(cfg_.nLayer) * cfg_.nEmbd, 1.0f);
            RwkvState testState = RwkvState::createZero(cfg_);
            std::vector<float> testLogits(static_cast<size_t>(cfg_.vocabSize),
                                          0.0f);
            std::vector<float> testXOut(static_cast<size_t>(cfg_.nEmbd), 0.0f);
            inferStep(testX.data(), testEnn.data(), testState, testLogits.data(),
                      testXOut.data());
            for (int32_t i = 0; i < std::min<int32_t>(64, cfg_.vocabSize); ++i) {
                if (!std::isfinite(testLogits[static_cast<size_t>(i)])) {
                    throw std::runtime_error(
                        "Non-finite output detected during MTK APU sanity check");
                }
            }
        } catch (...) {
            cleanupHandles();
            throw;
        }

        device_ = "MTK-APU(" + mdlaDesc + ")";
    }

    ~MediaTekNeuronServiceRunner() override { cleanupHandles(); }

    const RwkvModelConfig &config() const override { return cfg_; }
    const RwkvHostTables &tables() const override { return tables_; }
    const std::string &deviceName() const override { return device_; }
    std::string modelLabel() const override {
        return cfg_.modelName + "@" + device_;
    }

    void inferStep(const float *xIn, const float *ennTok, RwkvState &state,
                   float *logitsOut, float *xOut) override {
        const size_t dBytes = static_cast<size_t>(cfg_.nEmbd) * sizeof(float);
        const size_t ldBytes =
            static_cast<size_t>(cfg_.nLayer) * dBytes;
        const size_t vBytes =
            static_cast<size_t>(cfg_.vocabSize) * sizeof(float);

        api_.execSetInput(exec_, 0, nullptr, xIn, dBytes);
        api_.execSetInput(exec_, 1, nullptr, ennTok, ldBytes);
        api_.execSetInput(exec_, 2, nullptr, state.ap.data(),
                          state.ap.size() * sizeof(float));
        api_.execSetInput(exec_, 3, nullptr, state.fp.data(),
                          state.fp.size() * sizeof(float));
        api_.execSetInput(exec_, 4, nullptr, state.rp.data(),
                          state.rp.size() * sizeof(float));
        api_.execSetInput(exec_, 5, nullptr, state.st.data(),
                          state.st.size() * sizeof(float));
        api_.execSetInput(exec_, 6, nullptr, state.kr.data(),
                          state.kr.size() * sizeof(float));
        api_.execSetInput(exec_, 7, nullptr, state.vr.data(),
                          state.vr.size() * sizeof(float));

        api_.execSetOutput(exec_, 0, nullptr, logitsOut, vBytes);
        api_.execSetOutput(exec_, 1, nullptr, xOut, dBytes);
        api_.execSetOutput(exec_, 2, nullptr, nextState_.ap.data(),
                           nextState_.ap.size() * sizeof(float));
        api_.execSetOutput(exec_, 3, nullptr, nextState_.fp.data(),
                           nextState_.fp.size() * sizeof(float));
        api_.execSetOutput(exec_, 4, nullptr, nextState_.rp.data(),
                           nextState_.rp.size() * sizeof(float));
        api_.execSetOutput(exec_, 5, nullptr, nextState_.st.data(),
                           nextState_.st.size() * sizeof(float));
        api_.execSetOutput(exec_, 6, nullptr, nextState_.kr.data(),
                           nextState_.kr.size() * sizeof(float));
        api_.execSetOutput(exec_, 7, nullptr, nextState_.vr.data(),
                           nextState_.vr.size() * sizeof(float));

        if (api_.execSetBoostHint) {
            api_.execSetBoostHint(exec_, 100);
        }
        if (api_.execCompute(exec_) != 0) {
            throw std::runtime_error("NeuronExecution_compute failed");
        }
        state = nextState_;
    }

private:
    void buildAndCompileStepModel(
        const LoadedMobileWeights &w, void *mdlaDev,
        const std::filesystem::path &cacheDir,
        const std::filesystem::path &exportDlaPath = {}) {
        using namespace neuron_abi;
        GraphBuilder gb(api_);

        const uint32_t L = static_cast<uint32_t>(cfg_.nLayer);
        const uint32_t D = static_cast<uint32_t>(cfg_.nEmbd);
        const uint32_t HS = static_cast<uint32_t>(cfg_.headSize);
        const uint32_t H = D / HS;
        const uint32_t RD = static_cast<uint32_t>(cfg_.rosaDim);
        const uint32_t RH = RD / 4;
        const uint32_t FD = static_cast<uint32_t>(w.ffnDim);
        const uint32_t V = static_cast<uint32_t>(cfg_.vocabSize);
        const uint32_t loRank = static_cast<uint32_t>(w.layers[0].loRank);
        const uint32_t gRank = static_cast<uint32_t>(w.layers[0].gRank);

        // Shared zero-bias vectors & scalar constants
        uint32_t zeroD = gb.addConstF32({D}, std::vector<float>(D, 0.0f));
        uint32_t zeroRD = gb.addConstF32({RD}, std::vector<float>(RD, 0.0f));
        uint32_t zeroFD = gb.addConstF32({FD}, std::vector<float>(FD, 0.0f));
        uint32_t zeroV = gb.addConstF32({V}, std::vector<float>(V, 0.0f));
        uint32_t zeroLo =
            gb.addConstF32({loRank}, std::vector<float>(loRank, 0.0f));
        uint32_t zeroG =
            gb.addConstF32({gRank}, std::vector<float>(gRank, 0.0f));

        uint32_t cOne1D = gb.addConstF32({1, 1}, {1.0f});
        uint32_t cDec = gb.addConstF32({1, 1}, {-0.606531f});
        uint32_t cInvSqrt64 =
            gb.addConstF32({1, 1}, {1.0f / std::sqrt(static_cast<float>(HS))});
        uint32_t cEpsKk = gb.addConstF32({1, 1}, {1e-4f});
        uint32_t cSign64 = gb.addConstF32({1, 1, 1}, {64.0f});
        uint32_t cSign16 = gb.addConstF32({1, 1, 1}, {16.0f});
        uint32_t cThree = gb.addConstF32({1, 1}, {3.0f});
        uint32_t cRosaEps = gb.addConstF32({1, 1}, {1e-3f});
        uint32_t cPrio =
            gb.addConstF32({1, 8}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});

        // 8 Model Inputs
        uint32_t pX = gb.addTensorF32({1, D});
        uint32_t pEnn = gb.addTensorF32({L, D});
        uint32_t pAp = gb.addTensorF32({L, D});
        uint32_t pFp = gb.addTensorF32({L, D});
        uint32_t pRp = gb.addTensorF32({L, D});
        uint32_t pSt = gb.addTensorF32({L, H, HS, HS});
        uint32_t pKr = gb.addTensorF32({L, RH, 8, 4});
        uint32_t pVr = gb.addTensorF32({L, RH, 8, 4});

        auto hardSign = [&](uint32_t z) {
            uint32_t t1 =
                gb.unaryOp(OP_TANH, gb.mul(z, cSign64, {RH, 1, 4}), {RH, 1, 4});
            uint32_t t2 = gb.unaryOp(OP_TANH, gb.mul(t1, cSign16, {RH, 1, 4}),
                                     {RH, 1, 4});
            return gb.unaryOp(OP_TANH, gb.mul(t2, cSign16, {RH, 1, 4}),
                              {RH, 1, 4});
        };

        uint32_t x = gb.layerNorm2D(pX, 1, D, w.layers[0].ln0W,
                                    w.layers[0].ln0B, 1e-4f, zeroD);
        uint32_t vFirst = 0;

        std::vector<uint32_t> newAp, newFp, newRp, newSt, newKr, newVr;
        newAp.reserve(L);
        newFp.reserve(L);
        newRp.reserve(L);
        newSt.reserve(L);
        newKr.reserve(L);
        newVr.reserve(L);

        const int32_t iD = static_cast<int32_t>(D);
        const int32_t iH = static_cast<int32_t>(H);
        const int32_t iHS = static_cast<int32_t>(HS);
        const int32_t iRD = static_cast<int32_t>(RD);
        const int32_t iRH = static_cast<int32_t>(RH);

        for (uint32_t l = 0; l < L; ++l) {
            const auto &blk = w.layers[l];
            const int32_t il = static_cast<int32_t>(l);

            uint32_t h1 =
                gb.layerNorm2D(x, 1, D, blk.ln1W, blk.ln1B, 1e-4f, zeroD);
            uint32_t xp = gb.slice(pAp, {il, 0}, {1, iD});
            newAp.push_back(h1);
            uint32_t xx = gb.sub(xp, h1, {1, D});

            auto tmixShift = [&](const std::vector<float> &mixVec) {
                uint32_t mId = gb.addConstF32Ref({1, D}, mixVec);
                return gb.add(h1, gb.mul(xx, mId, {1, D}), {1, D});
            };
            uint32_t xr = tmixShift(blk.x_r);
            uint32_t xw = tmixShift(blk.x_w);
            uint32_t xk = tmixShift(blk.x_k);
            uint32_t xv = tmixShift(blk.x_v);
            uint32_t xa = tmixShift(blk.x_a);
            uint32_t xg = tmixShift(blk.x_g);

            uint32_t r = gb.linearInt8(xr, blk.receptance, zeroD);

            uint32_t wLo = gb.unaryOp(
                OP_TANH, gb.linearFp32(xw, blk.w1T, loRank, D, zeroLo),
                {1, loRank});
            uint32_t w0Id = gb.addConstF32Ref({D}, blk.w0);
            uint32_t wRaw = gb.linearFp32(wLo, blk.w2T, D, loRank, w0Id);

            uint32_t k = gb.linearInt8(xk, blk.key, zeroD);
            uint32_t v = gb.linearInt8(xv, blk.value, zeroD);

            if (l == 0) {
                vFirst = v;
            } else {
                uint32_t vLo = gb.linearFp32(xv, blk.v1T, loRank, D, zeroLo);
                uint32_t v0Id = gb.addConstF32Ref({D}, blk.v0);
                uint32_t vg = gb.unaryOp(
                    OP_LOGISTIC, gb.linearFp32(vLo, blk.v2T, D, loRank, v0Id),
                    {1, D});
                v = gb.add(v, gb.mul(gb.sub(vFirst, v, {1, D}), vg, {1, D}),
                           {1, D});
            }

            uint32_t aLo = gb.linearFp32(xa, blk.a1T, loRank, D, zeroLo);
            uint32_t a0Id = gb.addConstF32Ref({D}, blk.a0);
            uint32_t a = gb.unaryOp(
                OP_LOGISTIC, gb.linearFp32(aLo, blk.a2T, D, loRank, a0Id),
                {1, D});

            uint32_t gLo = gb.unaryOp(
                OP_LOGISTIC, gb.linearFp32(xg, blk.g1T, gRank, D, zeroG),
                {1, gRank});
            uint32_t g = gb.linearFp32(gLo, blk.g2T, D, gRank, zeroD);

            // Rank-1 WKV-7 State Update
            uint32_t kkId = gb.addConstF32Ref({1, D}, blk.k_k);
            uint32_t kkRaw = gb.reshape(gb.mul(k, kkId, {1, D}), {iH, iHS});
            uint32_t kkSqMean = gb.addTensorF32({H, 1});
            gb.emitOp(OP_MEAN,
                      {gb.mul(kkRaw, kkRaw, {H, HS}), gb.axisLast,
                       gb.keepDimsI32},
                      {kkSqMean});
            uint32_t kkInv = gb.mul(
                gb.unaryOp(OP_RSQRT, gb.add(kkSqMean, cEpsKk, {H, 1}), {H, 1}),
                cInvSqrt64, {H, 1});
            uint32_t kkNorm = gb.mul(kkRaw, kkInv, {H, HS});

            uint32_t kaId = gb.addConstF32Ref({1, D}, blk.k_a);
            uint32_t kMod = gb.add(
                k,
                gb.mul(k, gb.mul(gb.sub(a, cOne1D, {1, D}), kaId, {1, D}),
                       {1, D}),
                {1, D});

            uint32_t wDec = gb.reshape(
                gb.unaryOp(
                    OP_EXP,
                    gb.mul(gb.unaryOp(OP_LOGISTIC, wRaw, {1, D}), cDec, {1, D}),
                    {1, D}),
                {iH, 1, iHS});

            uint32_t stSlice =
                gb.slice(pSt, {il, 0, 0, 0}, {1, iH, iHS, iHS});
            uint32_t st = gb.reshape(stSlice, {iH, iHS, iHS});
            uint32_t kkCol = gb.reshape(kkNorm, {iH, iHS, 1});
            uint32_t uCol = gb.addTensorF32({H, HS, 1});
            gb.emitOp(OP_BATCH_MATMUL,
                      {st, kkCol, gb.boolFalse, gb.boolFalse}, {uCol});

            uint32_t bRow = gb.reshape(
                gb.mul(kkNorm, gb.reshape(a, {iH, iHS}), {H, HS}),
                {iH, 1, iHS});
            uint32_t vCol = gb.reshape(v, {iH, iHS, 1});
            uint32_t kmRow = gb.reshape(kMod, {iH, 1, iHS});

            uint32_t ubOuter = gb.addTensorF32({H, HS, HS});
            gb.emitOp(OP_BATCH_MATMUL,
                      {uCol, bRow, gb.boolFalse, gb.boolFalse}, {ubOuter});
            uint32_t vkmOuter = gb.addTensorF32({H, HS, HS});
            gb.emitOp(OP_BATCH_MATMUL,
                      {vCol, kmRow, gb.boolFalse, gb.boolFalse}, {vkmOuter});

            uint32_t stNext =
                gb.add(gb.sub(gb.mul(st, wDec, {H, HS, HS}),
                              ubOuter, {H, HS, HS}),
                       vkmOuter, {H, HS, HS});
            newSt.push_back(gb.reshape(stNext, {1, iH, iHS, iHS}));

            uint32_t rCol = gb.reshape(r, {iH, iHS, 1});
            uint32_t outWkvCol = gb.addTensorF32({H, HS, 1});
            gb.emitOp(OP_BATCH_MATMUL,
                      {stNext, rCol, gb.boolFalse, gb.boolFalse}, {outWkvCol});
            uint32_t outWkv = gb.reshape(outWkvCol, {iH, iHS});
            uint32_t gn = gb.layerNorm2D(outWkv, H, HS, blk.lnxW, blk.lnxB,
                                         6.4e-4f, zeroD);

            uint32_t rkId = gb.addConstF32Ref({H, HS}, blk.r_k);
            uint32_t rkProd = gb.mul(
                gb.mul(gb.reshape(r, {iH, iHS}), gb.reshape(kMod, {iH, iHS}),
                       {H, HS}),
                rkId, {H, HS});
            uint32_t rkDot = gb.addTensorF32({H, 1});
            gb.emitOp(OP_REDUCE_SUM,
                      {rkProd, gb.axisLast, gb.keepDimsBoolTrue}, {rkDot});

            uint32_t attPre = gb.reshape(
                gb.add(gn, gb.mul(rkDot, gb.reshape(v, {iH, iHS}), {H, HS}),
                       {H, HS}),
                {1, iD});
            uint32_t attOut =
                gb.linearInt8(gb.mul(attPre, g, {1, D}), blk.output, zeroD);

            // Continuous-Exact 4-bit QKV-ROSA (100% MDLA-native)
            uint32_t rxp = gb.slice(pRp, {il, 0}, {1, iD});
            newRp.push_back(h1);
            uint32_t rxx = gb.sub(rxp, h1, {1, D});

            auto rShift = [&](const std::vector<float> &mixVec) {
                uint32_t mId = gb.addConstF32Ref({1, D}, mixVec);
                return gb.add(h1, gb.mul(rxx, mId, {1, D}), {1, D});
            };
            uint32_t rq = gb.reshape(
                gb.linearInt8(rShift(blk.rosa_xq), blk.rosa_q, zeroRD),
                {iRH, 1, 4});
            uint32_t rk = gb.reshape(
                gb.linearInt8(rShift(blk.rosa_xk), blk.rosa_k, zeroRD),
                {iRH, 1, 4});
            uint32_t rv = gb.reshape(
                gb.linearInt8(rShift(blk.rosa_xv), blk.rosa_v, zeroRD),
                {iRH, 1, 4});

            uint32_t qSign = hardSign(rq);
            uint32_t kSign = hardSign(rk);
            uint32_t vSign = hardSign(rv);

            uint32_t kr = gb.reshape(
                gb.slice(pKr, {il, 0, 0, 0}, {1, iRH, 8, 4}), {iRH, 8, 4});
            uint32_t vr = gb.reshape(
                gb.slice(pVr, {il, 0, 0, 0}, {1, iRH, 8, 4}), {iRH, 8, 4});

            uint32_t dots = gb.addTensorF32({RH, 8});
            gb.emitOp(OP_REDUCE_SUM,
                      {gb.mul(qSign, kr, {RH, 8, 4}), gb.axisLast,
                       gb.keepDimsBoolFalse},
                      {dots});
            uint32_t exactHit =
                gb.unaryOp(OP_RELU, gb.sub(dots, cThree, {RH, 8}), {RH, 8});
            uint32_t scores = gb.mul(exactHit, cPrio, {RH, 8});
            uint32_t maxSc = gb.addTensorF32({RH, 1});
            gb.emitOp(OP_REDUCE_MAX,
                      {scores, gb.axisLast, gb.keepDimsBoolTrue}, {maxSc});
            uint32_t isBest = gb.mul(
                gb.unaryOp(
                    OP_RELU,
                    gb.add(gb.sub(scores, maxSc, {RH, 8}), cOne1D, {RH, 8}),
                    {RH, 8}),
                exactHit, {RH, 8});
            uint32_t sumBestRaw = gb.addTensorF32({RH, 1});
            gb.emitOp(OP_REDUCE_SUM,
                      {isBest, gb.axisLast, gb.keepDimsBoolTrue},
                      {sumBestRaw});
            uint32_t weights = gb.reshape(
                gb.div(isBest, gb.add(sumBestRaw, cRosaEps, {RH, 1}), {RH, 8}),
                {iRH, 1, 8});
            uint32_t matchedV3D = gb.addTensorF32({RH, 1, 4});
            gb.emitOp(OP_BATCH_MATMUL,
                      {weights, vr, gb.boolFalse, gb.boolFalse}, {matchedV3D});
            uint32_t rScaleId = gb.addConstF32Ref({1, RD}, blk.rosa_scale);
            uint32_t rosaOut = gb.linearInt8(
                gb.mul(gb.reshape(matchedV3D, {1, iRD}), rScaleId, {1, RD}),
                blk.rosa_o, zeroD);

            uint32_t krTail = gb.slice(kr, {0, 1, 0}, {iRH, 7, 4});
            uint32_t krNext = gb.addTensorF32({RH, 8, 4});
            gb.emitOp(OP_CONCAT, {krTail, kSign, gb.axis1}, {krNext});
            newKr.push_back(gb.reshape(krNext, {1, iRH, 8, 4}));

            uint32_t vrTail = gb.slice(vr, {0, 1, 0}, {iRH, 7, 4});
            uint32_t vrNext = gb.addTensorF32({RH, 8, 4});
            gb.emitOp(OP_CONCAT, {vrTail, vSign, gb.axis1}, {vrNext});
            newVr.push_back(gb.reshape(vrNext, {1, iRH, 8, 4}));

            x = gb.add(x, gb.add(attOut, rosaOut, {1, D}), {1, D});

            // Channel-Mix ReLU^2 FFN * DeepEmbed
            uint32_t h2 =
                gb.layerNorm2D(x, 1, D, blk.ln2W, blk.ln2B, 1e-4f, zeroD);
            uint32_t fxp = gb.slice(pFp, {il, 0}, {1, iD});
            newFp.push_back(h2);
            uint32_t fXkId = gb.addConstF32Ref({1, D}, blk.ffn_x_k);
            uint32_t fxk =
                gb.add(h2, gb.mul(gb.sub(fxp, h2, {1, D}), fXkId, {1, D}),
                       {1, D});
            uint32_t fk = gb.linearInt8(fxk, blk.ffn_key, zeroFD, true);
            uint32_t fv =
                gb.linearInt8(gb.mul(fk, fk, {1, FD}), blk.ffn_value, zeroD);
            uint32_t ennL = gb.slice(pEnn, {il, 0}, {1, iD});
            x = gb.add(x, gb.mul(fv, ennL, {1, D}), {1, D});
        }

        uint32_t xOut =
            gb.layerNorm2D(x, 1, D, w.lnOutW, w.lnOutB, 1e-4f, zeroD);
        uint32_t logitsOut = gb.linearInt8(xOut, w.head, zeroV);

        auto concatAxis0 = [&](std::vector<uint32_t> parts,
                               const std::vector<uint32_t> &outDims) {
            uint32_t outId = gb.addTensorF32(outDims);
            parts.push_back(gb.axis0);
            gb.emitOp(OP_CONCAT, parts, {outId});
            return outId;
        };

        uint32_t apOut = concatAxis0(newAp, {L, D});
        uint32_t fpOut = concatAxis0(newFp, {L, D});
        uint32_t rpOut = concatAxis0(newRp, {L, D});
        uint32_t stOut = concatAxis0(newSt, {L, H, HS, HS});
        uint32_t krOut = concatAxis0(newKr, {L, RH, 8, 4});
        uint32_t vrOut = concatAxis0(newVr, {L, RH, 8, 4});

        uint32_t mIns[8] = {pX, pEnn, pAp, pFp, pRp, pSt, pKr, pVr};
        uint32_t mOuts[8] = {logitsOut, xOut,  apOut, fpOut,
                             rpOut,     stOut, krOut, vrOut};
        if (api_.identifyIO(gb.model, 8, mIns, 8, mOuts) != 0) {
            throw std::runtime_error("NeuronModel_identifyInputsAndOutputs failed");
        }
        api_.relaxFp32ToFp16(gb.model, true);
        if (api_.modelFinish(gb.model) != 0) {
            throw std::runtime_error("NeuronModel_finish failed");
        }

        // Verify hardware MDLA operator coverage
        if (api_.compCreateForDevices && api_.compGetSupportedOps) {
            void *devComp = nullptr;
            const void *devs[1] = {mdlaDev};
            if (api_.compCreateForDevices(gb.model, devs, 1, &devComp) == 0 &&
                devComp) {
                std::vector<uint8_t> suppBuf(gb.numOps, 0);
                if (api_.compGetSupportedOps(
                        devComp, gb.numOps,
                        reinterpret_cast<bool *>(suppBuf.data())) == 0) {
                    uint32_t nSupp = 0;
                    for (uint8_t s : suppBuf) {
                        if (s) {
                            ++nSupp;
                        }
                    }
                    APU_LOGI("MDLA operator coverage: %u / %u ops supported",
                             nSupp, gb.numOps);
                }
                api_.compFree(devComp);
            }
        }

        if (api_.compCreate(gb.model, &comp_) != 0 || !comp_) {
            throw std::runtime_error("NeuronCompilation_create failed");
        }
        api_.compSetPreference(comp_, 2); // NEURON_PREFER_SUSTAINED_SPEED
        if (api_.compSetOptHint) {
            api_.compSetOptHint(comp_, 1); // NEURON_OPTIMIZATION_LOW_LATENCY
        }
        if (api_.compSetCaching && !cacheDir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(cacheDir, ec);
            uint8_t token[32]{};
            computeModelCacheToken(w.blocksPath, cfg_, token);
            api_.compSetCaching(comp_, cacheDir.c_str(), token);
        }

        if (api_.compFinish(comp_) != 0) {
            api_.compFree(comp_);
            comp_ = nullptr;
            throw std::runtime_error("NeuronCompilation_finish failed");
        }

        // Transfer ownership of gb.model to model_ (NeuronModel must outlive NeuronCompilation)
        model_ = gb.model;
        gb.model = nullptr;

        if (!exportDlaPath.empty()) {
            if (!api_.compGetSize || !api_.compStore) {
                throw std::runtime_error(
                    "storeCompiledNetwork symbols unavailable");
            }
            size_t netSz = 0;
            if (api_.compGetSize(comp_, &netSz) != 0 || netSz < 64) {
                throw std::runtime_error("getCompiledNetworkSize failed");
            }
            std::vector<uint8_t> netBuf(netSz);
            if (api_.compStore(comp_, netBuf.data(), netSz) != 0) {
                throw std::runtime_error("storeCompiledNetwork failed");
            }
            const uint8_t magic[4] = {0x1a, 0x7d, 0xe3, 0x3b};
            auto it = std::search(netBuf.begin(), netBuf.end(),
                                  std::begin(magic), std::end(magic));
            if (it == netBuf.end() || std::distance(netBuf.begin(), it) < 12) {
                throw std::runtime_error(
                    "DLA FlatBuffer magic not found in compiled network");
            }
            const uint8_t *dlaStart = &(*it) - 12;
            uint32_t fbSize = 0;
            std::memcpy(&fbSize, dlaStart, 4);
            size_t totalDlaBytes = 4 + static_cast<size_t>(fbSize);
            size_t maxRem = static_cast<size_t>(
                (netBuf.data() + netBuf.size()) - dlaStart);
            if (totalDlaBytes > maxRem) {
                throw std::runtime_error("Truncated DLA payload");
            }
            std::ofstream ofs(exportDlaPath, std::ios::binary);
            if (!ofs) {
                throw std::runtime_error("Cannot open output DLA: " +
                                         exportDlaPath.string());
            }
            ofs.write(reinterpret_cast<const char *>(dlaStart),
                      static_cast<std::streamsize>(totalDlaBytes));
            if (!ofs) {
                throw std::runtime_error("Failed writing output DLA");
            }
            APU_LOGI("Exported pure MDLA .dla: %s (%zu bytes)",
                     exportDlaPath.c_str(), totalDlaBytes);
            return;
        }

        if (api_.execCreate(comp_, &exec_) != 0 || !exec_) {
            throw std::runtime_error("NeuronExecution_create failed");
        }
    }

    void cleanupHandles() noexcept {
        if (exec_) {
            api_.execFree(exec_);
            exec_ = nullptr;
        }
        if (comp_) {
            api_.compFree(comp_);
            comp_ = nullptr;
        }
        if (model_) {
            api_.modelFree(model_);
            model_ = nullptr;
        }
    }

    neuron_abi::Api api_;
    RwkvModelConfig cfg_;
    RwkvHostTables tables_;
    RwkvState nextState_;
    std::string device_;
    void *model_ = nullptr;
    void *comp_ = nullptr;
    void *exec_ = nullptr;
};

// ============================================================================
// Tier 3: Self-Contained ARMv8.2+ SDOT (`vdotq_s32`) INT8 Step Runner Fallback
// ============================================================================

inline float dotF32(const float *a, const float *b, int32_t n) {
#if defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

inline void matVecF32(const std::vector<float> &wRowMajor, int32_t outDim,
                      int32_t inDim, const float *x, float *y) {
    for (int32_t r = 0; r < outDim; ++r) {
        y[r] = dotF32(wRowMajor.data() + static_cast<size_t>(r) * inDim, x,
                      inDim);
    }
}

inline void quantizeActSymI8(const float *x, int32_t n, int8_t *xqOut,
                             float &xScaleOut) {
#if defined(__aarch64__)
    float32x4_t vmax0 = vdupq_n_f32(0.0f);
    int32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vmax0 = vmaxq_f32(vmax0, vabsq_f32(vld1q_f32(x + i)));
    }
    float maxAbs = vmaxvq_f32(vmax0);
    for (; i < n; ++i) {
        maxAbs = std::max(maxAbs, std::fabs(x[i]));
    }
    if (maxAbs < 1e-8f) {
        xScaleOut = 0.0f;
        std::memset(xqOut, 0, static_cast<size_t>(n));
        return;
    }
    xScaleOut = maxAbs / 127.0f;
    float32x4_t vInv = vdupq_n_f32(127.0f / maxAbs);
    i = 0;
    for (; i + 16 <= n; i += 16) {
        int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i), vInv));
        int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 4), vInv));
        int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 8), vInv));
        int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + i + 12), vInv));
        int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        int8x16_t b0 = vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1));
        vst1q_s8(xqOut + i, vmaxq_s8(b0, vdupq_n_s8(-127)));
    }
    float inv = 127.0f / maxAbs;
    for (; i < n; ++i) {
        int32_t qi = static_cast<int32_t>(std::nearbyintf(x[i] * inv));
        xqOut[i] = static_cast<int8_t>(std::clamp(qi, -127, 127));
    }
#else
    float maxAbs = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        maxAbs = std::max(maxAbs, std::fabs(x[i]));
    }
    if (maxAbs < 1e-8f) {
        xScaleOut = 0.0f;
        std::memset(xqOut, 0, static_cast<size_t>(n));
        return;
    }
    xScaleOut = maxAbs / 127.0f;
    float inv = 127.0f / maxAbs;
    for (int32_t i = 0; i < n; ++i) {
        int32_t qi = static_cast<int32_t>(std::nearbyintf(x[i] * inv));
        xqOut[i] = static_cast<int8_t>(std::clamp(qi, -127, 127));
    }
#endif
}

inline void matVecQ8(const QuantMatI8 &w, const float *x, int8_t *xqScratch,
                     float *yOut) {
    const int32_t rows = w.rows;
    const int32_t cols = w.cols;
    float xScale = 0.0f;
    quantizeActSymI8(x, cols, xqScratch, xScale);
    if (xScale == 0.0f) {
        std::fill_n(yOut, rows, 0.0f);
        return;
    }

    for (int32_t r = 0; r < rows; ++r) {
        const int8_t *wRow = w.q.data() + static_cast<size_t>(r) * cols;
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
        int32x4_t s0 = vdupq_n_s32(0);
        int32x4_t s1 = vdupq_n_s32(0);
        int32x4_t s2 = vdupq_n_s32(0);
        int32x4_t s3 = vdupq_n_s32(0);
        int32_t c = 0;
        for (; c + 64 <= cols; c += 64) {
            s0 = vdotq_s32(s0, vld1q_s8(wRow + c), vld1q_s8(xqScratch + c));
            s1 = vdotq_s32(s1, vld1q_s8(wRow + c + 16),
                           vld1q_s8(xqScratch + c + 16));
            s2 = vdotq_s32(s2, vld1q_s8(wRow + c + 32),
                           vld1q_s8(xqScratch + c + 32));
            s3 = vdotq_s32(s3, vld1q_s8(wRow + c + 48),
                           vld1q_s8(xqScratch + c + 48));
        }
        for (; c + 16 <= cols; c += 16) {
            s0 = vdotq_s32(s0, vld1q_s8(wRow + c), vld1q_s8(xqScratch + c));
        }
        int32_t dot = vaddvq_s32(vaddq_s32(vaddq_s32(s0, s1), vaddq_s32(s2, s3)));
        for (; c < cols; ++c) {
            dot += static_cast<int32_t>(wRow[c]) *
                   static_cast<int32_t>(xqScratch[c]);
        }
#else
        int32_t dot = 0;
        for (int32_t c = 0; c < cols; ++c) {
            dot += static_cast<int32_t>(wRow[c]) *
                   static_cast<int32_t>(xqScratch[c]);
        }
#endif
        yOut[r] =
            static_cast<float>(dot) * (w.rowScale[static_cast<size_t>(r)] * xScale);
    }
}

inline void layerNorm1D(const float *x, const std::vector<float> &w,
                        const std::vector<float> &b, int32_t n, float eps,
                        float *out) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]);
    }
    float mean = static_cast<float>(sum / static_cast<double>(n));
    double sqSum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double d = static_cast<double>(x[i] - mean);
        sqSum += d * d;
    }
    float invStd = static_cast<float>(
        1.0 / std::sqrt(sqSum / static_cast<double>(n) + static_cast<double>(eps)));
    for (int32_t i = 0; i < n; ++i) {
        out[i] = (x[i] - mean) * invStd * w[static_cast<size_t>(i)] +
                 b[static_cast<size_t>(i)];
    }
}

class MobileInt8StepRunner final : public IRwkvStepRunner {
public:
    explicit MobileInt8StepRunner(const std::filesystem::path &modelDir) {
        cfg_ = loadModelConfigJson(modelDir / "model_config.json");
        tables_ = loadHostTablesNpz(modelDir / "host_tables.npz", cfg_);
        auto loaded = loadMobileWeightsBundle(modelDir, cfg_, tables_);

        layers_ = std::move(loaded.layers);
        lnOutW_ = std::move(loaded.lnOutW);
        lnOutB_ = std::move(loaded.lnOutB);
        head_ = std::move(loaded.head);
        ffnDim_ = loaded.ffnDim;
        xqScratch_.resize(static_cast<size_t>(loaded.maxCols));

        size_t d = static_cast<size_t>(cfg_.nEmbd);
        x_.resize(d);
        h1_.resize(d);
        h2_.resize(d);
        xr_.resize(d);
        xw_.resize(d);
        xk_.resize(d);
        xv_.resize(d);
        xa_.resize(d);
        xg_.resize(d);
        r_.resize(d);
        wRaw_.resize(d);
        k_.resize(d);
        v_.resize(d);
        vFirst_.resize(d);
        a_.resize(d);
        g_.resize(d);
        attIn_.resize(d);
        attOut_.resize(d);
        rosaOut_.resize(d);
        fv_.resize(d);
        tmpLo_.resize(static_cast<size_t>(layers_[0].loRank));
        tmpG_.resize(static_cast<size_t>(layers_[0].gRank));
        rq_.resize(static_cast<size_t>(cfg_.rosaDim));
        rk_.resize(static_cast<size_t>(cfg_.rosaDim));
        rv_.resize(static_cast<size_t>(cfg_.rosaDim));
        matchedV_.resize(static_cast<size_t>(cfg_.rosaDim));
        fk_.resize(static_cast<size_t>(ffnDim_));

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
        device_ = "ARM64-SDOT-INT8";
#elif defined(__aarch64__)
        device_ = "ARM64-NEON-INT8";
#else
        device_ = "CPU-INT8";
#endif
    }

    const RwkvModelConfig &config() const override { return cfg_; }
    const RwkvHostTables &tables() const override { return tables_; }
    const std::string &deviceName() const override { return device_; }
    std::string modelLabel() const override {
        return cfg_.modelName + "@" + device_;
    }

    void inferStep(const float *xIn, const float *ennTok, RwkvState &state,
                   float *logitsOut, float *xOut) override {
        const int32_t d = cfg_.nEmbd;
        const int32_t hs = cfg_.headSize;
        const int32_t nHead = d / hs;
        const int32_t rHeads = cfg_.rosaDim / 4;

        layerNorm1D(xIn, layers_[0].ln0W, layers_[0].ln0B, d, 1e-5f, x_.data());

        for (int32_t l = 0; l < cfg_.nLayer; ++l) {
            const auto &blk = layers_[static_cast<size_t>(l)];
            layerNorm1D(x_.data(), blk.ln1W, blk.ln1B, d, 1e-5f, h1_.data());

            float *ap = state.ap.data() + static_cast<size_t>(l) * d;
            for (int32_t i = 0; i < d; ++i) {
                float h = h1_[static_cast<size_t>(i)];
                float xx = ap[i] - h;
                ap[i] = h;
                xr_[static_cast<size_t>(i)] = h + xx * blk.x_r[static_cast<size_t>(i)];
                xw_[static_cast<size_t>(i)] = h + xx * blk.x_w[static_cast<size_t>(i)];
                xk_[static_cast<size_t>(i)] = h + xx * blk.x_k[static_cast<size_t>(i)];
                xv_[static_cast<size_t>(i)] = h + xx * blk.x_v[static_cast<size_t>(i)];
                xa_[static_cast<size_t>(i)] = h + xx * blk.x_a[static_cast<size_t>(i)];
                xg_[static_cast<size_t>(i)] = h + xx * blk.x_g[static_cast<size_t>(i)];
            }

            matVecQ8(blk.receptance, xr_.data(), xqScratch_.data(), r_.data());

            matVecF32(blk.w1T, blk.loRank, d, xw_.data(), tmpLo_.data());
            for (int32_t j = 0; j < blk.loRank; ++j) {
                tmpLo_[static_cast<size_t>(j)] =
                    std::tanh(tmpLo_[static_cast<size_t>(j)]);
            }
            matVecF32(blk.w2T, d, blk.loRank, tmpLo_.data(), wRaw_.data());
            for (int32_t i = 0; i < d; ++i) {
                wRaw_[static_cast<size_t>(i)] += blk.w0[static_cast<size_t>(i)];
            }

            matVecQ8(blk.key, xk_.data(), xqScratch_.data(), k_.data());
            matVecQ8(blk.value, xv_.data(), xqScratch_.data(), v_.data());

            if (l == 0) {
                vFirst_ = v_;
            } else {
                matVecF32(blk.v1T, blk.loRank, d, xv_.data(), tmpLo_.data());
                matVecF32(blk.v2T, d, blk.loRank, tmpLo_.data(), attOut_.data());
                for (int32_t i = 0; i < d; ++i) {
                    float vg = fastSigmoid(blk.v0[static_cast<size_t>(i)] +
                                           attOut_[static_cast<size_t>(i)]);
                    float vi = v_[static_cast<size_t>(i)];
                    v_[static_cast<size_t>(i)] =
                        vi + (vFirst_[static_cast<size_t>(i)] - vi) * vg;
                }
            }

            matVecF32(blk.a1T, blk.loRank, d, xa_.data(), tmpLo_.data());
            matVecF32(blk.a2T, d, blk.loRank, tmpLo_.data(), a_.data());
            for (int32_t i = 0; i < d; ++i) {
                a_[static_cast<size_t>(i)] = fastSigmoid(
                    blk.a0[static_cast<size_t>(i)] + a_[static_cast<size_t>(i)]);
            }

            matVecF32(blk.g1T, blk.gRank, d, xg_.data(), tmpG_.data());
            for (int32_t j = 0; j < blk.gRank; ++j) {
                tmpG_[static_cast<size_t>(j)] =
                    fastSigmoid(tmpG_[static_cast<size_t>(j)]);
            }
            matVecF32(blk.g2T, d, blk.gRank, tmpG_.data(), g_.data());

            float kkNorm[64];
            float kMod[64];
            float bVec[64];
            float wDec[64];
            float outWkv[64];

            for (int32_t h = 0; h < nHead; ++h) {
                int32_t ho = h * hs;
                float sq = 1e-6f;
                for (int32_t c = 0; c < hs; ++c) {
                    float kr = k_[ho + c] * blk.k_k[ho + c];
                    kkNorm[c] = kr;
                    sq += kr * kr;
                }
                float invNorm = 1.0f / std::sqrt(sq);
                for (int32_t c = 0; c < hs; ++c) {
                    kkNorm[c] *= invNorm;
                    kMod[c] = k_[ho + c] *
                              (1.0f + (a_[ho + c] - 1.0f) * blk.k_a[ho + c]);
                    bVec[c] = kkNorm[c] * a_[ho + c];
                    wDec[c] = std::exp(-0.606531f * fastSigmoid(wRaw_[ho + c]));
                }

                float *stH = state.st.data() +
                             static_cast<size_t>(l * nHead + h) * hs * hs;
                const float *rH = r_.data() + ho;

                double sumOut = 0.0;
                for (int32_t row = 0; row < hs; ++row) {
                    float *stRow = stH + row * hs;
                    float uRow = dotF32(stRow, kkNorm, hs);
                    float vRowVal = v_[ho + row];
                    float ow = 0.0f;
                    for (int32_t c = 0; c < hs; ++c) {
                        float sNext = stRow[c] * wDec[c] - uRow * bVec[c] +
                                      vRowVal * kMod[c];
                        stRow[c] = sNext;
                        ow += sNext * rH[c];
                    }
                    outWkv[row] = ow;
                    sumOut += static_cast<double>(ow);
                }

                float mean = static_cast<float>(sumOut / hs);
                double varSum = 0.0;
                for (int32_t c = 0; c < hs; ++c) {
                    double df = static_cast<double>(outWkv[c] - mean);
                    varSum += df * df;
                }
                float invStd = static_cast<float>(
                    1.0 / std::sqrt(varSum / hs + 64e-5));

                float rkDot = 0.0f;
                for (int32_t c = 0; c < hs; ++c) {
                    rkDot += rH[c] * kMod[c] * blk.r_k[ho + c];
                }
                for (int32_t c = 0; c < hs; ++c) {
                    float gn = (outWkv[c] - mean) * invStd * blk.lnxW[ho + c] +
                               blk.lnxB[ho + c];
                    attIn_[ho + c] = (gn + rkDot * v_[ho + c]) * g_[ho + c];
                }
            }

            matVecQ8(blk.output, attIn_.data(), xqScratch_.data(),
                     attOut_.data());

            float *rp = state.rp.data() + static_cast<size_t>(l) * d;
            for (int32_t i = 0; i < d; ++i) {
                float h = h1_[static_cast<size_t>(i)];
                float rxx = rp[i] - h;
                rp[i] = h;
                xr_[static_cast<size_t>(i)] =
                    h + rxx * blk.rosa_xq[static_cast<size_t>(i)];
                xk_[static_cast<size_t>(i)] =
                    h + rxx * blk.rosa_xk[static_cast<size_t>(i)];
                xv_[static_cast<size_t>(i)] =
                    h + rxx * blk.rosa_xv[static_cast<size_t>(i)];
            }
            matVecQ8(blk.rosa_q, xr_.data(), xqScratch_.data(), rq_.data());
            matVecQ8(blk.rosa_k, xk_.data(), xqScratch_.data(), rk_.data());
            matVecQ8(blk.rosa_v, xv_.data(), xqScratch_.data(), rv_.data());

            for (int32_t rh = 0; rh < rHeads; ++rh) {
                int32_t bo = rh * 4;
                float qSign[4], kSign[4], vSign[4];
                for (int b = 0; b < 4; ++b) {
                    qSign[b] = (rq_[bo + b] > 0.0f) ? 1.0f : -1.0f;
                    kSign[b] = (rk_[bo + b] > 0.0f) ? 1.0f : -1.0f;
                    vSign[b] = (rv_[bo + b] > 0.0f) ? 1.0f : -1.0f;
                }

                float *krH = state.kr.data() +
                             static_cast<size_t>(l * rHeads + rh) * 32;
                float *vrH = state.vr.data() +
                             static_cast<size_t>(l * rHeads + rh) * 32;

                float exactHit[8];
                float scores[8];
                float maxSc = 0.0f;
                for (int s = 0; s < 8; ++s) {
                    const float *ks = krH + s * 4;
                    float dot = qSign[0] * ks[0] + qSign[1] * ks[1] +
                                qSign[2] * ks[2] + qSign[3] * ks[3];
                    float eh = std::clamp(dot - 3.0f, 0.0f, 1.0f);
                    float sc = eh * static_cast<float>(s + 1);
                    exactHit[s] = eh;
                    scores[s] = sc;
                    if (sc > maxSc) {
                        maxSc = sc;
                    }
                }
                float isBest[8];
                float sumBest = 1e-6f;
                for (int s = 0; s < 8; ++s) {
                    float ib = (scores[s] == maxSc ? 1.0f : 0.0f) * exactHit[s];
                    isBest[s] = ib;
                    sumBest += ib;
                }
                float invBest = 1.0f / sumBest;
                for (int b = 0; b < 4; ++b) {
                    float accV = 0.0f;
                    for (int s = 0; s < 8; ++s) {
                        accV += (isBest[s] * invBest) * vrH[s * 4 + b];
                    }
                    matchedV_[bo + b] = accV * blk.rosa_scale[bo + b];
                }

                std::memmove(krH, krH + 4, 28 * sizeof(float));
                std::memcpy(krH + 28, kSign, 4 * sizeof(float));
                std::memmove(vrH, vrH + 4, 28 * sizeof(float));
                std::memcpy(vrH + 28, vSign, 4 * sizeof(float));
            }

            matVecQ8(blk.rosa_o, matchedV_.data(), xqScratch_.data(),
                     rosaOut_.data());

            for (int32_t i = 0; i < d; ++i) {
                x_[static_cast<size_t>(i)] +=
                    attOut_[static_cast<size_t>(i)] +
                    rosaOut_[static_cast<size_t>(i)];
            }

            layerNorm1D(x_.data(), blk.ln2W, blk.ln2B, d, 1e-5f, h2_.data());
            float *fp = state.fp.data() + static_cast<size_t>(l) * d;
            for (int32_t i = 0; i < d; ++i) {
                float h = h2_[static_cast<size_t>(i)];
                float fxx = fp[i] - h;
                fp[i] = h;
                xk_[static_cast<size_t>(i)] =
                    h + fxx * blk.ffn_x_k[static_cast<size_t>(i)];
            }
            matVecQ8(blk.ffn_key, xk_.data(), xqScratch_.data(), fk_.data());
            for (int32_t j = 0; j < ffnDim_; ++j) {
                float r = std::max(0.0f, fk_[static_cast<size_t>(j)]);
                fk_[static_cast<size_t>(j)] = r * r;
            }
            matVecQ8(blk.ffn_value, fk_.data(), xqScratch_.data(), fv_.data());

            const float *ennL = ennTok + static_cast<size_t>(l) * d;
            for (int32_t i = 0; i < d; ++i) {
                x_[static_cast<size_t>(i)] +=
                    fv_[static_cast<size_t>(i)] * ennL[i];
            }
        }

        layerNorm1D(x_.data(), lnOutW_, lnOutB_, d, 1e-5f, xOut);
        matVecQ8(head_, xOut, xqScratch_.data(), logitsOut);
    }

private:
    RwkvModelConfig cfg_;
    RwkvHostTables tables_;
    std::string device_;
    int32_t ffnDim_ = 1536;

    std::vector<HeronLayerWeights> layers_;
    std::vector<float> lnOutW_, lnOutB_;
    QuantMatI8 head_;

    std::vector<int8_t> xqScratch_;
    std::vector<float> x_, h1_, h2_, xr_, xw_, xk_, xv_, xa_, xg_;
    std::vector<float> r_, wRaw_, k_, v_, vFirst_, a_, g_;
    std::vector<float> attIn_, attOut_, rosaOut_, fv_;
    std::vector<float> tmpLo_, tmpG_, rq_, rk_, rv_, matchedV_, fk_;
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
    cfg.vocabSize = extractJsonIntMobile(s, "vocab_size", cfg.vocabSize);
    cfg.nLayer = extractJsonIntMobile(s, "n_layer", cfg.nLayer);
    cfg.nEmbd = extractJsonIntMobile(s, "n_embd", cfg.nEmbd);
    cfg.headSize = extractJsonIntMobile(s, "head_size", cfg.headSize);
    cfg.rosaDim = extractJsonIntMobile(s, "rosa_dim", cfg.rosaDim);
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
    if (preferredDevice != "CPU") {
        // Tier 1: Precompiled raw MediaTek .dla via direct in-process libneuron_runtime.so
        std::error_code ec;
        auto dlaPath = modelDir / "rwkv8_step_mtk.dla";
        if (std::filesystem::is_regular_file(dlaPath, ec)) {
            try {
                auto runner =
                    std::make_unique<MediaTekDlaDirectRunner>(modelDir, dlaPath);
                APU_LOGI("Loaded direct MediaTek DLA runner: %s",
                         runner->modelLabel().c_str());
                return runner;
            } catch (const std::exception &e) {
                APU_LOGW("Direct DLA runner unavailable (%s), falling back",
                         e.what());
            }
        }

        // Tier 2: Explicit on-device MediaTek APU compilation (when FCITX5_LLM_ANDROID_APU=1 or preferredDevice=="MTK-APU")
        const char *forceApu = std::getenv("FCITX5_LLM_ANDROID_APU");
        if (preferredDevice == "MTK-APU" ||
            (forceApu && std::string_view(forceApu) == "1") ||
            std::getenv("FCITX5_LLM_NEURON_LIB") != nullptr) {
            try {
                auto runner = std::make_unique<MediaTekNeuronServiceRunner>(
                    modelDir, cacheDir);
                APU_LOGI("Loaded MediaTek APU runner: %s",
                         runner->modelLabel().c_str());
                return runner;
            } catch (const std::exception &e) {
                APU_LOGW("MediaTek APU runner unavailable (%s), falling back to ARM64-SDOT-INT8",
                         e.what());
            }
        }
    }

    // Default fast path on ARMv8.2+ mobile SoCs: 4ms TTFT ARM64-SDOT-INT8
    return std::make_unique<MobileInt8StepRunner>(modelDir);
}

void compileMediaTekDlaFile(const std::filesystem::path &modelDir,
                            const std::filesystem::path &outDlaPath) {
    MediaTekNeuronServiceRunner exporter(modelDir, {}, outDlaPath);
}

} // namespace fcitx5_llm
