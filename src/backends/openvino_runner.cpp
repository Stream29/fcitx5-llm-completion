#include "step_runner.hpp"
#include "../core/npz_reader.hpp"
#include "../core/text_ops.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <sstream>
#include <stdexcept>

namespace fcitx5_llm {

namespace {

int32_t extractJsonInt(const std::string &json, const std::string &key,
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

class OpenVinoStepRunner final : public IRwkvStepRunner {
public:
    OpenVinoStepRunner(const std::filesystem::path &modelDir,
                       const std::filesystem::path &cacheDir,
                       const std::string &preferredDevice) {
        cfg_ = loadModelConfigJson(modelDir / "model_config.json");
        tables_ = loadHostTablesNpz(modelDir / "host_tables.npz", cfg_);

        auto xmlPath = modelDir / "rwkv8_int8.xml";
        if (!std::filesystem::is_regular_file(xmlPath)) {
            throw std::runtime_error("Missing OpenVINO IR model: " +
                                     xmlPath.string());
        }

        std::error_code ec;
        if (!cacheDir.empty()) {
            std::filesystem::create_directories(cacheDir, ec);
        }

        auto model = core_.read_model(xmlPath.string());
        auto avail = core_.get_available_devices();
        auto hasDev = [&](const std::string &d) {
            return std::find(avail.begin(), avail.end(), d) != avail.end();
        };

        if (!preferredDevice.empty() && hasDev(preferredDevice)) {
            device_ = preferredDevice;
        } else if (hasDev("NPU")) {
            device_ = "NPU";
        } else if (hasDev("GPU")) {
            device_ = "GPU";
        } else {
            device_ = "CPU";
        }

        ov::AnyMap props;
        if (!cacheDir.empty()) {
            props[ov::cache_dir.name()] = cacheDir.string();
        }
        props[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        if (device_ == "CPU") {
            props[ov::inference_num_threads.name()] = 2;
        }

        compiled_ = core_.compile_model(model, device_, props);
        req_ = compiled_.create_infer_request();
        initNpuOnlineAdapter(props);
    }

    const RwkvModelConfig &config() const override { return cfg_; }
    const RwkvHostTables &tables() const override { return tables_; }
    const std::string &deviceName() const override { return device_; }
    std::string modelLabel() const override {
        return cfg_.modelName + "@" + device_;
    }
    bool hasHardwareOnlineAdapter() const override { return adapterReady_; }

    void applyAdapterLogits(const float *xOut, float *logitsInOut) override {
        if (!adapterReady_ || !adapterNonZero_) {
            return;
        }
        size_t d = static_cast<size_t>(cfg_.nEmbd);
        size_t v = static_cast<size_t>(cfg_.vocabSize);
        std::memcpy(adapterFwdReq_.get_input_tensor(0).data<float>(), xOut,
                    d * sizeof(float));
        adapterFwdReq_.set_input_tensor(1, adapterW_[curW_]);
        adapterFwdReq_.infer();
        const float *delta =
            adapterFwdReq_.get_output_tensor(0).data<const float>();
        for (size_t i = 0; i < v; ++i) {
            logitsInOut[i] += delta[i];
        }
    }

    void updateAdapterStep(const float *xOut, const float *baseLogits,
                           int32_t tokenIdx, float lr, float weightDecay,
                           bool isNegativePush) override {
        if (!adapterReady_ || tokenIdx < 9 || tokenIdx >= cfg_.vocabSize ||
            lr <= 0.0f) {
            return;
        }
        size_t d = static_cast<size_t>(cfg_.nEmbd);
        size_t v = static_cast<size_t>(cfg_.vocabSize);

        std::memcpy(adapterSgdReq_.get_input_tensor(0).data<float>(), xOut,
                    d * sizeof(float));
        float *bPtr = adapterSgdReq_.get_input_tensor(1).data<float>();
        for (size_t i = 0; i < v; ++i) {
            bPtr[i] = (i < 9) ? -10000.0f
                              : std::clamp(baseLogits[i], -10000.0f, 10000.0f);
        }

        float *tVec = adapterSgdReq_.get_input_tensor(2).data<float>();
        tVec[static_cast<size_t>(tokenIdx)] = isNegativePush ? -1.0f : 1.0f;
        adapterSgdReq_.get_input_tensor(3).data<float>()[0] =
            isNegativePush ? 0.0f : 1.0f;
        adapterSgdReq_.get_input_tensor(4).data<float>()[0] = lr;
        adapterSgdReq_.get_input_tensor(5).data<float>()[0] =
            isNegativePush ? 1.0f : weightDecay;

        int nextW = 1 - curW_;
        adapterSgdReq_.set_input_tensor(6, adapterW_[curW_]);
        adapterSgdReq_.set_output_tensor(0, adapterW_[nextW]);
        adapterSgdReq_.infer();

        tVec[static_cast<size_t>(tokenIdx)] = 0.0f;
        curW_ = nextW;
        adapterNonZero_ = true;
    }

    bool loadAdapterFp16(const uint16_t *fp16Weights, size_t count) override {
        size_t expected =
            static_cast<size_t>(cfg_.vocabSize) * static_cast<size_t>(cfg_.nEmbd);
        if (!adapterReady_ || count != expected) {
            return false;
        }
        std::memcpy(adapterW_[curW_].data(), fp16Weights,
                    expected * sizeof(uint16_t));
        adapterNonZero_ = false;
        for (size_t i = 0; i < expected; ++i) {
            if ((fp16Weights[i] & 0x7FFFu) != 0) {
                adapterNonZero_ = true;
                break;
            }
        }
        return true;
    }

    bool saveAdapterFp16(uint16_t *fp16Out, size_t count) const override {
        size_t expected =
            static_cast<size_t>(cfg_.vocabSize) * static_cast<size_t>(cfg_.nEmbd);
        if (!adapterReady_ || count != expected) {
            return false;
        }
        std::memcpy(fp16Out, adapterW_[curW_].data(),
                    expected * sizeof(uint16_t));
        return true;
    }

    void inferStep(const float *xIn, const float *ennTok, RwkvState &state,
                   float *logitsOut, float *xOut) override {
        size_t dBytes = static_cast<size_t>(cfg_.nEmbd) * sizeof(float);
        size_t ldBytes =
            static_cast<size_t>(cfg_.nLayer) * cfg_.nEmbd * sizeof(float);
        size_t stBytes = state.st.size() * sizeof(float);
        size_t ringBytes = state.kr.size() * sizeof(float);
        size_t vBytes = static_cast<size_t>(cfg_.vocabSize) * sizeof(float);

        std::memcpy(req_.get_input_tensor(0).data<float>(), xIn, dBytes);
        std::memcpy(req_.get_input_tensor(1).data<float>(), ennTok, ldBytes);
        std::memcpy(req_.get_input_tensor(2).data<float>(), state.ap.data(),
                    ldBytes);
        std::memcpy(req_.get_input_tensor(3).data<float>(), state.fp.data(),
                    ldBytes);
        std::memcpy(req_.get_input_tensor(4).data<float>(), state.rp.data(),
                    ldBytes);
        std::memcpy(req_.get_input_tensor(5).data<float>(), state.st.data(),
                    stBytes);
        std::memcpy(req_.get_input_tensor(6).data<float>(), state.kr.data(),
                    ringBytes);
        std::memcpy(req_.get_input_tensor(7).data<float>(), state.vr.data(),
                    ringBytes);

        req_.infer();

        std::memcpy(logitsOut, req_.get_output_tensor(0).data<float>(), vBytes);
        std::memcpy(xOut, req_.get_output_tensor(1).data<float>(), dBytes);
        std::memcpy(state.ap.data(), req_.get_output_tensor(2).data<float>(),
                    ldBytes);
        std::memcpy(state.fp.data(), req_.get_output_tensor(3).data<float>(),
                    ldBytes);
        std::memcpy(state.rp.data(), req_.get_output_tensor(4).data<float>(),
                    ldBytes);
        std::memcpy(state.st.data(), req_.get_output_tensor(5).data<float>(),
                    stBytes);
        std::memcpy(state.kr.data(), req_.get_output_tensor(6).data<float>(),
                    ringBytes);
        std::memcpy(state.vr.data(), req_.get_output_tensor(7).data<float>(),
                    ringBytes);
    }

private:
    void initNpuOnlineAdapter(const ov::AnyMap &props) {
        using namespace ov;
        using namespace ov::opset13;
        try {
            const size_t v = static_cast<size_t>(cfg_.vocabSize);
            const size_t d = static_cast<size_t>(cfg_.nEmbd);
            auto eps = Constant::create(element::f32, Shape{1, 1}, {1e-6f});
            auto axis1 = Constant::create(element::i64, Shape{1}, {1});

            // 1. NPU Forward Graph: delta_logits = normalize(h_in) @ W^T
            {
                auto hIn = std::make_shared<Parameter>(element::f32, Shape{1, d});
                auto wIn = std::make_shared<Parameter>(element::f16, Shape{v, d});
                auto wF32 = std::make_shared<Convert>(wIn, element::f32);
                auto hSq = std::make_shared<Multiply>(hIn, hIn);
                auto hNorm = std::make_shared<Sqrt>(std::make_shared<Add>(
                    std::make_shared<ReduceSum>(hSq, axis1, true), eps));
                auto hUnit = std::make_shared<Divide>(hIn, hNorm);
                auto delta =
                    std::make_shared<MatMul>(hUnit, wF32, false, true);
                auto fwdModel = std::make_shared<Model>(
                    OutputVector{delta}, ParameterVector{hIn, wIn},
                    "Rwkv8OnlineAdapterFwd");
                adapterFwdComp_ = core_.compile_model(fwdModel, device_, props);
                adapterFwdReq_ = adapterFwdComp_.create_infer_request();
            }

            // 2. NPU Fused Cross-Entropy / Contrastive SGD + L2 Row-Clip Graph
            {
                auto hIn = std::make_shared<Parameter>(element::f32, Shape{1, d});
                auto baseLg =
                    std::make_shared<Parameter>(element::f32, Shape{1, v});
                auto targetVec =
                    std::make_shared<Parameter>(element::f32, Shape{1, v});
                auto gradMode =
                    std::make_shared<Parameter>(element::f32, Shape{1, 1});
                auto lrIn =
                    std::make_shared<Parameter>(element::f32, Shape{1, 1});
                auto decayIn =
                    std::make_shared<Parameter>(element::f32, Shape{1, 1});
                auto wIn = std::make_shared<Parameter>(element::f16, Shape{v, d});

                auto wF32 = std::make_shared<Convert>(wIn, element::f32);
                auto hSq = std::make_shared<Multiply>(hIn, hIn);
                auto hNorm = std::make_shared<Sqrt>(std::make_shared<Add>(
                    std::make_shared<ReduceSum>(hSq, axis1, true), eps));
                auto hUnit = std::make_shared<Divide>(hIn, hNorm);

                auto deltaLg =
                    std::make_shared<MatMul>(hUnit, wF32, false, true);
                auto totalLg = std::make_shared<Add>(baseLg, deltaLg);
                auto probs = std::make_shared<Softmax>(totalLg, 1);

                auto modeProbs = std::make_shared<Multiply>(probs, gradMode);
                auto err = std::make_shared<Subtract>(modeProbs, targetVec);
                auto scaledErr = std::make_shared<Multiply>(err, lrIn);
                auto gradW =
                    std::make_shared<MatMul>(scaledErr, hUnit, true, false);
                auto wDecayed = std::make_shared<Multiply>(wF32, decayIn);
                auto wNext = std::make_shared<Subtract>(wDecayed, gradW);

                auto maxNorm =
                    Constant::create(element::f32, Shape{1, 1}, {14.0f});
                auto wSq = std::make_shared<Multiply>(wNext, wNext);
                auto rowNorm = std::make_shared<Sqrt>(std::make_shared<Add>(
                    std::make_shared<ReduceSum>(wSq, axis1, true), eps));
                auto clipScale = std::make_shared<Minimum>(
                    Constant::create(element::f32, Shape{1, 1}, {1.0f}),
                    std::make_shared<Divide>(maxNorm, rowNorm));
                auto wClipped = std::make_shared<Multiply>(wNext, clipScale);
                auto wOut = std::make_shared<Convert>(wClipped, element::f16);

                auto sgdModel = std::make_shared<Model>(
                    OutputVector{wOut},
                    ParameterVector{hIn, baseLg, targetVec, gradMode, lrIn,
                                    decayIn, wIn},
                    "Rwkv8OnlineAdapterSgd");
                adapterSgdComp_ = core_.compile_model(sgdModel, device_, props);
                adapterSgdReq_ = adapterSgdComp_.create_infer_request();
            }

            adapterW_[0] = ov::Tensor(ov::element::f16, {v, d});
            adapterW_[1] = ov::Tensor(ov::element::f16, {v, d});
            std::memset(adapterW_[0].data(), 0, v * d * sizeof(uint16_t));
            std::memset(adapterW_[1].data(), 0, v * d * sizeof(uint16_t));
            std::fill_n(adapterSgdReq_.get_input_tensor(2).data<float>(), v,
                        0.0f);
            curW_ = 0;
            adapterNonZero_ = false;
            adapterReady_ = true;
        } catch (...) {
            adapterReady_ = false;
        }
    }

    RwkvModelConfig cfg_;
    RwkvHostTables tables_;
    std::string device_;
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest req_;

    ov::CompiledModel adapterFwdComp_;
    ov::CompiledModel adapterSgdComp_;
    ov::InferRequest adapterFwdReq_;
    ov::InferRequest adapterSgdReq_;
    ov::Tensor adapterW_[2];
    int curW_ = 0;
    bool adapterReady_ = false;
    bool adapterNonZero_ = false;
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
    cfg.vocabSize = extractJsonInt(s, "vocab_size", cfg.vocabSize);
    cfg.nLayer = extractJsonInt(s, "n_layer", cfg.nLayer);
    cfg.nEmbd = extractJsonInt(s, "n_embd", cfg.nEmbd);
    cfg.headSize = extractJsonInt(s, "head_size", cfg.headSize);
    cfg.rosaDim = extractJsonInt(s, "rosa_dim", cfg.rosaDim);
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
    return std::make_unique<OpenVinoStepRunner>(modelDir, cacheDir,
                                                preferredDevice);
}

} // namespace fcitx5_llm
