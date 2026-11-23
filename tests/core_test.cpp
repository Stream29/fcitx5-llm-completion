#include "backends/step_runner.hpp"
#include "core/npz_reader.hpp"
#include "core/online_adapter.hpp"
#include "core/online_sam.hpp"
#include "core/rwkv8_engine.hpp"
#include "core/suffix_index.hpp"
#include "core/text_ops.hpp"
#include "core/unicode_codec.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

#undef assert
#define assert(expr)                                                           \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::cerr << "ASSERTION FAILED at " << __FILE__ << ":" << __LINE__ \
                      << ": " #expr "\n";                                      \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

using namespace fcitx5_llm;

namespace {

void testInvisibleStrippingAndUtf8() {
    std::string dirty = std::string("你\xE2\x80\x8B好") + "\xE2\x81\xA0" +
                        "世界" + "\xEF\xBB\xBF" + "!";
    assert(stripInvisibleFormattingUtf8(dirty) == "你好世界!");
    // Preserve U+200D ZWJ (\xE2\x80\x8D)
    std::string zwj = "A\xE2\x80\x8D" "B";
    assert(stripInvisibleFormattingUtf8(zwj) == zwj);

    std::string s = "测试abc中";
    assert(utf8CharCount(s) == 6);
    assert(popLastUtf8Char(s) == "中");
    assert(s == "测试abc");
    assert(utf8TailChars(s, 4) == "试abc");
    assert(utf8HeadChars(s, 3) == "测试a");
}

void testPureUnicodeCodec() {
    std::vector<uint32_t> cps = {'\n', 'a', 'b', 0x4E2D, 0x6587}; // 中, 文
    PureUnicodeCodec codec(cps, 512);
    std::string sample = "ab中文🚀\n";
    auto ids = codec.encode(sample);
    assert(!ids.empty());
    assert(codec.decode(ids) == sample);
}

void testOnlineSamAndBackspace() {
    std::vector<uint32_t> cps;
    for (uint32_t c = 32; c < 127; ++c) {
        cps.push_back(c);
    }
    cps.push_back(0x4E0A); // 上
    cps.push_back(0x5C4F); // 屏
    cps.push_back(0x6D6E); // 浮
    cps.push_back(0x5C42); // 层
    PureUnicodeCodec codec(cps, 1024);
    EventAwareOnlineSAM sam(4096);

    for (int32_t t : codec.encode("上屏浮层")) {
        sam.pushCharTok(t);
    }
    sam.boundary();

    // Type "上屏" -> should recall "浮" with matchLen=2!
    auto qToks = codec.encode("上屏");
    SamMatch m{};
    for (int32_t t : qToks) {
        m = sam.pushCharTok(t);
    }
    assert(m.matchLen == 2);
    assert(codec.decode(std::span<const int32_t>(&m.nextTok, 1)) == "浮");

    // Backspace typo rollback: type "上屏X" then backspace 1 -> "X" tombstoned,
    // and surviving prefix "上屏" immediately recalls "浮" AND recalls "浮" in a new clause!
    sam.boundary();
    for (int32_t t : codec.encode("上屏X")) {
        sam.pushCharTok(t);
    }
    sam.tombstoneBackspace(1);
    m = sam.queryCurrent();
    assert(codec.decode(std::span<const int32_t>(&m.nextTok, 1)) == "浮");

    sam.boundary();
    for (int32_t t : codec.encode("上屏")) {
        m = sam.pushCharTok(t);
    }
    assert(codec.decode(std::span<const int32_t>(&m.nextTok, 1)) == "浮");
}

void testChunkerAndSuffixTrim() {
    std::string txt = "ex那边的spec🚀";
    auto chunks = segmentCompletionChunks(txt);
    std::string joined;
    for (const auto &c : chunks) {
        joined += c;
    }
    assert(joined == txt);
    assert(chunks.size() == 10);
    assert(chunks[0] == "e");
    assert(chunks[1] == "x");
    assert(chunks[2] == "那");
    assert(chunks[5] == "s");
    assert(chunks[9] == "🚀");

    assert(trimSuffixOverlap("移动光标再输入", "再输入，我的") == "移动光标");
    // Preserve closing paren opened inside generated text
    assert(trimSuffixOverlap("foo(bar)", ") + 1") == "foo(bar)");
}

void testBlacklistAndFp16() {
    AppBlacklist bl;
    assert(!bl.isAllowed("KeePassXC"));
    assert(!bl.isAllowed("1password"));
    assert(bl.isAllowed("kitty"));

    // IEEE 754 FP16 0x3C00 == 1.0f, 0xC000 == -2.0f
    assert(std::fabs(fp16ToFp32(0x3C00u) - 1.0f) < 1e-6f);
    assert(std::fabs(fp16ToFp32(0xC000u) - (-2.0f)) < 1e-6f);
}

void testDynamicSuffixAndDenoise() {
    // Pure language-agnostic information-theoretic test (zero character identity used —
    // works on arbitrary integer token IDs purely via unigram self-information & branching fanout!)
    DynamicSuffixMemory mem(1024, 4096);

    // Simulate two high-frequency tokens (100, 101) — could be English "in", Chinese "它的", or C++ ");"
    mem.setTokenUnigramPrior(100, 300.0f);
    mem.setTokenUnigramPrior(101, 450.0f);

    // 1. High-frequency low-self-information 2-token prefix (100, 101) followed by 3 diverse continuations
    // (high branching entropy -> 1-off accidental collision is suppressed universally!)
    for (int32_t nxt : {200, 201, 202}) {
        mem.boundary();
        mem.pushToken(100, 1.0f);
        mem.pushToken(101, 1.0f);
        mem.pushToken(nxt, 1.0f);
    }
    const int32_t qLowInfo[2] = {100, 101};
    auto [distNoisy, mlNoisy] = mem.queryDistribution(qLowInfo, nullptr, 2);
    assert(distNoisy.empty() && mlNoisy == 0);

    // Typo rollback check: push typo 999 then backspace -> completely rolled back!
    mem.boundary();
    mem.pushToken(100, 1.0f);
    mem.pushToken(101, 1.0f);
    mem.pushToken(999, 1.0f);
    mem.rollbackBackspace(1);
    assert(mem.countForSuffix2(100, 101, 999) < 0.01f);

    // Repeated / explicitly accepted continuation (300) overcomes entropy prior!
    mem.pushToken(300, 2.5f);
    mem.boundary();
    mem.pushToken(100, 1.0f);
    mem.pushToken(101, 1.0f);
    mem.pushToken(300, 2.5f);
    auto [distConfirmed, mlConfirmed] =
        mem.queryDistribution(qLowInfo, nullptr, 2);
    assert(mlConfirmed == 2 && !distConfirmed.empty() &&
           distConfirmed[0].first == 300);

    // 2. High-self-information (low unigram frequency) 2-token prefix (500, 501)
    // triggers IMMEDIATELY on the very 1st occurrence!
    mem.boundary();
    mem.pushToken(500, 1.0f);
    mem.pushToken(501, 1.0f);
    mem.pushToken(888, 1.0f);
    const int32_t qHighInfo[2] = {500, 501};
    auto [distRare, mlRare] = mem.queryDistribution(qHighInfo, nullptr, 2);
    assert(mlRare == 2 && !distRare.empty() && distRare[0].first == 888);
}

void testLiveNpuInferenceIfPresent() {
    const char *home = std::getenv("HOME");
    if (!home) {
        return;
    }
    std::filesystem::path modelDir =
        std::filesystem::path(home) /
        ".local/share/fcitx5-llm-completion/models/unicode-37m-rwkv8";
    std::filesystem::path cacheDir =
        std::filesystem::path(home) /
        ".cache/fcitx5-llm-completion/ov_cache";

    if (!std::filesystem::is_regular_file(modelDir / "rwkv8_int8.xml")) {
        std::cout << "[SKIP] Local NPU model not installed at " << modelDir
                  << "\n";
        return;
    }

    std::filesystem::path tmpAdapter =
        std::filesystem::temp_directory_path() /
        ("fcitx5_llm_npu_adapter_test_" + std::to_string(::getpid()) + ".bin");
    std::filesystem::path tmpSuffix = tmpAdapter.string() + ".suffix";
    std::error_code ec;
    std::filesystem::remove(tmpAdapter, ec);
    std::filesystem::remove(tmpSuffix, ec);

    {
        auto runner = createPlatformNpuRunner(modelDir, cacheDir, "NPU");
        Rwkv8CompletionEngine engine(std::move(runner), modelDir, {}, tmpAdapter);
        std::cout << "[NPU] Loaded " << engine.modelLabel()
                  << " (npu_ttt=" << engine.hasHardwareOnlineAdapter() << ")\n";
        assert(engine.deviceName() == "NPU");
        assert(engine.hasHardwareOnlineAdapter());

        // Warmup 2 calls
        for (int i = 0; i < 2; ++i) {
            engine.suggest({"kitty", "测试预热", "", 6});
        }

        SuggestParams sp{"kitty", "std::vec", "", 10};
        auto out = engine.suggest(sp);
        std::cout << "[NPU] 'std::vec' -> '" << out.text
                  << "' (ttft=" << out.ttftMs << "ms, latency=" << out.latencyMs
                  << "ms, finish=" << out.finishReason << ")\n";
        assert(!out.text.empty());
        assert(!out.chunks.empty());

        // Test 1: Online learning + typo backspace rollback + dismiss_key state purity
        InteractionEvent focusEv;
        focusEv.event = "focus_in";
        focusEv.program = "kitty";
        engine.handleEvent(focusEv);

        // Type "代号沧溟" then accidental typo "错" + backspace("错") + "星轨"
        auto typeStr = [&](const std::string &s) {
            InteractionEvent ev;
            ev.event = "user_type";
            ev.program = "kitty";
            ev.text = s;
            engine.handleEvent(ev);
            // Simulate normal continued typing dismissing intermediate popup
            InteractionEvent dk;
            dk.event = "dismiss_key";
            dk.program = "kitty";
            dk.remaining = "干扰猜测词";
            engine.handleEvent(dk);
        };
        typeStr("代号");
        typeStr("错");
        InteractionEvent bsEv;
        bsEv.event = "backspace";
        bsEv.program = "kitty";
        bsEv.text = "错";
        engine.handleEvent(bsEv);
        typeStr("沧溟星轨");
        engine.flushOnlineLearning();

        assert(engine.onlineUpdates() > 0);
        assert(std::filesystem::is_regular_file(tmpAdapter, ec));

        auto learnedOut = engine.suggest({"kitty", "再次确认代号沧", "", 8});
        std::cout << "[NPU Online Learn] '再次确认代号沧' -> '" << learnedOut.text
                  << "' (updates=" << engine.onlineUpdates() << ")\n";
        assert(learnedOut.text.starts_with("溟星轨"));

        // Test 2: Explicit Esc rejection suppresses suggestion & updates NPU adapter
        InteractionEvent escEv;
        escEv.event = "dismiss_esc";
        escEv.program = "kitty";
        escEv.remaining = learnedOut.text;
        engine.handleEvent(escEv);
        auto afterEsc = engine.suggest({"kitty", "再次确认代号沧", "", 8});
        std::cout << "[NPU After Esc] '再次确认代号沧' -> '" << afterEsc.text << "'\n";
        assert(!afterEsc.text.starts_with("溟"));

        // Test 2.5: Pure Neural NPU SGD semantic generalization + clean phrase reboot check!
        for (const char *sent : {"周一主题：青龙\n", "周二配色：青龙\n",
                                 "周三皮肤：青龙\n", "今天壁纸：青龙\n",
                                 "刚才图标：青龙\n", "周末字体：青龙\n",
                                 "项目暗号星轨协议\n"}) {
            InteractionEvent fe;
            fe.event = "focus_in";
            fe.program = "kitty";
            engine.handleEvent(fe);
            typeStr(sent);
        }
        engine.flushOnlineLearning();
        assert(std::filesystem::is_regular_file(tmpSuffix, ec));

        // Test 2.6: Coalesced cursor_jump + Incremental FIM state reuse (~8ms TTFT vs ~650ms)
        {
            using Clock = std::chrono::steady_clock;
            std::string longPre(180, 'a');
            longPre += "正在编辑中间段落：代号";
            std::string longSuf =
                "，后续还有很长一段正文内容用于验证FIM增量状态复用延迟表现。";
            for (int k = 0; k < 4; ++k) {
                longSuf += longSuf;
            }
            auto tJump0 = Clock::now();
            for (int j = 0; j < 10; ++j) {
                InteractionEvent cj;
                cj.event = "cursor_jump";
                cj.program = "kitty";
                cj.prefix = longPre.substr(0, longPre.size() - (9 - j));
                cj.suffix = longSuf;
                engine.handleEvent(cj);
            }
            auto jumpBatchMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now() - tJump0)
                    .count();
            std::cout << "[NPU 10x cursor_jump lazy enqueue] " << jumpBatchMs
                      << "ms\n";
            assert(jumpBatchMs <= 5);
            assert(engine.hasPendingContext());

            // Aborted prewarm leaves pending context intact
            int pwSteps = 0;
            bool pwDone =
                engine.prewarmPendingContext([&]() { return (++pwSteps) >= 3; });
            assert(!pwDone);
            assert(engine.hasPendingContext());

            // Full idle prewarm succeeds
            assert(engine.prewarmPendingContext());
            assert(!engine.hasPendingContext());

            // Incremental FIM keystroke: must reuse mainState_ with single-step <PREDICT> (~8ms TTFT!)
            typeStr("沧");
            std::string curFimPre = longPre + "沧";
            auto fimOut1 = engine.suggest({"kitty", curFimPre, longSuf, 8});
            std::cout << "[NPU Incremental FIM #1] mode=" << fimOut1.mode
                      << " ttft=" << fimOut1.ttftMs
                      << "ms latency=" << fimOut1.latencyMs << "ms -> '"
                      << fimOut1.text << "'\n";
            assert(fimOut1.mode == "fim");
            assert(fimOut1.ttftMs <= 25);
            assert(fimOut1.text.starts_with("溟星轨"));

            typeStr("溟");
            curFimPre += "溟";
            auto fimOut2 = engine.suggest({"kitty", curFimPre, longSuf, 8});
            std::cout << "[NPU Incremental FIM #2] mode=" << fimOut2.mode
                      << " ttft=" << fimOut2.ttftMs
                      << "ms latency=" << fimOut2.latencyMs << "ms -> '"
                      << fimOut2.text << "'\n";
            assert(fimOut2.mode == "fim");
            assert(fimOut2.ttftMs <= 25);
            assert(fimOut2.text.starts_with("星轨"));
        }

        // Test 2.7: Anti-Fragmentation Invariant — Typo + 8-Step Checkpoint Backspace + Per-Char accept_chunk == Pure Continuous Stream!
        {
            std::filesystem::path tmpCleanA =
                tmpAdapter.string() + ".cleanA";
            std::filesystem::path tmpCleanB =
                tmpAdapter.string() + ".cleanB";
            std::filesystem::remove(tmpCleanA, ec);
            std::filesystem::remove(tmpCleanA.string() + ".suffix", ec);
            std::filesystem::remove(tmpCleanB, ec);
            std::filesystem::remove(tmpCleanB.string() + ".suffix", ec);

            Rwkv8CompletionEngine engFrag(
                createPlatformNpuRunner(modelDir, cacheDir, "NPU"), modelDir,
                {}, tmpCleanA);
            Rwkv8CompletionEngine engPure(
                createPlatformNpuRunner(modelDir, cacheDir, "NPU"), modelDir,
                {}, tmpCleanB);

            engFrag.handleEvent({"focus_in", "gnome-shell", "", "", "", "", ""});
            engFrag.handleEvent({"user_type", "gnome-shell", "", "",
                                 "今天天气真不错，我们一错字", "", ""});
            engFrag.handleEvent({"backspace", "gnome-shell", "", "", "字", "", ""});
            engFrag.handleEvent({"backspace", "gnome-shell", "", "", "错", "", ""});
            // Post-backspace Wayland cursor_jump echo with matching prefix must be a 0ms no-op!
            engFrag.handleEvent({"cursor_jump", "gnome-shell",
                                 "今天天气真不错，我们一", "", "", "", ""});
            assert(!engFrag.hasPendingContext());
            engFrag.handleEvent({"user_type", "gnome-shell", "", "", "起", "", ""});
            for (const char *ch : {"去", "公", "园"}) {
                engFrag.handleEvent(
                    {"accept_chunk", "gnome-shell", "", "", "", ch, ""});
            }
            auto outFrag = engFrag.suggest(
                {"gnome-shell", "今天天气真不错，我们一起去公园", "", 10});

            engPure.handleEvent({"focus_in", "gnome-shell", "", "", "", "", ""});
            engPure.handleEvent({"user_type", "gnome-shell", "", "",
                                 "今天天气真不错，我们一起去公园", "", ""});
            auto outPure = engPure.suggest(
                {"gnome-shell", "今天天气真不错，我们一起去公园", "", 10});

            std::cout << "[NPU Anti-Fragmentation] frag='" << outFrag.text
                      << "' (ttft=" << outFrag.ttftMs << "ms) vs pure='"
                      << outPure.text << "'\n";
            assert(outFrag.ttftMs <= 25);
            assert(!outFrag.text.empty());
            assert(!outFrag.text.starts_with("，"));
            assert(outFrag.text == outPure.text);

            std::filesystem::remove(tmpCleanA, ec);
            std::filesystem::remove(tmpCleanA.string() + ".suffix", ec);
            std::filesystem::remove(tmpCleanB, ec);
            std::filesystem::remove(tmpCleanB.string() + ".suffix", ec);
        }

        // Test mid-rollout atomic cancellation
        int calls = 0;
        auto cancelledOut =
            engine.suggest({"kitty", "今天天气真不错，我们", "", 10},
                           [&]() { return (++calls) >= 2; });
        assert(cancelledOut.cancelled);
    }

    // Test 3a: Reload persistedba/full engine state (online_adapter.bin + online_suffix.bin)
    // in a brand-new Rwkv8CompletionEngine with ZERO warmup log — survives full reboot!
    {
        auto runner2 = createPlatformNpuRunner(modelDir, cacheDir, "NPU");
        Rwkv8CompletionEngine engine2(std::move(runner2), modelDir, {},
                                      tmpAdapter);
        assert(engine2.onlineUpdates() > 0);
        auto rebootLearned =
            engine2.suggest({"kitty", "确认项目暗号星轨", "", 8});
        std::cout << "[NPU Reboot Persistence] persisted_updates="
                  << engine2.onlineUpdates() << ", '确认项目暗号星轨' -> '"
                  << rebootLearned.text << "' (finish="
                  << rebootLearned.finishReason << ")\n";
        assert(rebootLearned.text.starts_with("协议"));
    }

    // Test 3b: Verify Pure NPU Neural FP16 Adapter (with suffix memory wiped) generalizes
    // > +3.0 logits (>20x probability boost) onto a 100% unseen sentence ("明天样式：")!
    std::filesystem::remove(tmpSuffix, ec);
    {
        PureUnicodeCodec codec(modelDir / "unicode_vocab.json", 4096);
        int32_t tokQing = codec.encode("青")[0];
        int32_t tokJiang = codec.encode("将")[0];

        auto runner3 = createPlatformNpuRunner(modelDir, cacheDir, "NPU");
        auto *r3Ptr = runner3.get();
        Rwkv8CompletionEngine engine3(std::move(runner3), modelDir, {},
                                      tmpAdapter);

        RwkvState st = RwkvState::createZero(r3Ptr->config());
        std::vector<float> xIn(512), enn(5120, 1.0f), lg(4096), xo(512);
        std::vector<int32_t> seq = codec.encode("kitty\n");
        seq.insert(seq.begin(), TOK_FOCUS);
        seq.push_back(TOK_TYPE);
        auto q = codec.encode("明天样式：");
        seq.insert(seq.end(), q.begin(), q.end());
        for (int32_t t : seq) {
            std::memcpy(xIn.data(),
                        r3Ptr->tables().emb.data() + static_cast<size_t>(t) * 512,
                        512 * sizeof(float));
            for (int l = 0; l < 10; ++l) {
                float *dst = enn.data() + l * 512;
                std::fill_n(dst, 512, 1.0f);
                const float *u =
                    r3Ptr->tables().ennU.data() + (l * 4096 + t) * 48;
                const float *v = r3Ptr->tables().ennV.data() + l * 48 * 512;
                for (int k = 0; k < 48; ++k) {
                    for (int j = 0; j < 512; ++j) {
                        dst[j] += u[k] * v[k * 512 + j];
                    }
                }
            }
            r3Ptr->inferStep(xIn.data(), enn.data(), st, lg.data(), xo.data());
        }
        std::vector<float> delta(4096, 0.0f);
        r3Ptr->applyAdapterLogits(xo.data(), delta.data());
        std::cout << "[NPU Pure-Neural Generalization] unseen '明天样式：' -> "
                  << "delta[青]=+" << delta[tokQing]
                  << " logits, delta[将]=" << delta[tokJiang] << " logits\n";
        assert(delta[tokQing] > 3.0f);
        assert(delta[tokJiang] < 0.0f);
    }
    std::filesystem::remove(tmpAdapter, ec);
    std::filesystem::remove(tmpSuffix, ec);
}

} // namespace

int main() {
    testInvisibleStrippingAndUtf8();
    testPureUnicodeCodec();
    testOnlineSamAndBackspace();
    testDynamicSuffixAndDenoise();
    testChunkerAndSuffixTrim();
    testBlacklistAndFp16();
    testLiveNpuInferenceIfPresent();
    std::cout << "ALL C++20 CORE & NPU TESTS PASSED!\n";
    return 0;
}
