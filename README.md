# fcitx5-llm-completion

100% Pure-C++20 Single-Library (`libllmcompletion.so`) Offline Pure-Unicode `RWKV-8 ("Heron") + Two-Tier Personal ROSA` NPU Inline Completion for Fcitx5 (Linux Intel AI Boost NPU & Android Qualcomm QNN / LiteRT NPU).

## Architecture (`CMake + Ninja`)

- **`src/core/`** (Cross-Platform Pure C++20 Static Library `fcitx5_llm_core` — shared across Linux & Android NDK):
  - `unicode_codec.{hpp,cpp}`: BPE-free `PureUnicodeCodec` (`Vocab = 4096`: 3,830 direct Unicode codepoints + 256 lossless UTF-8 fallback bytes + 10 interaction event tokens) and zero-width DOM placeholder sanitization (`U+200B`, `U+2060`, `U+FEFF`).
  - `online_sam.{hpp,cpp}`: Tier-A live session `EventAwareOnlineSAM` (`ROSA` exact cross-app suffix automaton with seamless tombstone backspace rollback and `predictions.jsonl` warm-start).
  - `online_adapter.{hpp,cpp}`: NPU-resident full-rank FP16 Test-Time Training manager (`online_adapter.bin`) + language-agnostic Bayesian-Shannon `DynamicSuffixMemory` (`online_suffix.bin`, zero hardcoded word lists).
  - `suffix_index.{hpp,cpp}` + `npz_reader.{hpp,cpp}`: Native `zlib` `.npz` loader + Tier-B `PersonalSuffixIndex` (`O(log N)` multi-order suffix retrieval & background unigram/branching priors).
  - `rwkv8_engine.{hpp,cpp}`: Hardware-agnostic `RWKV-8` recurrent state machine, Copy-on-Write speculative rollout, mid-token atomic preemption, and always-display step-0 decoding.
  - `text_ops.{hpp,cpp}`: Strict single-Unicode-codepoint stepping (`segmentCompletionChunks`), bracket-balance-aware FIM suffix deduplication (`trimSuffixOverlap`), hot-reloaded `AppBlacklist`, and `10 MiB × 5` rotating `PredictionLogger`.
- **`src/backends/`** (`IRwkvStepRunner` single-method NPU hardware abstraction):
  - `openvino_runner.cpp` (Linux `x86_64`): Native OpenVINO C++ Runtime (`openvino::runtime`) targeting `Intel(R) AI Boost` NPU (`NPU -> GPU -> CPU` automatic fallback, `~4.5 ms` TTFT on `< 50 MB` `37.26M` INT8 model).
  - `android_npu_runner.cpp` (Android `arm64-v8a`): Dynamic `QNN HTP / LiteRT / NNAPI ACCELERATOR` NPU runner.
- **`src/addon/llmcompletion.cpp`** (Self-Contained `libllmcompletion.so` — zero external daemon or socket):
  - Runs NPU initialization and inference asynchronously on an internal `std::thread` with lock-free atomic request preemption and `pipe2` `EventLoop` wakeup (`0 ms` main-thread blocking).
  - Renders proposals via zero-DOM-leak cursor-following `InputPanel` (`setAuxUp("⇥ ...")` on Desktop, `CandidateList` on Android), keeping `clientPreedit` 100% empty.

## Quick Start (Linux)

```bash
# Build + run C++20 unit & NPU integration tests
make test

# Install libllmcompletion.so and reload Fcitx5
make install
fcitx5 -r -d
```

## Android (`fcitx5-android` Plugin APK)

```bash
# Cross-compile arm64-v8a libllmcompletion.so and package standalone plugin APK
make android-apk

# Install onto Android phone running Fcitx5 for Android
adb install -r dist/org.fcitx.fcitx5.android.plugin.llm_completion-v2.0.0-arm64-v8a.apk
```

- **Collector-Only Mode (Default without local model)**: When `models/unicode-37m-rwkv8` is absent on the phone, the background worker runs with zero NPU/RAM overhead and records non-password interaction events to `/storage/emulated/0/Android/data/org.fcitx.fcitx5.android/files/data/llm-completion/predictions.jsonl`.

## Keybindings & Controls

- **`Ctrl + Tab`**: Accept one Unicode character (auto-chains when final character is accepted).
- **`Esc`** (or keep typing): Dismiss proposal (records reject state so dismissed text is not repeated).
- **`Ctrl + Alt + I`** or Fcitx5 tray menu (**内联补全: 开/关**): Toggle on/off.
- **`~/.config/fcitx5-llm-completion/blacklist.txt`**: One app name per line (`#` comments allowed), hot-reloaded on save.
- **`FCITX5_LLM_ONLINE_LR`**: Optional environment variable (`0.0`–`5.0`, default `1.45`) tuning real-time NPU SGD online adaptation rate.
