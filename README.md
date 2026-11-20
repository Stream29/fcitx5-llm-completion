# fcitx5-llm-completion

Copilot-style LLM inline completion for Fcitx5 (Linux & Android KMP core), backed by Aliyun DashScope zero-prompt Partial Mode streaming.

## Architecture

- **`core/`** (Kotlin Multiplatform: `commonMain`, `jvm`, `linuxX64`):
  - `DashScopeClient`: Zero-prompt Partial Mode (`partial: true`, `enable_thinking: false`), SSE streaming, BPE token `chunks`, 2048-char prefix window.
  - `AppRouter`: Per-app model routing (`qwen-flash` / `qwen3-coder-flash`) and blacklist filtering.
  - `Protocol`: NDJSON IPC schema shared across platforms.
- **`daemon/`** (Kotlin/Native `linuxX64` CLI):
  - `fcitx5-llm-completion-daemon`: Native binary using `ktor-client-curl` (HTTP/2 Keep-Alive), POSIX Unix Domain Socket (`$XDG_RUNTIME_DIR/fcitx5-llm-completion.sock`, mode `0600`), latest-wins request cancellation, and hot-reloadable `~/.config/fcitx5-llm-completion/blacklist.txt`.
- **`addon/`** (Thin C++ Fcitx5 Core module):
  - `libllmcompletion.so` + `llmcompletion.conf`: 0ms immediate trigger, Rime-safe `clientPreedit`, BPE word-chunk acceptance (`Ctrl+Tab`), full acceptance (`Ctrl+Shift+Tab`), cancel (`Esc`), and tray StatusArea toggle + `Ctrl+Alt+I`.

## Quick Start (Linux)

```bash
# Build daemon CLI + C++ addon
make all

# Install addon & daemon CLI
make install

# Run daemon (reads DASHSCOPE_API_KEY env or ~/ACodeSpace/dashscope.env)
fcitx5-llm-completion-daemon &

# Reload Fcitx5
fcitx5 -r -d
```

## Keybindings & Controls

- **`Ctrl + Tab`**: Accept one BPE word/token chunk.
- **`Ctrl + Shift + Tab`**: Accept the entire remaining proposal.
- **`Esc`** (or keep typing): Dismiss proposal.
- **`Ctrl + Alt + I`** or Fcitx5 tray menu (**内联补全: 开/关**): Toggle on/off.
- **`~/.config/fcitx5-llm-completion/blacklist.txt`**: One app name per line (`#` comments allowed), hot-reloaded on save.
