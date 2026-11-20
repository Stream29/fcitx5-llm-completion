package io.github.stream29.fcitx5llm

class AppRouter(
    val textModel: String = DEFAULT_TEXT_MODEL,
    val codeModel: String = DEFAULT_CODE_MODEL,
    private val codeApps: Set<String> = DEFAULT_CODE_APPS,
    private val builtinBlacklist: Set<String> = DEFAULT_BUILTIN_BLACKLIST,
) {
    fun isAllowed(program: String, userBlacklist: Set<String> = emptySet()): Boolean {
        val p = program.trim().lowercase()
        if (p.isEmpty()) return true
        if (p in builtinBlacklist) return false
        return p !in userBlacklist
    }

    fun selectModel(program: String): String {
        val p = program.trim().lowercase()
        return if (p in codeApps) codeModel else textModel
    }

    companion object {
        const val DEFAULT_TEXT_MODEL = "qwen-flash"
        const val DEFAULT_CODE_MODEL = "qwen3-coder-flash"

        val DEFAULT_CODE_APPS: Set<String> = setOf(
            "code", "code-oss", "cursor", "zed", "nvim", "vim",
            "alacritty", "kitty", "ghostty", "wezterm", "wezterm-gui",
            "gnome-terminal", "gnome-terminal-server", "kgx", "konsole", "foot",
            "idea", "goland", "clion", "pycharm", "webstorm", "antigravity",
        )

        val DEFAULT_BUILTIN_BLACKLIST: Set<String> = setOf(
            "keepassxc", "bitwarden", "1password", "pinentry",
            "pinentry-gnome3", "pinentry-gtk-2", "gcr-prompter",
            "polkit-gnome-authentication-agent-1",
        )

        fun parseBlacklistText(content: String): Set<String> =
            content.lineSequence()
                .map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .map { it.lowercase() }
                .toSet()
    }
}
