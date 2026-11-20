@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.stream29.fcitx5llm.daemon

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen
import platform.posix.getenv
import platform.posix.getuid

internal fun envVar(name: String): String? =
    getenv(name)?.toKString()?.trim()?.takeIf { it.isNotEmpty() }

internal fun homeDir(): String = envVar("HOME") ?: "/tmp"

internal fun defaultSocketPath(): String {
    val xdg = envVar("XDG_RUNTIME_DIR")
    return if (xdg != null) {
        "$xdg/fcitx5-llm-completion.sock"
    } else {
        "/tmp/fcitx5-llm-completion-${getuid()}.sock"
    }
}

internal fun defaultBlacklistPath(): String {
    val xdgCfg = envVar("XDG_CONFIG_HOME") ?: "${homeDir()}/.config"
    return "$xdgCfg/fcitx5-llm-completion/blacklist.txt"
}

internal fun readFileText(path: String): String? {
    val fp = fopen(path, "r") ?: return null
    try {
        val sb = StringBuilder()
        memScoped {
            val buf = allocArray<ByteVar>(4096)
            while (fgets(buf, 4096, fp) != null) {
                sb.append(buf.toKString())
            }
        }
        return sb.toString()
    } finally {
        fclose(fp)
    }
}

internal fun resolveApiKey(envPath: String?): String {
    envVar("DASHSCOPE_API_KEY")?.let { return it }

    val path = envPath?.takeIf { it.isNotBlank() } ?: "${homeDir()}/ACodeSpace/dashscope.env"
    val content = readFileText(path)
        ?: error("DASHSCOPE_API_KEY not set and cannot open env file: $path")

    for (raw in content.lineSequence()) {
        val line = raw.trim()
        if (line.isEmpty() || line.startsWith("#")) continue
        val clean = line.removePrefix("export ").trim()
        val idx = clean.indexOf('=')
        if (idx <= 0) continue
        val k = clean.substring(0, idx).trim()
        if (k != "DASHSCOPE_API_KEY") continue
        val v = clean.substring(idx + 1).trim().trim('"', '\'')
        if (v.isNotEmpty()) return v
    }
    error("DASHSCOPE_API_KEY not found in environment or $path")
}
