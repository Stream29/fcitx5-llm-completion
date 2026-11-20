@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.stream29.fcitx5llm.daemon

import io.github.stream29.fcitx5llm.AppRouter
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import platform.posix.stat

internal class BlacklistWatcher(
    private val path: String,
) {
    private var lastSec: Long = -1L
    private var lastNsec: Long = -1L
    private var cached: Set<String> = emptySet()

    fun current(): Set<String> {
        if (path.isBlank()) return emptySet()
        memScoped {
            val st = alloc<stat>()
            if (stat(path, st.ptr) != 0) {
                cached = emptySet()
                lastSec = -1L
                lastNsec = -1L
                return cached
            }
            val sec = st.st_mtim.tv_sec
            val nsec = st.st_mtim.tv_nsec
            if (sec == lastSec && nsec == lastNsec) {
                return cached
            }
            val text = readFileText(path) ?: return cached
            cached = AppRouter.parseBlacklistText(text)
            lastSec = sec
            lastNsec = nsec
            return cached
        }
    }
}
