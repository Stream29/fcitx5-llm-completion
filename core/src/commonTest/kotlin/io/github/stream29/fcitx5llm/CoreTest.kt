package io.github.stream29.fcitx5llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CoreTest {
    @Test
    fun testAppRouter() {
        val router = AppRouter()
        assertEquals(AppRouter.DEFAULT_CODE_MODEL, router.selectModel("code"))
        assertEquals(AppRouter.DEFAULT_TEXT_MODEL, router.selectModel("Telegram"))
        assertFalse(router.isAllowed("KeePassXC"))
        assertTrue(router.isAllowed("Telegram"))

        val userBlacklist = AppRouter.parseBlacklistText("# comment\nTelegram\n  Obsidian \n")
        assertFalse(router.isAllowed("telegram", userBlacklist))
        assertFalse(router.isAllowed("obsidian", userBlacklist))
        assertTrue(router.isAllowed("firefox", userBlacklist))
    }

    @Test
    fun testPreparePrefix() {
        val (trimmed, ws) = DashScopeClient.preparePrefix("hello world  ")
        assertEquals("hello world", trimmed)
        assertEquals("  ", ws)

        val longText = "a".repeat(3000)
        val (clipped, _) = DashScopeClient.preparePrefix(longText)
        assertEquals(DashScopeClient.MAX_PREFIX_CHARS, clipped.length)
    }

    @Test
    fun testProtocolRoundTrip() {
        val req = ipcJson.decodeFromString(
            IpcRequest.serializer(),
            """{"type":"suggest","req_id":42,"program":"Telegram","prefix":"你好"}""",
        )
        assertEquals("suggest", req.type)
        assertEquals(42L, req.reqId)
        assertEquals("Telegram", req.program)

        val resp = IpcResponse(
            type = "proposal",
            reqId = 42L,
            text = "世界",
            chunks = listOf("世", "界"),
            done = true,
            model = "qwen-flash",
            latencyMs = 250,
        )
        val encoded = ipcJson.encodeToString(IpcResponse.serializer(), resp)
        assertTrue(encoded.contains("\"chunks\":[\"世\",\"界\"]"))
        assertTrue(encoded.contains("\"done\":true"))
    }
}
