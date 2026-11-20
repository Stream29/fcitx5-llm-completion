package org.fcitx.fcitx5.android.plugin.llm_completion

import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.LocalServerSocket
import android.net.LocalSocket
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Messenger
import android.util.Log
import io.github.stream29.fcitx5llm.AppRouter
import io.github.stream29.fcitx5llm.DashScopeClient
import io.github.stream29.fcitx5llm.IpcRequest
import io.github.stream29.fcitx5llm.IpcResponse
import io.github.stream29.fcitx5llm.SuggestRequest
import io.github.stream29.fcitx5llm.ipcJson
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.util.concurrent.atomic.AtomicLong

class LlmCompletionService : Service() {

    private val handler = Handler(Looper.getMainLooper())
    private lateinit var messenger: Messenger
    private var scope: CoroutineScope? = null
    private var serverSocket: LocalServerSocket? = null
    private var httpClient: HttpClient? = null
    private val router = AppRouter()

    override fun onBind(intent: Intent): IBinder {
        messenger = Messenger(handler)
        startServer()
        return messenger.binder
    }

    override fun onUnbind(intent: Intent?): Boolean {
        stopServer()
        return false
    }

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }

    private fun startServer() {
        if (scope != null) return
        val s = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        scope = s
        val client = HttpClient(OkHttp)
        httpClient = client

        s.launch {
            try {
                val server = LocalServerSocket(ABSTRACT_SOCKET_NAME)
                serverSocket = server
                Log.i(TAG, "Listening on abstract socket @$ABSTRACT_SOCKET_NAME")
                while (isActive) {
                    val conn = server.accept()
                    s.launch { handleClient(this, conn, client) }
                }
            } catch (e: Exception) {
                if (isActive) {
                    Log.e(TAG, "Socket server stopped: ${e.message}", e)
                }
            }
        }
    }

    private fun stopServer() {
        runCatching { serverSocket?.close() }
        serverSocket = null
        scope?.cancel()
        scope = null
        httpClient?.close()
        httpClient = null
    }

    private fun resolveApiKey(): String {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val prefKey = prefs.getString(KEY_API_KEY, "")?.trim().orEmpty()
        if (prefKey.isNotEmpty()) return prefKey

        val envFile = File(filesDir, "dashscope.env")
        if (envFile.isFile) {
            for (raw in envFile.readLines()) {
                val line = raw.trim()
                if (line.startsWith("DASHSCOPE_API_KEY=")) {
                    val v = line.removePrefix("DASHSCOPE_API_KEY=").trim().trim('"', '\'')
                    if (v.isNotEmpty()) return v
                }
            }
        }
        return ""
    }

    private fun resolveUserBlacklist(): Set<String> {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_BLACKLIST, "").orEmpty()
        return AppRouter.parseBlacklistText(raw)
    }

    private suspend fun handleClient(
        scope: CoroutineScope,
        socket: LocalSocket,
        client: HttpClient,
    ) {
        val writeMutex = Mutex()
        val currentReqId = AtomicLong(0L)
        val pendingQueue = Channel<IpcRequest>(Channel.CONFLATED)
        val out = socket.outputStream

        suspend fun sendResponse(resp: IpcResponse) {
            val bytes = (ipcJson.encodeToString(IpcResponse.serializer(), resp) + "\n").toByteArray()
            writeMutex.withLock {
                runCatching {
                    out.write(bytes)
                    out.flush()
                }
            }
        }

        val worker = scope.launch {
            for (req in pendingQueue) {
                if (currentReqId.get() != req.reqId) continue
                val apiKey = resolveApiKey()
                if (apiKey.isEmpty()) {
                    sendResponse(IpcResponse(type = "error", reqId = req.reqId, error = "API key not configured"))
                    continue
                }
                val model = router.selectModel(req.program)
                val dashScope = DashScopeClient(httpClient = client, apiKey = apiKey)
                try {
                    val result = dashScope.suggestStream(
                        req = SuggestRequest(
                            model = model,
                            prefix = req.prefix,
                            maxTokens = req.maxTokens,
                        ),
                        shouldStop = { currentReqId.get() != req.reqId },
                    ) { accumulated, chunks ->
                        if (currentReqId.get() != req.reqId) return@suggestStream
                        scope.launch {
                            if (currentReqId.get() == req.reqId) {
                                sendResponse(
                                    IpcResponse(
                                        type = "proposal",
                                        reqId = req.reqId,
                                        text = accumulated,
                                        chunks = chunks,
                                        done = false,
                                        model = model,
                                    ),
                                )
                            }
                        }
                    }
                    if (currentReqId.get() == req.reqId) {
                        sendResponse(
                            IpcResponse(
                                type = "proposal",
                                reqId = req.reqId,
                                text = result.text,
                                chunks = result.chunks,
                                done = true,
                                model = result.model,
                                latencyMs = result.latency.inWholeMilliseconds,
                            ),
                        )
                    }
                } catch (_: CancellationException) {
                    break
                } catch (e: Exception) {
                    if (currentReqId.get() == req.reqId) {
                        Log.w(TAG, "suggest failed: ${e.message}")
                        sendResponse(
                            IpcResponse(
                                type = "error",
                                reqId = req.reqId,
                                model = model,
                                error = e.message ?: "request failed",
                            ),
                        )
                    }
                }
            }
        }

        try {
            val reader = BufferedReader(InputStreamReader(socket.inputStream))
            while (scope.isActive) {
                val line = reader.readLine()?.trim() ?: break
                if (line.isEmpty()) continue
                val req = runCatching {
                    ipcJson.decodeFromString(IpcRequest.serializer(), line)
                }.getOrElse { err ->
                    sendResponse(IpcResponse(type = "error", error = "invalid json: ${err.message}"))
                    continue
                }
                when (req.type) {
                    "ping" -> sendResponse(IpcResponse(type = "pong", reqId = req.reqId))
                    "cancel" -> currentReqId.set(0L)
                    "suggest" -> {
                        if (!router.isAllowed(req.program, resolveUserBlacklist()) || req.prefix.isEmpty()) {
                            currentReqId.set(0L)
                            sendResponse(IpcResponse(type = "proposal", reqId = req.reqId, text = "", done = true))
                            continue
                        }
                        currentReqId.set(req.reqId)
                        pendingQueue.trySend(req)
                    }
                }
            }
        } finally {
            currentReqId.set(0L)
            pendingQueue.close()
            worker.cancel()
            runCatching { socket.close() }
        }
    }

    companion object {
        private const val TAG = "LlmCompletionService"
        const val ABSTRACT_SOCKET_NAME = "fcitx5-llm-completion"
        const val PREFS_NAME = "llm_completion_prefs"
        const val KEY_API_KEY = "dashscope_api_key"
        const val KEY_BLACKLIST = "app_blacklist"
    }
}
