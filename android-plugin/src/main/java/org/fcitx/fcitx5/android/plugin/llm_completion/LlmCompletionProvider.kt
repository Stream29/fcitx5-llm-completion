package org.fcitx.fcitx5.android.plugin.llm_completion

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
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
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.util.concurrent.atomic.AtomicLong

object LlmCompletionBridge {
    private const val TAG = "LlmCompletionBridge"
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val httpClient by lazy { HttpClient(OkHttp) }
    private val router = AppRouter()

    private fun resolveApiKey(context: Context): String {
        val prefs = context.getSharedPreferences(LlmCompletionService.PREFS_NAME, Context.MODE_PRIVATE)
        val prefKey = prefs.getString(LlmCompletionService.KEY_API_KEY, "")?.trim().orEmpty()
        if (prefKey.isNotEmpty()) return prefKey

        val envFile = File(context.filesDir, "dashscope.env")
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

    private fun resolveUserBlacklist(context: Context): Set<String> {
        val prefs = context.getSharedPreferences(LlmCompletionService.PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(LlmCompletionService.KEY_BLACKLIST, "").orEmpty()
        return AppRouter.parseBlacklistText(raw)
    }

    fun attachSocket(appContext: Context, serverPfd: ParcelFileDescriptor) {
        scope.launch {
            val writeMutex = Mutex()
            val currentReqId = AtomicLong(0L)
            val pendingQueue = Channel<IpcRequest>(Channel.CONFLATED)
            val out = FileOutputStream(serverPfd.fileDescriptor)
            val input = FileInputStream(serverPfd.fileDescriptor)

            suspend fun sendResponse(resp: IpcResponse) {
                val bytes = (ipcJson.encodeToString(IpcResponse.serializer(), resp) + "\n").toByteArray()
                writeMutex.withLock {
                    runCatching {
                        out.write(bytes)
                        out.flush()
                    }
                }
            }

            val worker = launch {
                for (req in pendingQueue) {
                    if (currentReqId.get() != req.reqId) continue
                    val apiKey = resolveApiKey(appContext)
                    if (apiKey.isEmpty()) {
                        sendResponse(IpcResponse(type = "error", reqId = req.reqId, error = "API key not configured"))
                        continue
                    }
                    val model = router.selectModel(req.program)
                    Log.i(TAG, "suggest req=${req.reqId} program=\"${req.program}\" model=$model")
                    val dashScope = DashScopeClient(httpClient = httpClient, apiKey = apiKey)
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
                            launch {
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
                            Log.i(TAG, "done req=${req.reqId} latency=${result.latency.inWholeMilliseconds}ms chunks=${result.chunks.size}")
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
                val reader = BufferedReader(InputStreamReader(input))
                while (isActive) {
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
                            if (!router.isAllowed(req.program, resolveUserBlacklist(appContext)) || req.prefix.isEmpty()) {
                                currentReqId.set(0L)
                                sendResponse(IpcResponse(type = "proposal", reqId = req.reqId, text = "", done = true))
                                continue
                            }
                            currentReqId.set(req.reqId)
                            pendingQueue.trySend(req)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "socket closed: ${e.message}")
            } finally {
                currentReqId.set(0L)
                pendingQueue.close()
                worker.cancel()
                runCatching { serverPfd.close() }
            }
        }
    }
}

class LlmCompletionProvider : ContentProvider() {
    override fun onCreate(): Boolean = true

    override fun call(method: String, arg: String?, extras: Bundle?): Bundle? {
        if (method != "openSocket") return null
        if (callingPackage != "org.fcitx.fcitx5.android") {
            Log.w("LlmCompletionProvider", "Rejected caller: $callingPackage")
            return null
        }
        val appContext = context?.applicationContext ?: return null
        val pair = ParcelFileDescriptor.createSocketPair()
        LlmCompletionBridge.attachSocket(appContext, pair[0])
        Log.i("LlmCompletionProvider", "Handed socketpair fd to org.fcitx.fcitx5.android")
        return Bundle().apply {
            putParcelable("fd", pair[1])
        }
    }

    override fun query(uri: Uri, p: Array<out String>?, s: String?, sa: Array<out String>?, so: String?): Cursor? = null
    override fun getType(uri: Uri): String? = null
    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?): Int = 0
}
