@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.stream29.fcitx5llm.daemon

import io.github.stream29.fcitx5llm.AppRouter
import io.github.stream29.fcitx5llm.DashScopeClient
import io.github.stream29.fcitx5llm.IpcRequest
import io.github.stream29.fcitx5llm.IpcResponse
import io.github.stream29.fcitx5llm.SuggestRequest
import io.github.stream29.fcitx5llm.ipcJson
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.sizeOf
import kotlinx.cinterop.usePinned
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.IO
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import platform.linux.sockaddr_un
import platform.posix.AF_UNIX
import platform.posix.EINTR
import platform.posix.MSG_NOSIGNAL
import platform.posix.SOCK_STREAM
import platform.posix.accept
import platform.posix.bind
import platform.posix.chmod
import platform.posix.close
import platform.posix.errno
import platform.posix.listen
import platform.posix.memcpy
import platform.posix.memset
import platform.posix.read
import platform.posix.send
import platform.posix.sockaddr
import platform.posix.socket
import platform.posix.unlink
import kotlin.concurrent.AtomicInt
import kotlin.concurrent.AtomicLong
import kotlin.time.Duration

internal class UnixSocketServer(
    private val socketPath: String,
    private val dashScope: DashScopeClient,
    private val router: AppRouter,
    private val blacklist: BlacklistWatcher,
    private val requestTimeout: Duration,
) {
    private val listenFd = AtomicInt(-1)
    // Serialize libcurl calls across all clients to avoid ktor-client-curl multi-thread cancel races.
    private val curlMutex = Mutex()

    fun stop() {
        val fd = listenFd.getAndSet(-1)
        if (fd >= 0) {
            close(fd)
        }
        unlink(socketPath)
    }

    fun serve(scope: CoroutineScope) {
        unlink(socketPath)
        val fd = socket(AF_UNIX, SOCK_STREAM, 0)
        if (fd < 0) error("socket(AF_UNIX) failed: errno=$errno")

        memScoped {
            val addr = alloc<sockaddr_un>()
            memset(addr.ptr, 0, sizeOf<sockaddr_un>().convert())
            addr.sun_family = AF_UNIX.convert()
            val pathBytes = socketPath.encodeToByteArray()
            if (pathBytes.size >= 107) {
                close(fd)
                error("socket path too long: $socketPath")
            }
            pathBytes.usePinned { pinned ->
                memcpy(addr.sun_path, pinned.addressOf(0), pathBytes.size.convert())
            }
            if (bind(fd, addr.ptr.reinterpret<sockaddr>(), sizeOf<sockaddr_un>().convert()) < 0) {
                val e = errno
                close(fd)
                error("bind($socketPath) failed: errno=$e")
            }
        }
        chmod(socketPath, 0x180u) // 0600
        if (listen(fd, 16) < 0) {
            val e = errno
            close(fd)
            unlink(socketPath)
            error("listen($socketPath) failed: errno=$e")
        }

        listenFd.value = fd
        println("listening on $socketPath")

        try {
            while (scope.isActive && listenFd.value >= 0) {
                val clientFd = accept(fd, null, null)
                if (clientFd < 0) {
                    if (errno == EINTR) continue
                    break
                }
                scope.launch(Dispatchers.IO) {
                    handleClient(this, clientFd)
                }
            }
        } finally {
            stop()
        }
    }

    private suspend fun handleClient(scope: CoroutineScope, clientFd: Int) {
        val writeMutex = Mutex()
        val currentReqId = AtomicLong(0L)
        // CONFLATED queue keeps only the latest pending suggest request during rapid typing.
        val pendingQueue = Channel<IpcRequest>(Channel.CONFLATED)

        suspend fun sendResponse(resp: IpcResponse) {
            val line = ipcJson.encodeToString(IpcResponse.serializer(), resp) + "\n"
            val bytes = line.encodeToByteArray()
            writeMutex.withLock {
                bytes.usePinned { pinned ->
                    var offset = 0
                    while (offset < bytes.size) {
                        val n = send(
                            clientFd,
                            pinned.addressOf(offset),
                            (bytes.size - offset).convert(),
                            MSG_NOSIGNAL,
                        )
                        if (n <= 0) break
                        offset += n.toInt()
                    }
                }
            }
        }

        val workerJob = scope.launch(Dispatchers.IO) {
            for (req in pendingQueue) {
                if (currentReqId.value != req.reqId) continue

                val model = router.selectModel(req.program)
                println("suggest req=${req.reqId} program=\"${req.program}\" model=$model")

                try {
                    val result = curlMutex.withLock {
                        if (currentReqId.value != req.reqId) return@withLock null
                        dashScope.suggestStream(
                            req = SuggestRequest(
                                model = model,
                                prefix = req.prefix,
                                maxTokens = req.maxTokens,
                            ),
                            shouldStop = { currentReqId.value != req.reqId },
                        ) { accumulated, chunks ->
                            if (currentReqId.value != req.reqId) return@suggestStream
                            scope.launch(Dispatchers.IO) {
                                if (currentReqId.value == req.reqId) {
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
                    } ?: continue

                    if (currentReqId.value == req.reqId) {
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
                    if (currentReqId.value == req.reqId) {
                        sendResponse(
                            IpcResponse(
                                type = "error",
                                reqId = req.reqId,
                                model = model,
                                error = e.message ?: e::class.simpleName ?: "unknown error",
                            ),
                        )
                    }
                }
            }
        }

        val readBuf = ByteArray(4096)
        val lineBuf = StringBuilder()

        try {
            while (scope.isActive) {
                val n = readBuf.usePinned { pinned ->
                    read(clientFd, pinned.addressOf(0), readBuf.size.convert())
                }.toInt()
                if (n < 0) {
                    if (errno == EINTR) continue
                    break
                }
                if (n == 0) break

                lineBuf.append(readBuf.decodeToString(0, n))
                while (true) {
                    val nl = lineBuf.indexOf('\n')
                    if (nl < 0) break
                    val line = lineBuf.substring(0, nl).trim()
                    lineBuf.deleteRange(0, nl + 1)
                    if (line.isEmpty()) continue

                    val req = runCatching {
                        ipcJson.decodeFromString(IpcRequest.serializer(), line)
                    }.getOrElse { err ->
                        sendResponse(IpcResponse(type = "error", error = "invalid json: ${err.message}"))
                        continue
                    }

                    when (req.type) {
                        "ping" -> sendResponse(IpcResponse(type = "pong", reqId = req.reqId))

                        "cancel" -> {
                            currentReqId.value = 0L
                        }

                        "suggest" -> {
                            val userBlacklist = blacklist.current()
                            if (!router.isAllowed(req.program, userBlacklist) || req.prefix.isEmpty()) {
                                currentReqId.value = 0L
                                sendResponse(IpcResponse(type = "proposal", reqId = req.reqId, text = "", done = true))
                                continue
                            }
                            currentReqId.value = req.reqId
                            pendingQueue.trySend(req)
                        }

                        else -> sendResponse(IpcResponse(type = "error", reqId = req.reqId, error = "unknown type"))
                    }
                }
            }
        } finally {
            currentReqId.value = 0L
            pendingQueue.close()
            workerJob.cancel()
            close(clientFd)
        }
    }
}
