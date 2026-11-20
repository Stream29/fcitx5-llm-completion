@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.stream29.fcitx5llm.daemon

import io.github.stream29.fcitx5llm.AppRouter
import io.github.stream29.fcitx5llm.DashScopeClient
import io.ktor.client.HttpClient
import io.ktor.client.engine.curl.Curl
import kotlinx.cinterop.staticCFunction
import kotlinx.coroutines.runBlocking
import platform.posix.SIGINT
import platform.posix.SIGPIPE
import platform.posix.SIGTERM
import platform.posix.SIG_IGN
import platform.posix.exit
import platform.posix.signal
import kotlin.time.Duration.Companion.milliseconds

private const val VERSION = "0.1.0"
private var activeServer: UnixSocketServer? = null

fun main(args: Array<String>) {
    var socketPath = defaultSocketPath()
    var envFile: String? = null
    var blacklistPath = defaultBlacklistPath()
    var timeoutMs = 4000L
    var baseUrl = DashScopeClient.DEFAULT_BASE_URL

    var i = 0
    while (i < args.size) {
        when (val arg = args[i]) {
            "--version", "-v" -> {
                println("fcitx5-llm-completion-daemon $VERSION")
                return
            }
            "--help", "-h" -> {
                println(
                    """
                    fcitx5-llm-completion-daemon $VERSION
                    Usage: fcitx5-llm-completion-daemon [options]
                      --socket <path>       Unix domain socket path (default: $socketPath)
                      --env-file <path>     Env file containing DASHSCOPE_API_KEY
                      --blacklist <path>    Per-app blacklist file (default: $blacklistPath)
                      --timeout-ms <ms>     Per-request timeout in ms (default: $timeoutMs)
                      --base-url <url>      DashScope OpenAI-compatible base URL
                      --version, -v         Show version
                    """.trimIndent(),
                )
                return
            }
            "--socket" -> { socketPath = args.getOrNull(++i) ?: error("--socket requires a value") }
            "--env-file" -> { envFile = args.getOrNull(++i) ?: error("--env-file requires a value") }
            "--blacklist" -> { blacklistPath = args.getOrNull(++i) ?: error("--blacklist requires a value") }
            "--timeout-ms" -> { timeoutMs = args.getOrNull(++i)?.toLongOrNull() ?: error("--timeout-ms requires an integer") }
            "--base-url" -> { baseUrl = args.getOrNull(++i) ?: error("--base-url requires a value") }
            else -> error("Unknown argument: $arg (use --help)")
        }
        i++
    }

    val apiKey = resolveApiKey(envFile)
    signal(SIGPIPE, SIG_IGN)

    val httpClient = HttpClient(Curl)
    val dashScope = DashScopeClient(httpClient = httpClient, apiKey = apiKey, baseUrl = baseUrl)
    val server = UnixSocketServer(
        socketPath = socketPath,
        dashScope = dashScope,
        router = AppRouter(),
        blacklist = BlacklistWatcher(blacklistPath),
        requestTimeout = timeoutMs.milliseconds,
    )
    activeServer = server

    val onSignal = staticCFunction<Int, Unit> { _ ->
        activeServer?.stop()
        exit(0)
    }
    signal(SIGINT, onSignal)
    signal(SIGTERM, onSignal)

    println("fcitx5-llm-completion-daemon $VERSION (blacklist: $blacklistPath)")
    try {
        runBlocking {
            server.serve(this)
        }
    } finally {
        server.stop()
        httpClient.close()
    }
}
