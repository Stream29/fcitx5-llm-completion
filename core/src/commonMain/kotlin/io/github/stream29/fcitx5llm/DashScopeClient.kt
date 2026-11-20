package io.github.stream29.fcitx5llm

import io.ktor.client.HttpClient
import io.ktor.client.request.header
import io.ktor.client.request.preparePost
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsChannel
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.utils.io.readUTF8Line
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlin.time.Duration
import kotlin.time.TimeSource

data class SuggestRequest(
    val model: String = AppRouter.DEFAULT_TEXT_MODEL,
    val prefix: String,
    val maxTokens: Int = 8,
    val temperature: Double = 0.2,
    val stop: List<String> = listOf("\n"),
)

data class SuggestResult(
    val text: String,
    val chunks: List<String>,
    val model: String,
    val latency: Duration,
    val finishReason: String = "",
)

@Serializable
private data class ChatMessage(
    val role: String,
    val content: String,
    val partial: Boolean? = null,
)

@Serializable
private data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>,
    @SerialName("max_tokens")
    val maxTokens: Int,
    val temperature: Double,
    val stop: List<String>,
    val stream: Boolean,
    @SerialName("enable_thinking")
    val enableThinking: Boolean,
)

@Serializable
private data class ChatStreamChunk(
    val choices: List<Choice> = emptyList(),
) {
    @Serializable
    data class Choice(
        val delta: Delta = Delta(),
        @SerialName("finish_reason")
        val finishReason: String? = null,
    )

    @Serializable
    data class Delta(
        val content: String? = null,
    )
}

class DashScopeClient(
    private val httpClient: HttpClient,
    private val apiKey: String,
    baseUrl: String = DEFAULT_BASE_URL,
) {
    private val baseUrl = baseUrl.trimEnd('/')
    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    suspend fun suggestStream(
        req: SuggestRequest,
        shouldStop: () -> Boolean = { false },
        onUpdate: (accumulated: String, chunks: List<String>) -> Unit = { _, _ -> },
    ): SuggestResult {
        val (trimmedPrefix, trailingWS) = preparePrefix(req.prefix)
        if (trimmedPrefix.isEmpty()) {
            return SuggestResult(
                text = "",
                chunks = emptyList(),
                model = req.model,
                latency = Duration.ZERO,
            )
        }

        val maxTokens = if (req.maxTokens > 0) req.maxTokens else 8
        val payload = ChatCompletionRequest(
            model = req.model.ifEmpty { AppRouter.DEFAULT_TEXT_MODEL },
            messages = listOf(
                ChatMessage(
                    role = "assistant",
                    content = trimmedPrefix,
                    partial = true,
                ),
            ),
            maxTokens = maxTokens,
            temperature = req.temperature,
            stop = req.stop.ifEmpty { listOf("\n") },
            stream = true,
            enableThinking = false,
        )
        val bodyJson = json.encodeToString(ChatCompletionRequest.serializer(), payload)

        val mark = TimeSource.Monotonic.markNow()
        val acc = StringBuilder()
        val chunks = mutableListOf<String>()
        var finishReason = ""
        var pendingWS = trailingWS

        httpClient.preparePost("$baseUrl/chat/completions") {
            header(HttpHeaders.Authorization, "Bearer $apiKey")
            header(HttpHeaders.Accept, "text/event-stream")
            contentType(ContentType.Application.Json)
            setBody(bodyJson)
        }.execute { response ->
            if (!response.status.isSuccess()) {
                val errBody = response.bodyAsText().take(512)
                throw IllegalStateException("dashscope HTTP ${response.status.value}: $errBody")
            }

            val channel = response.bodyAsChannel()
            while (!shouldStop()) {
                val rawLine = channel.readUTF8Line() ?: break
                if (shouldStop()) break
                val line = rawLine.trim()
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data == "[DONE]") break

                val parsed = runCatching {
                    json.decodeFromString(ChatStreamChunk.serializer(), data)
                }.getOrNull() ?: continue
                val choice = parsed.choices.firstOrNull() ?: continue
                if (choice.finishReason != null) {
                    finishReason = choice.finishReason
                }
                var delta = choice.delta.content.orEmpty()
                if (delta.isEmpty()) continue

                if (pendingWS.isNotEmpty()) {
                    when {
                        delta.startsWith(pendingWS) -> {
                            delta = delta.removePrefix(pendingWS)
                            pendingWS = ""
                        }
                        pendingWS.startsWith(delta) -> {
                            pendingWS = pendingWS.removePrefix(delta)
                            delta = ""
                        }
                        else -> {
                            pendingWS = ""
                        }
                    }
                }
                if (delta.isEmpty()) continue

                acc.append(delta)
                chunks.add(delta)
                onUpdate(acc.toString(), chunks.toList())
            }
        }

        return SuggestResult(
            text = acc.toString(),
            chunks = chunks.toList(),
            model = payload.model,
            latency = mark.elapsedNow(),
            finishReason = finishReason,
        )
    }

    companion object {
        const val DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        const val MAX_PREFIX_CHARS = 2048

        internal fun preparePrefix(rawPrefix: String, maxChars: Int = MAX_PREFIX_CHARS): Pair<String, String> {
            val tail = if (rawPrefix.length <= maxChars) {
                rawPrefix
            } else {
                rawPrefix.substring(rawPrefix.length - maxChars)
            }
            val trimmed = tail.trimEnd(' ', '\t')
            val trailingWS = tail.substring(trimmed.length)
            return trimmed to trailingWS
        }
    }
}
