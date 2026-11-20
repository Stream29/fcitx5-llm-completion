package io.github.stream29.fcitx5llm

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

val ipcJson: Json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = false
}

@Serializable
data class IpcRequest(
    val type: String, // "suggest" | "cancel" | "ping"
    @SerialName("req_id")
    val reqId: Long = 0,
    val program: String = "",
    val im: String = "",
    val prefix: String = "",
    val suffix: String = "",
    @SerialName("max_tokens")
    val maxTokens: Int = 0,
)

@Serializable
data class IpcResponse(
    val type: String, // "proposal" | "pong" | "error"
    @SerialName("req_id")
    val reqId: Long? = null,
    val text: String? = null,
    val chunks: List<String>? = null,
    val done: Boolean? = null,
    val model: String? = null,
    @SerialName("latency_ms")
    val latencyMs: Long? = null,
    val error: String? = null,
)
