#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace fcitx5_llm {

bool isAsciiWordChar(unsigned char c);
std::string trimWhitespace(std::string_view s);
std::string jsonEscape(std::string_view in);
bool extractJsonStringField(std::string_view line, std::string_view key,
                            std::string &out);

std::vector<std::string> segmentCompletionChunks(std::string_view text);
std::string trimSuffixOverlap(std::string_view generated,
                              std::string_view suffix);
int32_t minRosaMatchLenForPrefix(std::string_view prefix);

class AppBlacklist {
public:
    explicit AppBlacklist(std::filesystem::path blacklistFile = {});
    bool isAllowed(std::string_view program);

private:
    void reloadIfChangedLocked();

    std::filesystem::path path_;
    std::filesystem::file_time_type lastMtime_{};
    bool hasLoadedOnce_ = false;
    std::unordered_set<std::string> builtin_;
    std::unordered_set<std::string> user_;
    std::mutex mutex_;
};

struct TapeLogRecord {
    std::string event;
    uint64_t reqId = 0;
    std::string program;
    std::string trigger;
    std::string model;
    std::string mode;
    std::optional<bool> surroundingValid;
    int cursor = -1;
    std::string pendingCommit;
    std::optional<size_t> prefixLen;
    std::string prefixTail;
    std::optional<size_t> suffixLen;
    std::string suffixHead;
    std::optional<int64_t> ttftMs;
    std::optional<int64_t> latencyMs;
    std::optional<int64_t> shownMs;
    std::string finishReason;
    std::string rawText;
    std::string text;
    std::vector<std::string> chunks;
    std::string accepted;
    std::string remaining;
    std::string reason;
    std::string error;
};

class PredictionLogger {
public:
    explicit PredictionLogger(std::filesystem::path logPath,
                              size_t maxBytes = 10 * 1024 * 1024,
                              int maxBackups = 5);

    const std::filesystem::path &logPath() const { return logPath_; }
    void log(const TapeLogRecord &rec);

    static std::string previewTail(std::string_view s, size_t maxChars = 64);
    static std::string previewHead(std::string_view s, size_t maxChars = 64);

private:
    void rotateLocked();

    std::filesystem::path logPath_;
    size_t maxBytes_;
    int maxBackups_;
    std::atomic<uint64_t> nextSeq_{0};
    bool dirEnsured_ = false;
    std::mutex mutex_;
};

} // namespace fcitx5_llm
