#include "text_ops.hpp"
#include "unicode_codec.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sys/stat.h>

namespace fcitx5_llm {

namespace {

bool isHorizontalWs(char c) { return c == ' ' || c == '\t'; }

size_t nextUtf8ScalarBytes(std::string_view s, size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t len = 1;
    if ((c & 0xE0) == 0xC0) {
        len = 2;
    } else if ((c & 0xF0) == 0xE0) {
        len = 3;
    } else if ((c & 0xF8) == 0xF0) {
        len = 4;
    }
    return std::min(len, s.size() - i);
}

std::vector<std::string> splitAtomicUnitsUtf8(std::string_view text) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < text.size()) {
        char c = text[i];
        if (isHorizontalWs(c)) {
            size_t start = i;
            while (i < text.size() && isHorizontalWs(text[i])) {
                ++i;
            }
            out.emplace_back(text.substr(start, i - start));
        } else if (isAsciiWordChar(static_cast<unsigned char>(c))) {
            size_t start = i;
            while (i < text.size() &&
                   isAsciiWordChar(static_cast<unsigned char>(text[i]))) {
                ++i;
            }
            out.emplace_back(text.substr(start, i - start));
        } else {
            size_t step = nextUtf8ScalarBytes(text, i);
            out.emplace_back(text.substr(i, step));
            i += step;
        }
    }
    return out;
}

int unclosedCount(std::string_view s, char open, char close) {
    int depth = 0;
    for (char c : s) {
        if (c == open) {
            ++depth;
        } else if (c == close && depth > 0) {
            --depth;
        }
    }
    return depth;
}

bool closesBracketOpenedInKept(std::string_view kept,
                               std::string_view removed) {
    constexpr std::pair<char, char> kPairs[] = {
        {'(', ')'}, {'[', ']'}, {'{', '}'}};
    std::string combined;
    combined.reserve(kept.size() + removed.size());
    combined.append(kept);
    combined.append(removed);

    for (auto [open, close] : kPairs) {
        if (removed.find(close) != std::string_view::npos &&
            unclosedCount(kept, open, close) >
                unclosedCount(combined, open, close)) {
            return true;
        }
    }
    for (char quote : {'"', '\''}) {
        if (removed.find(quote) != std::string_view::npos) {
            size_t kc = static_cast<size_t>(
                std::count(kept.begin(), kept.end(), quote));
            size_t cc = static_cast<size_t>(
                std::count(combined.begin(), combined.end(), quote));
            if ((kc % 2 == 1) && (cc % 2 == 0)) {
                return true;
            }
        }
    }
    return false;
}

std::string toLowerAscii(std::string_view s) {
    std::string out(s);
    for (char &c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

bool parseJsonStringAt(std::string_view line, size_t &pos, std::string &out) {
    if (pos >= line.size() || line[pos] != '"') {
        return false;
    }
    ++pos;
    out.clear();
    while (pos < line.size()) {
        char c = line[pos++];
        if (c == '"') {
            return true;
        }
        if (c == '\\' && pos < line.size()) {
            char esc = line[pos++];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u':
                if (pos + 4 <= line.size()) {
                    uint32_t cp = 0;
                    bool ok = true;
                    for (int i = 0; i < 4; ++i) {
                        char h = line[pos + static_cast<size_t>(i)];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') {
                            cp |= static_cast<uint32_t>(h - '0');
                        } else if (h >= 'a' && h <= 'f') {
                            cp |= static_cast<uint32_t>(h - 'a' + 10);
                        } else if (h >= 'A' && h <= 'F') {
                            cp |= static_cast<uint32_t>(h - 'A' + 10);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    pos += 4;
                    if (ok) {
                        if (cp <= 0x7Fu) {
                            out.push_back(static_cast<char>(cp));
                        } else if (cp <= 0x7FFu) {
                            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
                            out.push_back(
                                static_cast<char>(0x80u | (cp & 0x3Fu)));
                        } else {
                            out.push_back(
                                static_cast<char>(0xE0u | (cp >> 12)));
                            out.push_back(static_cast<char>(
                                0x80u | ((cp >> 6) & 0x3Fu)));
                            out.push_back(
                                static_cast<char>(0x80u | (cp & 0x3Fu)));
                        }
                    }
                }
                break;
            default:
                out.push_back(esc);
                break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

std::string nowIso8601Local() {
    using namespace std::chrono;
    auto tp = system_clock::now();
    auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tmLocal{};
    localtime_r(&tt, &tmLocal);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03d",
                  tmLocal.tm_year + 1900, tmLocal.tm_mon + 1, tmLocal.tm_mday,
                  tmLocal.tm_hour, tmLocal.tm_min, tmLocal.tm_sec,
                  static_cast<int>(ms.count()));
    return buf;
}

} // namespace

bool isAsciiWordChar(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

std::string trimWhitespace(std::string_view s) {
    size_t a = 0;
    while (a < s.size() &&
           (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
        ++a;
    }
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' ||
                     s[b - 1] == '\n')) {
        --b;
    }
    return std::string(s.substr(a, b - a));
}

std::string jsonEscape(std::string_view in) {
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    return out;
}

bool extractJsonStringField(std::string_view line, std::string_view key,
                            std::string &out) {
    std::string needle = "\"" + std::string(key) + "\":";
    auto pos = line.find(needle);
    if (pos == std::string_view::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) {
        ++pos;
    }
    return parseJsonStringAt(line, pos, out);
}

std::vector<std::string> segmentCompletionChunks(std::string_view text) {
    if (text.empty()) {
        return {};
    }
    auto atomic = splitAtomicUnitsUtf8(text);
    std::vector<std::string> merged;
    std::string pendingWs;

    for (const auto &piece : atomic) {
        if (piece.empty()) {
            continue;
        }
        bool allWs = std::all_of(piece.begin(), piece.end(), isHorizontalWs);
        if (allWs) {
            pendingWs += piece;
            continue;
        }
        bool allWord =
            std::all_of(piece.begin(), piece.end(), [](char c) {
                return isAsciiWordChar(static_cast<unsigned char>(c));
            });
        if (pendingWs.empty() && !merged.empty() && allWord &&
            !merged.back().empty() &&
            isAsciiWordChar(static_cast<unsigned char>(merged.back().back()))) {
            merged.back() += piece;
        } else {
            merged.push_back(pendingWs + piece);
            pendingWs.clear();
        }
    }
    if (!pendingWs.empty()) {
        if (!merged.empty()) {
            merged.back() += pendingWs;
        } else {
            merged.push_back(std::move(pendingWs));
        }
    }
    return merged;
}

std::string trimSuffixOverlap(std::string_view generated,
                              std::string_view suffix) {
    if (generated.empty() || suffix.empty()) {
        return std::string(generated);
    }
    auto nlPos = suffix.find_first_of("\r\n");
    std::string_view sLine =
        (nlPos == std::string_view::npos) ? suffix : suffix.substr(0, nlPos);
    if (sLine.empty()) {
        return std::string(generated);
    }

    size_t maxK = std::min(generated.size(), sLine.size());
    for (size_t k = maxK; k >= 1; --k) {
        std::string_view cand = sLine.substr(0, k);
        if (generated.substr(generated.size() - k) != cand) {
            continue;
        }
        if (k == 1 && isAsciiWordChar(static_cast<unsigned char>(cand[0]))) {
            continue;
        }
        // Ensure cut falls on valid UTF-8 codepoint boundary
        size_t cutIdx = generated.size() - k;
        if (cutIdx < generated.size() &&
            (static_cast<unsigned char>(generated[cutIdx]) & 0xC0) == 0x80) {
            continue;
        }
        std::string_view kept = generated.substr(0, cutIdx);
        if (closesBracketOpenedInKept(kept, cand)) {
            continue;
        }
        return std::string(kept);
    }
    return std::string(generated);
}

int32_t minRosaMatchLenForPrefix(std::string_view prefix) {
    (void)prefix;
    return 2;
}

AppBlacklist::AppBlacklist(std::filesystem::path blacklistFile)
    : path_(std::move(blacklistFile)),
      builtin_{"keepassxc",
               "bitwarden",
               "1password",
               "pinentry",
               "pinentry-gnome3",
               "pinentry-gtk-2",
               "gcr-prompter",
               "polkit-gnome-authentication-agent-1",
               "com.x8bit.bitwarden",
               "com.bitwarden.mobile",
               "com.agilebits.onepassword",
               "keepass2android.keepass2android",
               "keepass2android.keepass2android_nonet",
               "com.kunzisoft.keepass.free",
               "com.kunzisoft.keepass.libre",
               "proton.android.pass",
               "com.google.android.apps.authenticator2",
               "com.azure.authenticator",
               "com.twofasapp"} {}

void AppBlacklist::reloadIfChangedLocked() {
    if (path_.empty()) {
        return;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path_, ec)) {
        if (hasLoadedOnce_) {
            user_.clear();
            hasLoadedOnce_ = false;
        }
        return;
    }
    auto mt = std::filesystem::last_write_time(path_, ec);
    if (!ec && hasLoadedOnce_ && mt == lastMtime_) {
        return;
    }
    lastMtime_ = mt;
    hasLoadedOnce_ = true;
    user_.clear();
    std::ifstream ifs(path_);
    std::string line;
    while (std::getline(ifs, line)) {
        std::string t = trimWhitespace(line);
        if (!t.empty() && t[0] != '#') {
            user_.insert(toLowerAscii(t));
        }
    }
}

bool AppBlacklist::isAllowed(std::string_view program) {
    std::string p = toLowerAscii(trimWhitespace(program));
    if (p.empty()) {
        return true;
    }
    std::lock_guard<std::mutex> lk(mutex_);
    reloadIfChangedLocked();
    if (builtin_.contains(p)) {
        return false;
    }
    return !user_.contains(p);
}

PredictionLogger::PredictionLogger(std::filesystem::path logPath,
                                   size_t maxBytes, int maxBackups)
    : logPath_(std::move(logPath)), maxBytes_(maxBytes),
      maxBackups_(maxBackups) {}

std::string PredictionLogger::previewTail(std::string_view s, size_t maxChars) {
    return utf8TailChars(s, maxChars);
}

std::string PredictionLogger::previewHead(std::string_view s, size_t maxChars) {
    return utf8HeadChars(s, maxChars);
}

void PredictionLogger::rotateLocked() {
    std::error_code ec;
    for (int i = maxBackups_; i >= 1; --i) {
        std::filesystem::path src =
            (i == 1) ? logPath_
                     : std::filesystem::path(logPath_.string() + "." +
                                             std::to_string(i - 1));
        std::filesystem::path dst(logPath_.string() + "." + std::to_string(i));
        if (std::filesystem::exists(src, ec)) {
            std::filesystem::rename(src, dst, ec);
            ::chmod(dst.c_str(), 0600);
        }
    }
}

void PredictionLogger::log(const TapeLogRecord &rec) {
    if (logPath_.empty()) {
        return;
    }
    uint64_t seq = ++nextSeq_;
    std::string line;
    line.reserve(256);
    line += "{\"seq\":" + std::to_string(seq) + ",\"ts\":\"" +
            jsonEscape(nowIso8601Local()) + "\",\"event\":\"" +
            jsonEscape(rec.event) + "\"";
    if (rec.reqId != 0) {
        line += ",\"req_id\":" + std::to_string(rec.reqId);
    }
    if (!rec.program.empty()) {
        line += ",\"program\":\"" + jsonEscape(rec.program) + "\"";
    }
    if (!rec.trigger.empty()) {
        line += ",\"trigger\":\"" + jsonEscape(rec.trigger) + "\"";
    }
    if (!rec.model.empty()) {
        line += ",\"model\":\"" + jsonEscape(rec.model) + "\"";
    }
    if (!rec.mode.empty()) {
        line += ",\"mode\":\"" + jsonEscape(rec.mode) + "\"";
    }
    if (rec.surroundingValid.has_value()) {
        line += std::string(",\"surrounding_valid\":") +
                (*rec.surroundingValid ? "true" : "false");
    }
    if (rec.cursor >= 0) {
        line += ",\"cursor\":" + std::to_string(rec.cursor);
    }
    if (!rec.pendingCommit.empty()) {
        line += ",\"pending_commit\":\"" + jsonEscape(rec.pendingCommit) + "\"";
    }
    if (rec.prefixLen.has_value()) {
        line += ",\"prefix_len\":" + std::to_string(*rec.prefixLen);
    }
    if (!rec.prefixTail.empty()) {
        line += ",\"prefix_tail\":\"" + jsonEscape(rec.prefixTail) + "\"";
    }
    if (rec.suffixLen.has_value()) {
        line += ",\"suffix_len\":" + std::to_string(*rec.suffixLen);
    }
    if (!rec.suffixHead.empty()) {
        line += ",\"suffix_head\":\"" + jsonEscape(rec.suffixHead) + "\"";
    }
    if (rec.ttftMs.has_value()) {
        line += ",\"ttft_ms\":" + std::to_string(*rec.ttftMs);
    }
    if (rec.latencyMs.has_value()) {
        line += ",\"latency_ms\":" + std::to_string(*rec.latencyMs);
    }
    if (rec.shownMs.has_value()) {
        line += ",\"shown_ms\":" + std::to_string(*rec.shownMs);
    }
    if (!rec.finishReason.empty()) {
        line += ",\"finish_reason\":\"" + jsonEscape(rec.finishReason) + "\"";
    }
    if (!rec.rawText.empty()) {
        line += ",\"raw_text\":\"" + jsonEscape(rec.rawText) + "\"";
    }
    if (!rec.text.empty()) {
        line += ",\"text\":\"" + jsonEscape(rec.text) + "\"";
    }
    if (!rec.chunks.empty()) {
        line += ",\"chunks\":[";
        for (size_t i = 0; i < rec.chunks.size(); ++i) {
            if (i > 0) {
                line.push_back(',');
            }
            line += "\"" + jsonEscape(rec.chunks[i]) + "\"";
        }
        line.push_back(']');
    }
    if (!rec.accepted.empty()) {
        line += ",\"accepted\":\"" + jsonEscape(rec.accepted) + "\"";
    }
    if (!rec.remaining.empty()) {
        line += ",\"remaining\":\"" + jsonEscape(rec.remaining) + "\"";
    }
    if (!rec.reason.empty()) {
        line += ",\"reason\":\"" + jsonEscape(rec.reason) + "\"";
    }
    if (!rec.error.empty()) {
        line += ",\"error\":\"" + jsonEscape(rec.error) + "\"";
    }
    line += "}\n";

    std::lock_guard<std::mutex> lk(mutex_);
    std::error_code ec;
    if (!dirEnsured_) {
        auto parent = logPath_.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            ::chmod(parent.c_str(), 0700);
        }
        dirEnsured_ = true;
    }
    if (std::filesystem::is_regular_file(logPath_, ec) &&
        std::filesystem::file_size(logPath_, ec) + line.size() > maxBytes_) {
        rotateLocked();
    }
    FILE *fp = std::fopen(logPath_.c_str(), "ab");
    if (fp) {
        ::chmod(logPath_.c_str(), 0600);
        std::fwrite(line.data(), 1, line.size(), fp);
        std::fflush(fp);
        std::fclose(fp);
    }
}

} // namespace fcitx5_llm
