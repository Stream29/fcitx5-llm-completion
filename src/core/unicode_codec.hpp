#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fcitx5_llm {

inline constexpr int32_t TOK_END = 0;
inline constexpr int32_t TOK_PAD = 1;
inline constexpr int32_t TOK_ACCEPT = 2;
inline constexpr int32_t TOK_PREDICT = 3;
inline constexpr int32_t TOK_FOCUS = 4;
inline constexpr int32_t TOK_AFTER = 5;
inline constexpr int32_t TOK_BEFORE = 6;
inline constexpr int32_t TOK_TYPE = 7;
inline constexpr int32_t TOK_PROPOSE = 8;
inline constexpr int32_t TOK_REJECT = 9;
inline constexpr int32_t BYTE_OFFSET = 10;
inline constexpr int32_t CHAR_OFFSET = 266;

std::string stripInvisibleFormattingUtf8(std::string_view in);
std::string popLastUtf8Char(std::string &s);
std::string utf8TailBytes(std::string_view s, size_t maxBytes);
std::string utf8HeadBytes(std::string_view s, size_t maxBytes);
std::string utf8TailChars(std::string_view s, size_t maxChars);
std::string utf8HeadChars(std::string_view s, size_t maxChars);
size_t utf8CharCount(std::string_view s);

class PureUnicodeCodec {
public:
    PureUnicodeCodec() = default;
    PureUnicodeCodec(const std::filesystem::path &vocabJsonPath,
                     int32_t vocabSize);
    PureUnicodeCodec(const std::vector<uint32_t> &codepoints, int32_t vocabSize);

    std::vector<int32_t> encode(std::string_view text) const;
    std::string decode(std::span<const int32_t> ids) const;

    int32_t vocabSize() const { return vocabSize_; }
    const std::unordered_set<int32_t> &newlineTokenIds() const {
        return newlineTokIds_;
    }

private:
    int32_t vocabSize_ = 4096;
    std::unordered_map<uint32_t, int32_t> cp2id_;
    std::unordered_map<int32_t, uint32_t> id2cp_;
    std::unordered_set<int32_t> newlineTokIds_;
};

} // namespace fcitx5_llm
