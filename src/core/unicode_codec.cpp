#include "unicode_codec.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fcitx5_llm {

namespace {

bool decodeNextUtf8Codepoint(std::string_view s, size_t &i, uint32_t &cp,
                             size_t &byteLen) {
    if (i >= s.size()) {
        return false;
    }
    unsigned char b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        cp = b0;
        byteLen = 1;
        i += 1;
        return true;
    }
    if ((b0 & 0xE0) == 0xC0 && i + 2 <= s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[i + 1]);
        if ((b1 & 0xC0) == 0x80) {
            cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
            byteLen = 2;
            i += 2;
            return true;
        }
    } else if ((b0 & 0xF0) == 0xE0 && i + 3 <= s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[i + 2]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80) {
            cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
            byteLen = 3;
            i += 3;
            return true;
        }
    } else if ((b0 & 0xF8) == 0xF0 && i + 4 <= s.size()) {
        unsigned char b1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char b2 = static_cast<unsigned char>(s[i + 2]);
        unsigned char b3 = static_cast<unsigned char>(s[i + 3]);
        if ((b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 &&
            (b3 & 0xC0) == 0x80) {
            cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) |
                 ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
            byteLen = 4;
            i += 4;
            return true;
        }
    }
    cp = b0;
    byteLen = 1;
    i += 1;
    return false;
}

void appendCodepointUtf8(uint32_t cp, std::string &out) {
    if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

std::vector<uint32_t> parseCodepointsArrayJson(const std::string &jsonText) {
    auto keyPos = jsonText.find("\"codepoints\"");
    if (keyPos == std::string::npos) {
        throw std::runtime_error("Missing \"codepoints\" in unicode_vocab.json");
    }
    auto brOpen = jsonText.find('[', keyPos);
    auto brClose = jsonText.find(']', brOpen);
    if (brOpen == std::string::npos || brClose == std::string::npos) {
        throw std::runtime_error(
            "Malformed \"codepoints\" array in unicode_vocab.json");
    }
    std::vector<uint32_t> cps;
    size_t i = brOpen + 1;
    while (i < brClose) {
        while (i < brClose && (jsonText[i] < '0' || jsonText[i] > '9')) {
            ++i;
        }
        if (i >= brClose) {
            break;
        }
        uint32_t val = 0;
        while (i < brClose && jsonText[i] >= '0' && jsonText[i] <= '9') {
            val = val * 10u + static_cast<uint32_t>(jsonText[i] - '0');
            ++i;
        }
        cps.push_back(val);
    }
    return cps;
}

} // namespace

std::string stripInvisibleFormattingUtf8(std::string_view in) {
    if (in.empty()) {
        return {};
    }
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (i + 3 <= in.size()) {
            unsigned char b0 = static_cast<unsigned char>(in[i]);
            unsigned char b1 = static_cast<unsigned char>(in[i + 1]);
            unsigned char b2 = static_cast<unsigned char>(in[i + 2]);
            if ((b0 == 0xE2 && b1 == 0x80 && b2 == 0x8B) ||
                (b0 == 0xE2 && b1 == 0x81 && b2 == 0xA0) ||
                (b0 == 0xEF && b1 == 0xBB && b2 == 0xBF)) {
                i += 3;
                continue;
            }
        }
        out.push_back(in[i]);
        ++i;
    }
    return out;
}

std::string popLastUtf8Char(std::string &s) {
    if (s.empty()) {
        return {};
    }
    size_t i = s.size();
    while (i > 0) {
        --i;
        unsigned char c = static_cast<unsigned char>(s[i]);
        if ((c & 0xC0) != 0x80) {
            break;
        }
    }
    std::string popped = s.substr(i);
    s.erase(i);
    return popped;
}

std::string utf8TailBytes(std::string_view s, size_t maxBytes) {
    if (s.size() <= maxBytes) {
        return std::string(s);
    }
    size_t cut = s.size() - maxBytes;
    while (cut < s.size() &&
           (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) {
        ++cut;
    }
    return std::string(s.substr(cut));
}

std::string utf8HeadBytes(std::string_view s, size_t maxBytes) {
    if (s.size() <= maxBytes) {
        return std::string(s);
    }
    size_t end = maxBytes;
    while (end > 0 &&
           (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) {
        --end;
    }
    return std::string(s.substr(0, end));
}

std::string utf8TailChars(std::string_view s, size_t maxChars) {
    if (maxChars == 0 || s.empty()) {
        return {};
    }
    std::vector<size_t> starts;
    starts.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            starts.push_back(i);
        }
    }
    if (starts.size() <= maxChars) {
        return std::string(s);
    }
    return std::string(s.substr(starts[starts.size() - maxChars]));
}

std::string utf8HeadChars(std::string_view s, size_t maxChars) {
    if (maxChars == 0 || s.empty()) {
        return {};
    }
    size_t count = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
            if (count == maxChars) {
                return std::string(s.substr(0, i));
            }
            ++count;
        }
    }
    return std::string(s);
}

size_t utf8CharCount(std::string_view s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

PureUnicodeCodec::PureUnicodeCodec(const std::filesystem::path &vocabJsonPath,
                                   int32_t vocabSize) {
    std::ifstream ifs(vocabJsonPath);
    if (!ifs) {
        throw std::runtime_error("Cannot open unicode_vocab.json: " +
                                 vocabJsonPath.string());
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    *this = PureUnicodeCodec(parseCodepointsArrayJson(ss.str()), vocabSize);
}

PureUnicodeCodec::PureUnicodeCodec(const std::vector<uint32_t> &codepoints,
                                   int32_t vocabSize)
    : vocabSize_(vocabSize) {
    cp2id_.reserve(codepoints.size());
    id2cp_.reserve(codepoints.size());
    for (size_t i = 0; i < codepoints.size(); ++i) {
        int32_t tid = CHAR_OFFSET + static_cast<int32_t>(i);
        if (tid >= vocabSize_) {
            break;
        }
        cp2id_[codepoints[i]] = tid;
        id2cp_[tid] = codepoints[i];
    }
    for (int32_t t : encode("\n")) {
        newlineTokIds_.insert(t);
    }
    for (int32_t t : encode("\r")) {
        newlineTokIds_.insert(t);
    }
}

std::vector<int32_t> PureUnicodeCodec::encode(std::string_view text) const {
    std::vector<int32_t> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        uint32_t cp = 0;
        size_t byteLen = 0;
        bool valid = decodeNextUtf8Codepoint(text, i, cp, byteLen);
        if (valid) {
            if (auto it = cp2id_.find(cp); it != cp2id_.end()) {
                out.push_back(it->second);
                continue;
            }
        }
        for (size_t b = 0; b < byteLen; ++b) {
            int32_t tid = BYTE_OFFSET +
                          static_cast<unsigned char>(text[start + b]);
            if (tid < vocabSize_) {
                out.push_back(tid);
            }
        }
    }
    return out;
}

std::string PureUnicodeCodec::decode(std::span<const int32_t> ids) const {
    std::string raw;
    raw.reserve(ids.size() * 3);
    for (int32_t tid : ids) {
        if (tid >= BYTE_OFFSET && tid < CHAR_OFFSET) {
            raw.push_back(static_cast<char>(tid - BYTE_OFFSET));
        } else if (auto it = id2cp_.find(tid); it != id2cp_.end()) {
            appendCodepointUtf8(it->second, raw);
        }
    }
    // Strip any incomplete trailing multi-byte UTF-8 fragment
    std::string clean;
    clean.reserve(raw.size());
    size_t i = 0;
    while (i < raw.size()) {
        size_t start = i;
        uint32_t cp = 0;
        size_t byteLen = 0;
        if (decodeNextUtf8Codepoint(raw, i, cp, byteLen)) {
            clean.append(raw, start, byteLen);
        }
    }
    return clean;
}

} // namespace fcitx5_llm
