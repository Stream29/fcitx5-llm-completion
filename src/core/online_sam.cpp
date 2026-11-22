#include "online_sam.hpp"
#include "text_ops.hpp"

#include <algorithm>
#include <fstream>
#include <string>

namespace fcitx5_llm {

EventAwareOnlineSAM::EventAwareOnlineSAM(size_t maxTokens)
    : maxTokens_(maxTokens) {
    resetAll();
}

void EventAwareOnlineSAM::resetAll() {
    states_.clear();
    states_.reserve(4096);
    states_.push_back(Node{});
    hist_.clear();
    livePosStack_.clear();
    last_ = 0;
}

void EventAwareOnlineSAM::boundary() {
    last_ = 0;
    livePosStack_.clear();
    hist_.push_back(-1);
}

void EventAwareOnlineSAM::tombstoneBackspace(int nToks) {
    int count = std::max(1, nToks);
    for (int i = 0; i < count; ++i) {
        if (!livePosStack_.empty()) {
            int32_t pos = livePosStack_.back();
            livePosStack_.pop_back();
            if (pos >= 0 && static_cast<size_t>(pos) < hist_.size()) {
                hist_[pos] = -1;
            }
        }
    }
    // Trim trailing tombstones so contiguous typing resumes seamlessly from the surviving prefix!
    while (!hist_.empty() && hist_.back() == -1 && !livePosStack_.empty() &&
           static_cast<size_t>(livePosStack_.back() + 1) < hist_.size()) {
        hist_.pop_back();
    }
    if (!hist_.empty() && !livePosStack_.empty() &&
        static_cast<size_t>(livePosStack_.back() + 1) == hist_.size()) {
        // Re-anchor last_ along surviving recent tokens (up to 32 tokens)
        size_t startIdx =
            livePosStack_.size() > 32 ? livePosStack_.size() - 32 : 0;
        int32_t st = 0;
        for (size_t i = startIdx; i < livePosStack_.size(); ++i) {
            int32_t p = livePosStack_[i];
            if (p < 0 || static_cast<size_t>(p) >= hist_.size() ||
                hist_[p] < BYTE_OFFSET) {
                st = 0;
                continue;
            }
            int32_t ch = hist_[p];
            while (st > 0 && states_[st].findNext(ch) == -1) {
                st = states_[st].link;
            }
            int32_t nxt = states_[st].findNext(ch);
            st = (nxt != -1) ? nxt : 0;
        }
        last_ = st;
    } else {
        last_ = 0;
        hist_.push_back(-1);
    }
}

SamMatch EventAwareOnlineSAM::pushCharTok(int32_t ch) {
    if (ch < BYTE_OFFSET) {
        return {};
    }
    if (hist_.size() > maxTokens_) {
        size_t keepFrom = hist_.size() - (maxTokens_ / 2);
        std::vector<int32_t> kept;
        kept.reserve(hist_.size() - keepFrom);
        for (size_t i = keepFrom; i < hist_.size(); ++i) {
            if (hist_[i] >= BYTE_OFFSET || hist_[i] == -1) {
                kept.push_back(hist_[i]);
            }
        }
        resetAll();
        for (int32_t t : kept) {
            if (t == -1) {
                boundary();
            } else {
                pushCharTok(t);
            }
        }
    }

    int32_t pos = static_cast<int32_t>(hist_.size());
    if (last_ > 0 && pos > 0 && hist_[pos - 1] >= BYTE_OFFSET) {
        int32_t u = last_;
        int32_t prevP = pos - 1;
        while (u > 0 && states_[u].endpos < prevP) {
            int32_t oldEp = states_[u].endpos;
            if (oldEp >= 0 && static_cast<size_t>(oldEp + 1) < hist_.size() &&
                hist_[oldEp + 1] >= BYTE_OFFSET) {
                states_[u].prevEndpos = oldEp;
            }
            states_[u].endpos = prevP;
            u = states_[u].link;
        }
    }

    hist_.push_back(ch);
    livePosStack_.push_back(pos);
    int32_t cur = static_cast<int32_t>(states_.size());
    Node curNode;
    curNode.length = states_[last_].length + 1;
    states_.push_back(std::move(curNode));

    int32_t p = last_;
    while (p != -1 && states_[p].findNext(ch) == -1) {
        states_[p].setNext(ch, cur);
        p = states_[p].link;
    }
    if (p == -1) {
        states_[cur].link = 0;
    } else {
        int32_t q = states_[p].findNext(ch);
        if (states_[p].length + 1 == states_[q].length) {
            states_[cur].link = q;
        } else {
            int32_t clone = static_cast<int32_t>(states_.size());
            Node cloneNode = states_[q];
            cloneNode.length = states_[p].length + 1;
            states_.push_back(std::move(cloneNode));
            while (p != -1 && states_[p].findNext(ch) == q) {
                states_[p].setNext(ch, clone);
                p = states_[p].link;
            }
            states_[q].link = clone;
            states_[cur].link = clone;
        }
    }

    last_ = cur;
    return queryCurrent();
}

SamMatch EventAwareOnlineSAM::queryCurrent() const {
    return queryFrom(last_, states_[last_].length);
}

SamMatch EventAwareOnlineSAM::queryFrom(int32_t stateIdx,
                                        int32_t curLen) const {
    int32_t v = stateIdx;
    while (v > 0 && static_cast<size_t>(v) < states_.size()) {
        for (int32_t ep : {states_[v].endpos, states_[v].prevEndpos}) {
            if (ep >= 0 && static_cast<size_t>(ep + 1) < hist_.size()) {
                int32_t cand = hist_[ep + 1];
                if (cand >= BYTE_OFFSET) {
                    return {cand, std::min(curLen, states_[v].length)};
                }
            }
        }
        v = states_[v].link;
    }
    return {-1, 0};
}

SamSpecState EventAwareOnlineSAM::stepSpeculative(int32_t stateIdx,
                                                  int32_t curLen,
                                                  int32_t ch) const {
    if (ch < BYTE_OFFSET) {
        return {stateIdx, curLen, queryFrom(stateIdx, curLen)};
    }
    while (stateIdx > 0 && states_[stateIdx].findNext(ch) == -1) {
        stateIdx = states_[stateIdx].link;
        curLen = states_[stateIdx].length;
    }
    int32_t nxtState = states_[stateIdx].findNext(ch);
    if (nxtState != -1) {
        stateIdx = nxtState;
        curLen += 1;
    } else {
        stateIdx = 0;
        curLen = 0;
    }
    return {stateIdx, curLen, queryFrom(stateIdx, curLen)};
}

size_t EventAwareOnlineSAM::warmupFromLog(const std::filesystem::path &logPath,
                                          const PureUnicodeCodec &codec,
                                          size_t maxChars) {
    std::vector<std::filesystem::path> files;
    std::filesystem::path rot1 = logPath.string() + ".1";
    std::error_code ec;
    if (std::filesystem::is_regular_file(rot1, ec)) {
        files.push_back(rot1);
    }
    if (std::filesystem::is_regular_file(logPath, ec)) {
        files.push_back(logPath);
    }
    if (files.empty()) {
        return 0;
    }

    std::vector<std::string> segments;
    std::string curBuf;
    std::string lastLegacyPrefix;

    for (const auto &p : files) {
        std::ifstream ifs(p);
        if (!ifs) {
            continue;
        }
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        size_t startIdx = lines.size() > 12000 ? lines.size() - 12000 : 0;
        for (size_t i = startIdx; i < lines.size(); ++i) {
            const std::string &ln = lines[i];
            std::string et;
            extractJsonStringField(ln, "event", et);
            bool hasSeq = (ln.find("\"seq\":") != std::string::npos);

            if (!hasSeq) {
                if (et == "suggest_start") {
                    std::string ptRaw;
                    if (extractJsonStringField(ln, "prefix_tail", ptRaw)) {
                        std::string pt = stripInvisibleFormattingUtf8(ptRaw);
                        if (!pt.empty()) {
                            if (!lastLegacyPrefix.empty() &&
                                !pt.starts_with(lastLegacyPrefix) &&
                                utf8CharCount(lastLegacyPrefix) >= 2) {
                                segments.push_back(lastLegacyPrefix);
                            }
                            lastLegacyPrefix = std::move(pt);
                        }
                    }
                }
                continue;
            }

            if (!lastLegacyPrefix.empty()) {
                if (utf8CharCount(lastLegacyPrefix) >= 2) {
                    segments.push_back(lastLegacyPrefix);
                }
                lastLegacyPrefix.clear();
            }

            if (et == "focus_in" || et == "cursor_jump") {
                if (utf8CharCount(curBuf) >= 2) {
                    segments.push_back(curBuf);
                }
                curBuf.clear();
                std::string ptRaw;
                if (extractJsonStringField(ln, "prefix_tail", ptRaw)) {
                    std::string pt = stripInvisibleFormattingUtf8(ptRaw);
                    if (utf8CharCount(pt) >= 2) {
                        segments.push_back(std::move(pt));
                    }
                }
            } else if (et == "user_type") {
                std::string txtRaw;
                if (extractJsonStringField(ln, "text", txtRaw)) {
                    std::string txt = stripInvisibleFormattingUtf8(txtRaw);
                    if (txt == "\n") {
                        if (utf8CharCount(curBuf) >= 2) {
                            segments.push_back(curBuf);
                        }
                        curBuf.clear();
                    } else {
                        curBuf += txt;
                    }
                }
            } else if (et == "accept_chunk" || et == "accept_word" ||
                       et == "accept_all") {
                std::string accRaw;
                if (extractJsonStringField(ln, "accepted", accRaw)) {
                    curBuf += stripInvisibleFormattingUtf8(accRaw);
                }
            } else if (et == "backspace") {
                popLastUtf8Char(curBuf);
            }
        }
    }

    if (utf8CharCount(lastLegacyPrefix) >= 2) {
        segments.push_back(lastLegacyPrefix);
    }
    if (utf8CharCount(curBuf) >= 2) {
        segments.push_back(curBuf);
    }

    std::vector<std::string> deduped;
    size_t totalChars = 0;
    for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
        std::string s = trimWhitespace(*it);
        size_t cLen = utf8CharCount(s);
        if (cLen < 2) {
            continue;
        }
        if (!deduped.empty() &&
            (deduped.back() == s || deduped.back().starts_with(s))) {
            continue;
        }
        deduped.push_back(std::move(s));
        totalChars += cLen;
        if (totalChars >= maxChars) {
            break;
        }
    }
    std::reverse(deduped.begin(), deduped.end());

    size_t fedChars = 0;
    for (const auto &s : deduped) {
        boundary();
        for (int32_t t : codec.encode(s)) {
            if (t >= BYTE_OFFSET) {
                pushCharTok(t);
            }
        }
        fedChars += utf8CharCount(s);
    }
    boundary();
    return fedChars;
}

} // namespace fcitx5_llm
