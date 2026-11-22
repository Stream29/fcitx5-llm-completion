#pragma once

#include "unicode_codec.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace fcitx5_llm {

struct SamMatch {
    int32_t nextTok = -1;
    int32_t matchLen = 0;
};

struct SamSpecState {
    int32_t stateIdx = 0;
    int32_t curLen = 0;
    SamMatch match;
};

class EventAwareOnlineSAM {
public:
    explicit EventAwareOnlineSAM(size_t maxTokens = 98304);

    void resetAll();
    void boundary();
    void tombstoneBackspace(int nToks = 1);
    SamMatch pushCharTok(int32_t ch);
    SamMatch queryCurrent() const;
    SamMatch queryFrom(int32_t stateIdx, int32_t curLen) const;
    SamSpecState stepSpeculative(int32_t stateIdx, int32_t curLen,
                                 int32_t ch) const;

    size_t warmupFromLog(const std::filesystem::path &logPath,
                         const PureUnicodeCodec &codec,
                         size_t maxChars = 48000);

    int32_t lastState() const { return last_; }
    int32_t lastLength() const { return states_[last_].length; }

private:
    struct Node {
        std::vector<std::pair<int32_t, int32_t>> trans;
        int32_t link = -1;
        int32_t length = 0;
        int32_t endpos = -1;
        int32_t prevEndpos = -1;

        int32_t findNext(int32_t ch) const {
            for (const auto &kv : trans) {
                if (kv.first == ch) {
                    return kv.second;
                }
            }
            return -1;
        }

        void setNext(int32_t ch, int32_t target) {
            for (auto &kv : trans) {
                if (kv.first == ch) {
                    kv.second = target;
                    return;
                }
            }
            trans.emplace_back(ch, target);
        }
    };

    size_t maxTokens_;
    std::vector<Node> states_;
    std::vector<int32_t> hist_;
    std::vector<int32_t> livePosStack_;
    int32_t last_ = 0;
};

} // namespace fcitx5_llm
