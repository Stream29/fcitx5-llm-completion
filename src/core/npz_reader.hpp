#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace fcitx5_llm {

struct NpyArray {
    std::string descr;
    std::vector<size_t> shape;
    std::vector<uint8_t> rawBytes;

    size_t numElements() const;
    std::vector<float> asFloat32() const;
    std::vector<uint64_t> asUint64() const;
    std::vector<uint16_t> asUint16() const;
    std::vector<uint8_t> asUint8() const;
};

using NpzArchive = std::unordered_map<std::string, NpyArray>;

float fp16ToFp32(uint16_t h);
NpzArchive loadNpzArchive(const std::filesystem::path &npzPath);

} // namespace fcitx5_llm
