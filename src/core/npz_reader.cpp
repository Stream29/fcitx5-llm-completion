#include "npz_reader.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <zlib.h>

namespace fcitx5_llm {

namespace {

uint16_t readLe16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readLe32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t readLe64(const uint8_t *p) {
    return static_cast<uint64_t>(readLe32(p)) |
           (static_cast<uint64_t>(readLe32(p + 4)) << 32);
}

std::vector<uint8_t> inflateRawDeflate(const uint8_t *compData, size_t compSize,
                                       size_t uncompSize) {
    std::vector<uint8_t> out(uncompSize);
    if (uncompSize == 0) {
        return out;
    }
    z_stream strm{};
    strm.next_in = const_cast<Bytef *>(compData);
    strm.avail_in = static_cast<uInt>(compSize);
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(uncompSize);

    if (inflateInit2(&strm, -MAX_WBITS) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed while reading .npz");
    }
    int ret = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);
    if (ret != Z_STREAM_END) {
        throw std::runtime_error("inflate failed while reading .npz");
    }
    return out;
}

NpyArray parseNpyBuffer(const std::vector<uint8_t> &buf) {
    if (buf.size() < 10 || std::memcmp(buf.data(), "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("Invalid .npy header magic");
    }
    uint8_t major = buf[6];
    size_t hdrLen = 0;
    size_t hdrStart = 0;
    if (major == 1) {
        hdrLen = readLe16(buf.data() + 8);
        hdrStart = 10;
    } else if (major >= 2) {
        if (buf.size() < 12) {
            throw std::runtime_error("Truncated .npy v2 header");
        }
        hdrLen = readLe32(buf.data() + 8);
        hdrStart = 12;
    } else {
        throw std::runtime_error("Unsupported .npy version");
    }
    if (hdrStart + hdrLen > buf.size()) {
        throw std::runtime_error("Truncated .npy header dictionary");
    }

    std::string header(reinterpret_cast<const char *>(buf.data() + hdrStart),
                       hdrLen);
    NpyArray arr;

    // Extract 'descr': '...'
    auto dPos = header.find("'descr'");
    if (dPos == std::string::npos) {
        dPos = header.find("\"descr\"");
    }
    if (dPos != std::string::npos) {
        auto colon = header.find(':', dPos);
        auto q1 = header.find_first_of("'\"", colon);
        auto q2 = header.find(header[q1], q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            arr.descr = header.substr(q1 + 1, q2 - q1 - 1);
        }
    }

    // Check fortran_order: False
    auto fPos = header.find("fortran_order");
    if (fPos != std::string::npos) {
        auto tPos = header.find("True", fPos);
        auto cPos = header.find(',', fPos);
        if (tPos != std::string::npos &&
            (cPos == std::string::npos || tPos < cPos)) {
            throw std::runtime_error("Fortran-order .npy arrays unsupported");
        }
    }

    // Extract 'shape': (d0, d1, ...)
    auto sPos = header.find("'shape'");
    if (sPos == std::string::npos) {
        sPos = header.find("\"shape\"");
    }
    if (sPos != std::string::npos) {
        auto p1 = header.find('(', sPos);
        auto p2 = header.find(')', p1);
        if (p1 != std::string::npos && p2 != std::string::npos) {
            std::string tup = header.substr(p1 + 1, p2 - p1 - 1);
            size_t idx = 0;
            while (idx < tup.size()) {
                while (idx < tup.size() &&
                       (tup[idx] < '0' || tup[idx] > '9')) {
                    ++idx;
                }
                if (idx >= tup.size()) {
                    break;
                }
                size_t val = 0;
                while (idx < tup.size() && tup[idx] >= '0' && tup[idx] <= '9') {
                    val = val * 10 + static_cast<size_t>(tup[idx] - '0');
                    ++idx;
                }
                arr.shape.push_back(val);
            }
        }
    }

    size_t dataOffset = hdrStart + hdrLen;
    arr.rawBytes.assign(buf.begin() + static_cast<ptrdiff_t>(dataOffset),
                        buf.end());
    return arr;
}

} // namespace

float fp16ToFp32(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return std::bit_cast<float>(bits);
}

size_t NpyArray::numElements() const {
    if (shape.empty()) {
        return 1;
    }
    size_t n = 1;
    for (size_t d : shape) {
        n *= d;
    }
    return n;
}

std::vector<float> NpyArray::asFloat32() const {
    size_t n = numElements();
    std::vector<float> out(n);
    if (descr == "<f4" || descr == "f4") {
        if (rawBytes.size() < n * 4) {
            throw std::runtime_error("Truncated <f4 array in .npz");
        }
        std::memcpy(out.data(), rawBytes.data(), n * 4);
    } else if (descr == "<f2" || descr == "f2") {
        if (rawBytes.size() < n * 2) {
            throw std::runtime_error("Truncated <f2 array in .npz");
        }
        for (size_t i = 0; i < n; ++i) {
            out[i] = fp16ToFp32(readLe16(rawBytes.data() + i * 2));
        }
    } else {
        throw std::runtime_error("Unsupported float descr in .npz: " + descr);
    }
    return out;
}

std::vector<uint64_t> NpyArray::asUint64() const {
    size_t n = numElements();
    if ((descr != "<u8" && descr != "u8") || rawBytes.size() < n * 8) {
        throw std::runtime_error("Invalid <u8 array in .npz: " + descr);
    }
    std::vector<uint64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = readLe64(rawBytes.data() + i * 8);
    }
    return out;
}

std::vector<uint16_t> NpyArray::asUint16() const {
    size_t n = numElements();
    if ((descr != "<u2" && descr != "u2") || rawBytes.size() < n * 2) {
        throw std::runtime_error("Invalid <u2 array in .npz: " + descr);
    }
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = readLe16(rawBytes.data() + i * 2);
    }
    return out;
}

std::vector<uint8_t> NpyArray::asUint8() const {
    size_t n = numElements();
    if (rawBytes.size() < n) {
        throw std::runtime_error("Truncated |u1 array in .npz");
    }
    return std::vector<uint8_t>(rawBytes.begin(),
                                rawBytes.begin() + static_cast<ptrdiff_t>(n));
}

NpzArchive loadNpzArchive(const std::filesystem::path &npzPath) {
    std::ifstream ifs(npzPath, std::ios::binary | std::ios::ate);
    if (!ifs) {
        throw std::runtime_error("Cannot open .npz file: " + npzPath.string());
    }
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> fileBuf(static_cast<size_t>(sz));
    if (sz > 0 && !ifs.read(reinterpret_cast<char *>(fileBuf.data()), sz)) {
        throw std::runtime_error("Failed reading .npz file: " +
                                 npzPath.string());
    }

    // Locate ZIP End of Central Directory (EOCD: 0x06054b50) from end of file
    if (fileBuf.size() < 22) {
        throw std::runtime_error("Invalid .npz ZIP file (too small)");
    }
    size_t eocdPos = std::string::npos;
    size_t minSearch = fileBuf.size() > 65557 ? fileBuf.size() - 65557 : 0;
    for (size_t p = fileBuf.size() - 22;; --p) {
        if (readLe32(fileBuf.data() + p) == 0x06054b50u) {
            eocdPos = p;
            break;
        }
        if (p == minSearch) {
            break;
        }
    }
    if (eocdPos == std::string::npos) {
        throw std::runtime_error("ZIP EOCD not found in .npz");
    }

    uint16_t totalEntries = readLe16(fileBuf.data() + eocdPos + 10);
    uint64_t cdOffset = readLe32(fileBuf.data() + eocdPos + 16);

    // Handle ZIP64 EOCD Locator (0x07064b50) if present right before EOCD
    if (eocdPos >= 20 && readLe32(fileBuf.data() + eocdPos - 20) == 0x07064b50u) {
        uint64_t z64Off = readLe64(fileBuf.data() + eocdPos - 12);
        if (z64Off + 56 <= fileBuf.size() &&
            readLe32(fileBuf.data() + z64Off) == 0x06064b50u) {
            totalEntries = static_cast<uint16_t>(
                readLe64(fileBuf.data() + z64Off + 32));
            cdOffset = readLe64(fileBuf.data() + z64Off + 48);
        }
    }

    NpzArchive archive;
    size_t pos = static_cast<size_t>(cdOffset);
    for (uint16_t ei = 0; ei < totalEntries && pos + 46 <= fileBuf.size();
         ++ei) {
        if (readLe32(fileBuf.data() + pos) != 0x02014b50u) {
            break;
        }
        uint16_t method = readLe16(fileBuf.data() + pos + 10);
        uint64_t compSz = readLe32(fileBuf.data() + pos + 20);
        uint64_t uncompSz = readLe32(fileBuf.data() + pos + 24);
        uint16_t nameLen = readLe16(fileBuf.data() + pos + 28);
        uint16_t extraLen = readLe16(fileBuf.data() + pos + 30);
        uint16_t commentLen = readLe16(fileBuf.data() + pos + 32);
        uint64_t lfhOffset = readLe32(fileBuf.data() + pos + 42);

        std::string name(
            reinterpret_cast<const char *>(fileBuf.data() + pos + 46), nameLen);

        // Parse ZIP64 extra field if sizes/offsets are 0xFFFFFFFF
        size_t exPos = pos + 46 + nameLen;
        size_t exEnd = exPos + extraLen;
        while (exPos + 4 <= exEnd && exEnd <= fileBuf.size()) {
            uint16_t tag = readLe16(fileBuf.data() + exPos);
            uint16_t szTag = readLe16(fileBuf.data() + exPos + 2);
            size_t cur = exPos + 4;
            if (tag == 0x0001u) {
                if (uncompSz == 0xFFFFFFFFu && cur + 8 <= exPos + 4 + szTag) {
                    uncompSz = readLe64(fileBuf.data() + cur);
                    cur += 8;
                }
                if (compSz == 0xFFFFFFFFu && cur + 8 <= exPos + 4 + szTag) {
                    compSz = readLe64(fileBuf.data() + cur);
                    cur += 8;
                }
                if (lfhOffset == 0xFFFFFFFFu && cur + 8 <= exPos + 4 + szTag) {
                    lfhOffset = readLe64(fileBuf.data() + cur);
                }
            }
            exPos += 4 + szTag;
        }
        pos += 46 + nameLen + extraLen + commentLen;

        if (lfhOffset + 30 > fileBuf.size() ||
            readLe32(fileBuf.data() + lfhOffset) != 0x04034b50u) {
            throw std::runtime_error("Invalid ZIP local file header in .npz");
        }
        uint16_t lfhNameLen = readLe16(fileBuf.data() + lfhOffset + 26);
        uint16_t lfhExtraLen = readLe16(fileBuf.data() + lfhOffset + 28);
        size_t dataStart =
            static_cast<size_t>(lfhOffset) + 30 + lfhNameLen + lfhExtraLen;
        if (dataStart + compSz > fileBuf.size()) {
            throw std::runtime_error("Truncated ZIP entry payload in .npz");
        }

        std::vector<uint8_t> npyBytes;
        if (method == 0) {
            npyBytes.assign(
                fileBuf.begin() + static_cast<ptrdiff_t>(dataStart),
                fileBuf.begin() + static_cast<ptrdiff_t>(dataStart + compSz));
        } else if (method == 8) {
            npyBytes = inflateRawDeflate(fileBuf.data() + dataStart,
                                         static_cast<size_t>(compSz),
                                         static_cast<size_t>(uncompSz));
        } else {
            throw std::runtime_error("Unsupported ZIP compression method in .npz");
        }

        if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") {
            name.erase(name.size() - 4);
        }
        archive.emplace(std::move(name), parseNpyBuffer(npyBytes));
    }
    return archive;
}

} // namespace fcitx5_llm
