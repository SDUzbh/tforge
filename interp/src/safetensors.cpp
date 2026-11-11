#include "tfi/safetensors.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "tfi/json.h"

namespace tfi {

namespace {

float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; --exp; }
            bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

SafeTensors load_safetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint64_t header_len = 0;
    unsigned char len_bytes[8];
    if (!f.read(reinterpret_cast<char*>(len_bytes), 8)) throw std::runtime_error(path + ": file too short");
    for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];
    if (header_len > (uint64_t(1) << 30)) throw std::runtime_error(path + ": implausible header size");
    std::string header(header_len, '\0');
    if (!f.read(&header[0], static_cast<std::streamsize>(header_len))) throw std::runtime_error(path + ": truncated header");
    const json::Value root = json::parse(header);
    const std::streamoff data_start = 8 + static_cast<std::streamoff>(header_len);

    f.seekg(0, std::ios::end);
    const uint64_t data_size = static_cast<uint64_t>(f.tellg()) - static_cast<uint64_t>(data_start);

    SafeTensors out;
    for (size_t k = 0; k < root.keys.size(); ++k) {
        const std::string& name = root.keys[k];
        const json::Value& info = root.values[k];
        if (name == "__metadata__") {
            for (size_t m = 0; m < info.keys.size(); ++m) out.metadata[info.keys[m]] = info.values[m].as_string();
            continue;
        }
        const std::string& dtype = info.at("dtype").as_string();
        Shape shape;
        for (const auto& d : info.at("shape").array) shape.push_back(d.as_int());
        const uint64_t begin = static_cast<uint64_t>(info.at("data_offsets").array.at(0).as_int());
        const uint64_t end = static_cast<uint64_t>(info.at("data_offsets").array.at(1).as_int());
        if (end < begin || end > data_size) throw std::runtime_error(path + ": bad offsets for " + name);
        const int64_t n = shape_numel(shape);

        std::vector<char> raw(end - begin);
        f.seekg(data_start + static_cast<std::streamoff>(begin));
        if (!raw.empty() && !f.read(raw.data(), static_cast<std::streamsize>(raw.size())))
            throw std::runtime_error(path + ": cannot read " + name);

        auto check_bytes = [&](size_t width) {
            if (raw.size() != static_cast<size_t>(n) * width)
                throw std::runtime_error(path + ": " + name + " byte size does not match its shape");
        };
        if (dtype == "I64") {
            check_bytes(8);
            IntTensor t{shape, std::vector<int64_t>(n)};
            std::memcpy(t.data.data(), raw.data(), raw.size());
            out.int_tensors[name] = std::move(t);
            continue;
        }
        std::vector<float> values(n);
        if (dtype == "F32") {
            check_bytes(4);
            std::memcpy(values.data(), raw.data(), raw.size());
        } else if (dtype == "F16" || dtype == "BF16") {
            check_bytes(2);
            for (int64_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, raw.data() + 2 * i, 2);
                values[i] = dtype == "F16" ? f16_to_f32(h) : bf16_to_f32(h);
            }
        } else {
            throw std::runtime_error(path + ": unsupported dtype " + dtype + " for " + name);
        }
        out.tensors[name] = Tensor(shape, std::move(values));
    }
    return out;
}

}  // namespace tfi
