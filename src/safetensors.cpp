#include "tfi/safetensors.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "tfi/json.h"
#include "half.h"

namespace tfi {

namespace {

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

SafeTensors load_safetensors(const std::string& path) { return load_safetensors(path, nullptr); }

SafeTensors load_safetensors(const std::string& path, const TensorSink& take) {
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
                values[i] = dtype == "F16" ? detail::half_to_float(h) : bf16_to_f32(h);
            }
        } else {
            throw std::runtime_error(path + ": unsupported dtype " + dtype + " for " + name);
        }
        raw = std::vector<char>();
        Tensor t(shape, std::move(values));
        if (take) take(name, std::move(t));
        else out.tensors[name] = std::move(t);
    }
    return out;
}

}  // namespace tfi
