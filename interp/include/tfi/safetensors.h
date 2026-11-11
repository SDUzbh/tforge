// Reader for safetensors files (F32, F16, BF16 converted to float32; I64 kept as integers).
#pragma once

#include <map>
#include <string>
#include <vector>

#include "tfi/tensor.h"

namespace tfi {

struct IntTensor {
    Shape shape;
    std::vector<int64_t> data;
};

struct SafeTensors {
    std::map<std::string, Tensor> tensors;
    std::map<std::string, IntTensor> int_tensors;
    std::map<std::string, std::string> metadata;
};

SafeTensors load_safetensors(const std::string& path);

}  // namespace tfi
