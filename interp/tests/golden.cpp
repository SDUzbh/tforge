// Compare the C++ forward pass with a PyTorch reference written by tforge.export.export_reference.
//
//   tfi_golden model.safetensors reference.safetensors [tolerance]
//
// Checks the logits and every cached activation; the tolerance is relative to each
// tensor's largest magnitude. Exit code 0 means everything matched.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "tfi/model.h"
#include "tfi/safetensors.h"

using namespace tfi;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.safetensors reference.safetensors [tolerance]\n", argv[0]);
        return 2;
    }
    const float tol = argc > 3 ? std::strtof(argv[3], nullptr) : 1e-4f;
    try {
        const Model model = Model::load(argv[1]);
        const SafeTensors ref = load_safetensors(argv[2]);
        const IntTensor& tok = ref.int_tensors.at("tokens");
        ActivationCache cache;
        cache["logits"] = model.run_with_cache(tok.data, cache);

        int failures = 0, checked = 0;
        for (const auto& kv : ref.tensors) {
            std::string name = kv.first;
            if (name.rfind("cache.", 0) == 0) name = name.substr(6);
            auto it = cache.find(name);
            if (it == cache.end()) {
                std::printf("MISSING  %s (in reference, not produced by tfi)\n", name.c_str());
                ++failures;
                continue;
            }
            ++checked;
            if (it->second.shape != kv.second.shape) {
                std::printf("SHAPE    %s: tfi %s vs reference %s\n", name.c_str(), shape_str(it->second.shape).c_str(),
                            shape_str(kv.second.shape).c_str());
                ++failures;
                continue;
            }
            const float diff = max_abs_diff(it->second, kv.second);
            const float rel = diff / std::max(1e-6f, max_abs(kv.second));
            if (!(rel <= tol)) {
                std::printf("DIFF     %s: max abs diff %.3g (relative %.3g)\n", name.c_str(), diff, rel);
                ++failures;
            }
        }
        for (const auto& kv : cache) {
            const std::string key = kv.first == "logits" ? "logits" : "cache." + kv.first;
            if (!ref.tensors.count(key)) {
                std::printf("EXTRA    %s (produced by tfi, not in reference)\n", kv.first.c_str());
                ++failures;
            }
        }
        std::printf("%s: %d tensors checked, %d problems\n", failures ? "FAIL" : "OK", checked, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
