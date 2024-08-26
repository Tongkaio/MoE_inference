#include <torch/extension.h>

torch::Tensor forward(torch::Tensor inputs, 
                      torch::Tensor gate_outputs,
                      const int k,
                      torch::Tensor m1, torch::Tensor m2,
                      torch::Tensor s1, torch::Tensor s2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}