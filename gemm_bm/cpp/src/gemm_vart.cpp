#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vitis/ai/env_config.hpp>
#include <xir/graph/graph.hpp>
#include "vart/dpu/vitis_dpu_runner_factory.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"

static std::unique_ptr<vart::TensorBuffer> create_cpu_flat_tensor_buffer(
    const xir::Tensor* tensor) {
  return std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Wrong parameters !" << std::endl;
        return -1;
    }

    const std::string model_file = argv[1];
    std::string xmodel_name = model_file.substr(model_file.find_last_of("/\\") + 1);
    std::cout << "Xmodel name: " << xmodel_name << std::endl;

    const auto filename = argv[1];
    const auto kernel_name = std::string("subgraph_GEMMModel__GEMMModel_Conv2d_gemm__ret");

    auto runner = vart::dpu::DpuRunnerFactory::create_dpu_runner(filename, kernel_name);
    auto input_tensors = runner->get_input_tensors();
    auto output_tensors = runner->get_output_tensors();

    // Create input and output buffers
    auto input_scale = vart::get_input_scale(input_tensors);
    auto output_scale = vart::get_output_scale(output_tensors);

    CHECK_EQ(input_tensors.size(), 1u) << "only 1 in put tensor is supported";
    CHECK_EQ(output_tensors.size(), 1u) << "only 1 out put tensor is supported";
    auto input_tensor = input_tensors[0];
    auto output_tensor = output_tensors[0];
    auto input_tensor_buffer = create_cpu_flat_tensor_buffer(input_tensor);
    auto output_tensor_buffer = create_cpu_flat_tensor_buffer(output_tensor);

    auto input_shape = input_tensor->get_shape();
    std::cout << "Input tensor shape: ";
    for (const auto& dim : input_shape) {
        std::cout << dim << " ";
    }

    if (input_shape.size() != 4) {
        std::cerr << "Input tensor shape wrong!" << std::endl;
        return -1;
    }

    int first_dim = input_shape[0];
    int second_dim = input_shape[1];
    int third_dim = input_shape[2];
    int forth_dim = input_shape[3];

    std::cout << "=========== Input shape: [" << first_dim << ", " << second_dim << ", " << third_dim << ", " << forth_dim << "] ===========" << std::endl;

    for (int b = 0; b < first_dim; ++b) {
        uint64_t data_in = 0u;
        size_t size_in = 0u;
        std::tie(data_in, size_in) = input_tensor_buffer->data({b, 0, 0, 0});
        signed char* input_data_ptr = reinterpret_cast<signed char*>(data_in);

        for (size_t i = 0; i < size_in; ++i) {
            input_data_ptr[i] = static_cast<signed char>(rand() % 255 - 128);
        }
    }

    // Run the model
    auto v = runner->execute_async({input_tensor_buffer.get()},
                                    {output_tensor_buffer.get()});

    auto status = runner->wait((int)v.first, -1);
    CHECK_EQ(status, 0) << "failed to run dpu";
    printf("run dpu success\n");

    return 0;
}
