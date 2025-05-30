#include "op/layer.h"
#include "op/swiglu.h"
#include "./base/API_trace.h"
#include "kernels/kernels_interface.h"

namespace op {
  static const bool apiTraceEnabled = (std::getenv("api_trace") != nullptr);

SwiGLULayer::SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, op::LayerType::kLayerSwiGLU, "SwiGLU"),
      hidden_dim_(hidden_dim) {
  reset_input_size(2);
  reset_output_size(1);
}

base::Status SwiGLULayer::checkArgs() const {
  base::Status status;
  const int32_t input_tensor_num = 2;
  for (int32_t i = 0; i < input_tensor_num; ++i) {
    status = check_tensor_with_dim(get_input(0),
                                   device_type_, data_type_, hidden_dim_);
    if (!status) {
      LOG(ERROR) << "The input tensor " <<
        std::to_string(i) << " error in the swiglu layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0),
                                 device_type_, data_type_, hidden_dim_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the swiglu layer.";
    return status;
  }
  return base::error::Success();
}

base::Status SwiGLULayer::compute() {
  auto input1 = this->get_input(0);
  auto input2 = this->get_input(1);
  auto output = this->get_output(0);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  kernel::get_swiglu_kernel(device_type_)(input1, input2, output,
                                          cuda_config_ ? 
                                          cuda_config_->stream :
                                          nullptr);
  return base::error::Success();
}

base::Status SwiGLULayer::forward() {
  if (apiTraceEnabled) {
    api_trace::API_trace trace("MatmulLayer::forward()");
    trace.set_tensor("input1", this->get_input(0));
    trace.set_tensor("input2", this->get_input(1));
    trace.set_tensor("output", this->get_output(0));

    trace.print_tensor();
  }

  auto status = checkArgs();
  if (!status) {
    return status;
  }

  status = this->compute();
  if (!status) {
    return status;
  }

  return base::error::Success();
}
} // namespace op
