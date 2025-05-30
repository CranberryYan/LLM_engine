#ifndef KERNELLAB_INCLUDE_API_TRACE_H_
#define KERNELLAB_INCLUDE_API_TRACE_H_
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <unistd.h>

#include "base.h"
#include "para.h"
#include "../tensor/tensor.h"

namespace api_trace {
class API_trace {
public:
  API_trace(std::string func_);
  ~API_trace();

  void set_tensor(std::string name, tensor::Tensor tensor);

  const char* DeviceTypeToStr(base::DeviceType type);

  const char* DataTypeToStr(base::DataType type);

  void print_tensor(std::string name, tensor::Tensor ten);

  void print_tensor();

  void print_scatter_type(para::ScatterOpType op_type);

  void print_softmax_type(para::SoftmaxOpType op_type);
private:
  std::string func_;
  std::map<std::string, tensor::Tensor> tensor_map;
};
} // namespace api_trace
#endif // KERNELLAB_INCLUDE_API_TRACE_H_
