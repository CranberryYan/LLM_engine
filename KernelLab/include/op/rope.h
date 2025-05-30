#ifndef KERNELLAB_INCLUDE_OP_ROPE_H_
#define KERNELLAB_INCLUDE_OP_ROPE_H_
#include "layer.h"
namespace op {
class RoPELayer : public Layer {
public:
  explicit RoPELayer(base::DeviceType device_type,
                     int32_t dim,
                     int32_t kv_dim,
                     int32_t head_size);

  base::Status checkArgs() const override;

  base::Status compute() override;

  base::Status forward() override;
private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
};
} // namespace op
#endif // KERNELLAB_INCLUDE_OP_ROPE_H_
