#ifndef ENGINE_INCLUDE_OP_EMBEDDING_H_
#define ENGINE_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"

namespace op
{
struct EmbeddingOutput
{
  explicit EmbeddingOutput(tensor::Tensor input_tokens,
                           tensor::Tensor input_embeddings,
                           tensor::Tensor input_token_num) :
    input_tokens(std::move(input_tokens)),
    input_embeddings(std::move(input_embeddings)),
    input_token_num(std::move(input_token_num)) { }

  tensor::Tensor input_tokens;
  tensor::Tensor input_embeddings;
  tensor::Tensor input_token_num;
};

class EmbeddingLayer : public LayerParam {
public:
  explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim,
                          int32_t seq_len, int32_t vocab_size);
  
  base::Status checkArgs() const override;

  base::Status compute() override;

private:
  int32_t dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t vocab_size_ = 0;
};
} // namespace op
#endif // ENGINE_INCLUDE_OP_EMBEDDING_H_
