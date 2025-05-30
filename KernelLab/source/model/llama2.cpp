#include <utility>
#include <glog/logging.h>
#include <cuda_runtime_api.h>
#include <sentencepiece_processor.h>
#include "base/tick.h"
#include "model/llama2.h"
#include "op/mha.h"
#include "op/rope.h"
#include "op/matmul.h"
#include "op/rmsnorm.h"
#include "../op/kernels/cpu/rope_kernel.h"
#include "../op/kernels/cuda/rope_kernel.cuh"

namespace model {
  void Llama2Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
  if (add_layer_) {
    add_layer_->set_cuda_config(config);
    add_layer_->to_cuda();
  }

  if (rope_layer_) {
    rope_layer_->set_cuda_config(config);
    rope_layer_->to_cuda();
  }

  if (swiglu_layer_) {
    swiglu_layer_->set_cuda_config(config);
    swiglu_layer_->to_cuda();
  }

  if (cls_layer_) {
    cls_layer_->set_cuda_config(config);
    cls_layer_->to_cuda();
  }

  if (embedding_layer_) {
    embedding_layer_->set_cuda_config(config);
    embedding_layer_->to_cuda();
  }

  if (mha_layer_) {
    mha_layer_->set_cuda_config(config);
    mha_layer_->to_cuda();
  }

  for (auto& weight_layer : wq_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wk_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wv_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : wo_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w1_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w2_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& weight_layer : w3_layers_) {
    if (weight_layer) {
      weight_layer->set_cuda_config(config);
      weight_layer->to_cuda();
    }
  }

  for (auto& rms_norm_layer : rmsnorm_layers_) {
    if (rms_norm_layer) {
      rms_norm_layer->to_cuda();
      rms_norm_layer->set_cuda_config(config);
    }
  }
}

Llama2Model::Llama2Model(base::TokenizerType tokenizer_type,
                         std::string token_path,
                         std::string model_path,
                         bool is_quant_model) :
  Model(tokenizer_type,
  base::ModelType::kModelTypeLLama2,
  std::move(token_path),
  std::move(model_path),
  is_quant_model) { }

base::Status Llama2Model::init(base::DeviceType device_type) {
  device_type_ = device_type;

  if (token_path_.empty()) {
    return base::error::PathNotValid(token_path_);
  }
  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError(
      "The cpu device do not support int8 quant model.");
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    cudaSetDevice(0);
    cuda_config_ = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config_->stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return base::error::InternalError("The cuda handle create failed.");
    }
  }

  base::Status read_status = gen_model_from_file();
  if (!read_status) {
    return read_status;
  }

  init_mem();
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    kernel::sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
              get_buffer(ModelBufferType::kSinCache).ptr<float>(),
              get_buffer(ModelBufferType::kCosCache).ptr<float>());
  } else {
    CHECK_NE(cuda_config_, nullptr);
    kernel::sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache),
                                  cuda_config_->stream);
  }

  sampler_ = std::make_unique<sampler::ArgMaxSampler>(device_type_);

  return base::error::Success();
}

base::Status Llama2Model::predict(const tensor::Tensor& input,
                                  const tensor::Tensor& pos_tensor,
                                  bool is_prompt, int& next) const {
  base::Status status = forward(input, pos_tensor, next);
  if (!status) {
    return status;
  }

  next = post_processing(pos_tensor, is_prompt);

  return base::error::Success();
}

base::Status Llama2Model::forward(const tensor::Tensor& input,
                                  const tensor::Tensor& pos_tensor,
                                  int& next) const {
  if (input.is_empty()) {
    return base::error::InvalidArgument("The input tensor is empty.");
  }

  if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
    return base::error::InternalError("Unsupported int8 quant in the cpu device");
  }

  for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
    attention_rms(layer_idx, input);
    attention_qkv(layer_idx, pos_tensor);
    attention_mha(layer_idx, pos_tensor);
    feed_forward(layer_idx, input);
  }
  cls_logits(input);

  return base::error::Success();
}

op::EmbeddingOutput Llama2Model::embedding(
  std::vector<int>& tokens) {
  tensor::Tensor &input_tokens =
    get_buffer(ModelBufferType::kInputTokens);
  tensor::Tensor &input_embeddings =
    get_buffer(ModelBufferType::kInputEmbeddings);

  if (input_tokens.size() != tokens.size()) {
    input_tokens.reshape({static_cast<int32_t>(tokens.size())});
    input_embeddings.reshape(
      {static_cast<int32_t>(tokens.size()), config_->dim_});
  }

  for (int32_t i = 0; i < tokens.size(); ++i) {
    input_tokens.index<int32_t>(i) = tokens.at(i);
  }

  tensor::Tensor input_token_num =
    tensor::Tensor(base::DataType::kDataTypeFp32,
                   static_cast<int32_t>(tokens.size()));
  LOG_IF(FATAL, !llama_layers_->embedding_layer_)
      << "The embedding layer in the llama2 model is null pointer.";
  STATUS_CHECK(llama_layers_->embedding_layer_->forward(
    input_tokens, input_token_num, input_embeddings));

  op::EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);

  return output;
}

void Llama2Model::init_mem() {
  std::shared_ptr<base::DeviceAllocator> alloc;
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    alloc = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    alloc = base::CUDADeviceAllocatorFactory::get_instance();
  }

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK_NE(cuda_config_, nullptr);
    llama_layers_->to_cuda(cuda_config_);
  }

  std::shared_ptr<base::DeviceAllocator> alloc_cpu =
    base::CPUDeviceAllocatorFactory::get_instance();
  std::shared_ptr<base::DeviceAllocator> alloc_cu =
    base::CUDADeviceAllocatorFactory::get_instance();

  // 初始化tensor(无数据)并insert到buffer的对应位置(key)中
  tensor::Tensor input_tokens(
    base::DataType::kDataTypeInt32, 1, true, alloc_cpu, nullptr);
  tensor::Tensor input_embeddings(
    base::DataType::kDataTypeFp32, 1, config_->dim_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
  CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

  tensor::Tensor sin_cache(
    base::DataType::kDataTypeFp32,
    config_->head_size_ * config_->seq_len_, true, alloc, nullptr);
  tensor::Tensor cos_cache(
    base::DataType::kDataTypeFp32,
    config_->head_size_ * config_->seq_len_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
  CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

  tensor::Tensor rms_output(base::DataType::kDataTypeFp32,
    config_->dim_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
  CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
  CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
  CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

  tensor::Tensor w1_output(base::DataType::kDataTypeFp32,
    config_->hidden_dim_, true, alloc, nullptr);
  tensor::Tensor w3_output(base::DataType::kDataTypeFp32,
    config_->hidden_dim_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
  CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

  tensor::Tensor key_cache(base::DataType::kDataTypeFp32,
    config_->layer_num_, config_->seq_len_, config_->kv_dim_,
    true, alloc, nullptr);
  tensor::Tensor val_cache(base::DataType::kDataTypeFp32,
    config_->layer_num_, config_->seq_len_, config_->kv_dim_,
    true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
  CHECK(insert_buffer(ModelBufferType::kValCache, val_cache));

  tensor::Tensor query(base::DataType::kDataTypeFp32,
    config_->dim_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kQuery, query));
  CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

  tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32,
    1, true, alloc_cpu, nullptr);
  CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

  tensor::Tensor attn(base::DataType::kDataTypeFp32,
    config_->head_num_, config_->seq_len_, true, alloc, nullptr);
  CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));

  tensor::Tensor forward_output(base::DataType::kDataTypeFp32,
    std::abs(config_->vocab_size_), true, alloc, nullptr);
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32,
                                      std::abs(config_->vocab_size_), true,
                                      alloc_cpu, nullptr);
    CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU,
      forward_output_cpu));
  }

  CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

// 创建带参Layer
//  Embedding, Linear(MHA(生成多头部分)), RMSNorm, 
void Llama2Model::create_param_layers() {
  CHECK(!is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  // Embedding
  auto cpu_device_type = base::DeviceType::kDeviceCPU;
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
    device_type_,
    config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

  const void* weight_embedding = raw_model_data_->weight(0);
  llama_layers_->embedding_layer_->set_weight(
    0,
    {std::abs(config_->vocab_size_), config_->dim_},
    weight_embedding,
    cpu_device_type);

  // Linear
  int32_t dim = config_->dim_;

  // Embedding: dim * config_->vocab_size_
  // RMSNorm: dim * config_->layer_num_
  size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

  // q
  for (int i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> wq =
      std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wq->set_weight(0, {dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos += dim * dim;
  }

  // k
  for (int i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> wk =
      std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wk->set_weight(0, {config_->kv_dim_, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos += config_->kv_dim_ * dim;
  }
  
  // v
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> wv =
      std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
    wv->set_weight(0, {config_->kv_dim_, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim;
  }

  // out
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> wo =
      std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
    wo->set_weight(0, {dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos += dim * dim;
  }

  // skip RMSNorm
  pos += config_->layer_num_ * dim;

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> w1 =
      std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w1->set_weight(0, {hidden_dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos += dim * hidden_dim;
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> w2 =
      std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
    w2->set_weight(0, {dim, hidden_dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos += dim * hidden_dim;
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::MatmulLayer> w3 =
      std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
    w3->set_weight(0, {hidden_dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos += dim * hidden_dim;
  }

  // skip RMSNorm(最后一次, 不需要进行Layer_num次循环)
  pos += dim;
  // skip freqs_cos and freqs_sin weight
  pos += config_->seq_len_ * config_->head_size_;

  llama_layers_->cls_layer_ =
    std::make_shared<op::MatmulLayer>(device_type_, std::abs(config_->vocab_size_), dim);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    llama_layers_->cls_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim},
                                          this->raw_model_data_->weight(0),
                                          cpu_device_type);
  } else {
    llama_layers_->cls_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim},
                                          this->raw_model_data_->weight(pos),
                                          cpu_device_type);
  }

  // RMSNorm
  size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);
  for (int i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RMSNormLayer> rms_norm_layer =
      std::make_shared<op::RMSNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0,
      {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    rmsnorm_pos += config_->dim_;
  }

  // skip attention.wq attention.wk attention.wv attention.wo
  // q
  rmsnorm_pos +=
    config_->layer_num_ * config_->dim_ * config_->dim_;
  // k and v
  rmsnorm_pos +=
    config_->layer_num_ *
    config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  rmsnorm_pos +=
    config_->layer_num_ *
    config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
  // out
  rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    std::shared_ptr<op::RMSNormLayer> rms_norm_layer =
        std::make_shared<op::RMSNormLayer>(device_type_, config_->dim_);
    const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
    rms_norm_layer->set_weight(0,
      {config_->dim_}, weight_rmsnorm, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

    rmsnorm_pos += config_->dim_;
  }

  // skip ffn.w1 ffn.w2 ffn.w3
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
  rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

  std::shared_ptr<op::RMSNormLayer> rms_final_layer =
    std::make_shared<op::RMSNormLayer>(device_type_, config_->dim_);
  const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
  rms_final_layer->set_weight(0,
    {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
  llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

// 同上, 仅数据类型不同
void Llama2Model::create_param_quant_layers() {
  CHECK(is_quant_model_);
  CHECK(llama_layers_ != nullptr);

  size_t pos = 0;
  int32_t dim = config_->dim_;
  auto cpu_device_type = base::DeviceType::kDeviceCPU;

  // query
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wq->set_group_size(group_size_);
    wq->set_weight(0, {dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wq_layers_.push_back(wq);
    pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
  }

  // key
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wk = std::make_shared<op::MatmulLayer>(
      device_type_, config_->kv_dim_, dim, true);
    wk->set_group_size(group_size_);
    wk->set_weight(0, {config_->kv_dim_, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wk_layers_.push_back(wk);
    pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
  }

  // value
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wv = std::make_shared<op::MatmulLayer>(
      device_type_, config_->kv_dim_, dim, true);
    wv->set_group_size(group_size_);
    wv->set_weight(0, {config_->kv_dim_, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wv_layers_.push_back(wv);
    pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
  }

  // output
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
    wo->set_group_size(group_size_);
    wo->set_weight(0, {dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->wo_layers_.push_back(wo);
    pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
  }

  // w1 layers
  int32_t hidden_dim = config_->hidden_dim_;
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w1 = std::make_shared<op::MatmulLayer>(
      device_type_, hidden_dim, dim, true);
    w1->set_group_size(group_size_);
    w1->set_weight(0, {hidden_dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w1_layers_.push_back(w1);
    pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
  }

  // w2 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w2 = std::make_shared<op::MatmulLayer>(
      device_type_, dim, hidden_dim, true);
    w2->set_group_size(group_size_);
    w2->set_weight(0, {dim, hidden_dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w2_layers_.push_back(w2);
    pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
  }

  // w3 layers
  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    auto w3 = std::make_shared<op::MatmulLayer>(
      device_type_, hidden_dim, dim, true);
    w3->set_group_size(group_size_);
    w3->set_weight(0, {hidden_dim, dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    llama_layers_->w3_layers_.push_back(w3);
    pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
  }

  // wcls layer
  auto cls_layer = std::make_shared<op::MatmulLayer>(
    device_type_, std::abs(config_->vocab_size_), dim, true);
  cls_layer->set_group_size(group_size_);
  if (config_->is_shared_weight_) {
    // using token embedding weight
    cls_layer->set_weight(0, {std::abs(config_->vocab_size_), dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
  } else {
    // no shared
    cls_layer->set_weight(0, {std::abs(config_->vocab_size_), dim},
      this->raw_model_data_->weight(pos), cpu_device_type);
    pos = pos + std::abs(config_->vocab_size_) * dim +
      cls_layer->get_scale_num() * sizeof(float);
  }
  llama_layers_->cls_layer_ = cls_layer;

  // embedding layer
  float* weight_ptr = (float*)raw_model_data_->weight(pos);
  llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
      device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
  llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim},
                                              weight_ptr,
                                              cpu_device_type);
  weight_ptr += std::abs(config_->vocab_size_) * dim;

  // rmsnorm attention attention,ffn,final
  for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
    std::shared_ptr<op::RMSNormLayer> rms_norm_layer =
        std::make_shared<op::RMSNormLayer>(device_type_, dim);

    rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
    weight_ptr += dim;
  }
}

// 创建不带参Layer
//  Add, RoPe, SwiGLU, MHA(qkv计算部分)
void Llama2Model::create_nonparam_layers() {
  CHECK(llama_layers_ != nullptr);
  llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
    device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

  llama_layers_->mha_layer_ = std::make_shared<op::MHA>(
    device_type_, 0, config_->kv_mul_, config_->kv_dim_,
    config_->seq_len_, config_->head_num_, config_->head_size_);

  llama_layers_->add_layer_ =
    std::make_shared<op::VecAddLayer>(device_type_);

  llama_layers_->swiglu_layer_ =
    std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

base::Status Llama2Model::create_layers() {
  if (!llama_layers_) {
    llama_layers_ = std::make_unique<Llama2Layers>();
  }

  if (!is_quant_model_) {
    create_param_layers();
  } else {
    create_param_quant_layers();
  }

  create_nonparam_layers();

  if (!llama_layers_->embedding_layer_) {
    return base::error::InternalError(
      "Create the embedding layer for the llama model failed!");
  }

  if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
    return base::error::InternalError(
      "Create the rmsnorm layers for the llama model failed!");
  }

  if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
      llama_layers_->wk_layers_.size() != config_->layer_num_ ||
      llama_layers_->wv_layers_.size() != config_->layer_num_ ||
      llama_layers_->wo_layers_.size() != config_->layer_num_) {
    return base::error::InternalError(
      "Create the matmul layer in the attention and ffn attention layers for \
        the llama model failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
        !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i)) {
      return base::error::InternalError(
        "Create the matmul layer in the attention and ffn attention layers for \
          the llama model failed.");
    }
  }

  if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
      llama_layers_->w2_layers_.size() != config_->layer_num_ ||
      llama_layers_->w3_layers_.size() != config_->layer_num_) {
    return base::error::InternalError(
      "Create the matmul layer in the feedforward layers \
        for the llama model failed.");
  }

  for (int32_t i = 0; i < config_->layer_num_; ++i) {
    if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
        !llama_layers_->w3_layers_.at(i)) {
      return base::error::InternalError(
        "Create the matmul layer in the feedforward layers \
          for the llama model failed.");
    }
  }

  if (!llama_layers_->rope_layer_) {
    return base::error::InternalError(
      "Create the rope layer for the llama model failed!");
  }

  if (!llama_layers_->add_layer_) {
    return base::error::InternalError(
      "Create the add layer for the llama model failed!");
  }

  if (!llama_layers_->mha_layer_) {
    return base::error::InternalError(
      "Create the mha layer for the llama model failed!");
  }

  if (!llama_layers_->swiglu_layer_) {
    return base::error::InternalError(
      "Create the SwiGLU layer for the llama model failed!");
  }

  return base::error::Success();
}

void Llama2Model::attention_mha(int32_t layer_idx,
                                const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);

  // MHA
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
  tensor::Tensor val_cache = get_buffer(ModelBufferType::kValCache);
  tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
  tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);

  const std::shared_ptr<op::Layer> &mha_layer = llama_layers_->mha_layer_;
  CHECK_NE(mha_layer, nullptr)
    << "The multi head attention layer is null pointer.";
  int pos = pos_tensor.index<int32_t>(0);
  std::dynamic_pointer_cast<op::MHA>(mha_layer)->set_pos(pos);
  std::dynamic_pointer_cast<op::MHA>(mha_layer)->set_layer_idx(layer_idx);
  STATUS_CHECK(mha_layer->forward(query,
                                  score_storage,
                                  key_cache,
                                  val_cache,
                                  mha_output));

  tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
  const std::shared_ptr<op::Layer> &wo_layer =
    llama_layers_->wo_layers_.at(layer_idx);
  CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
  STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void Llama2Model::attention_rms(int32_t layer_idx,
                                const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);

  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  std::shared_ptr<op::Layer> rmsnorm_layer =
    llama_layers_->rmsnorm_layers_.at(layer_idx);
  if (!rmsnorm_layer) {
    LOG(FATAL)
      << "The attention rmsnorm layer is a null pointer in the llama2 model";
  }
  STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void Llama2Model::attention_qkv(int32_t layer_idx,
                                const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);

  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  const auto &[key, val] =
    slice_kv_cache(layer_idx, pos);

  // query
  const std::shared_ptr<op::Layer> &query_layer =
    llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr)
    << "The query layer in the attention block is null pointer.";

  tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  const std::shared_ptr<op::Layer> &key_layer =
    llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr)
    << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

  // value
  const std::shared_ptr<op::Layer> &value_layer =
    llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr)
    << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(query, key, pos_tensor,
    get_buffer(ModelBufferType::kSinCache),
    get_buffer(ModelBufferType::kCosCache),
    tensor::Tensor{}));
}

void Llama2Model::feed_forward(int32_t layer_idx,
                               const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  CHECK_NE(llama_layers_->add_layer_, nullptr)
    << "The add layer in the feedforward block is null pointer";

  // res_add
  STATUS_CHECK(
    llama_layers_->add_layer_->forward(input,
                                       get_buffer(ModelBufferType::kAttnOutput),
                                       input));

  // ffn rmsnorm
  tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
  const std::shared_ptr<op::Layer> &ffn_rmsnorm =
    llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
  CHECK_NE(ffn_rmsnorm, nullptr)
    << "The final rmsnorm layer in the feedforward block is null pointer";
  STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

  // w1
  tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
  const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
  CHECK_NE(w1_layer, nullptr)
    << "The w1 layer in the feedforward block is null pointer";
  STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

  // w3
  tensor::Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
  const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
  CHECK_NE(w3_layer, nullptr)
    << "The w3 layer in the feedforward block is null pointer";
  STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

  // SwiGLU
  CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
    << "The swiglu layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->swiglu_layer_->forward(
    w1_output, w3_ouput, w1_output));

  // w2
  tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
  const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
  CHECK_NE(w2_layer, nullptr)
    << "The w2 layer in the feedforward block is null pointer";
  STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

  // residual add
  CHECK_NE(llama_layers_->add_layer_, nullptr)
    << "The add layer in the feedforward block is null pointer";
  STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void Llama2Model::cls_logits(const tensor::Tensor& input) const {
  CHECK(llama_layers_ != nullptr);
  const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
  CHECK_NE(norm, nullptr);
  STATUS_CHECK(norm->forward(input, input));

  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  CHECK_NE(llama_layers_->cls_layer_, nullptr);
  STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t Llama2Model::post_processing(const tensor::Tensor& pos,
                                     bool is_prompt) const {
  tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
  const float* forward_logits = forward_output.ptr<float>();

  int32_t next = 0;
  if (is_prompt) {
    next = -1;
  } else {
    next = static_cast<int32_t>(sampler_->sample(
      forward_logits, forward_output.size(),
      cuda_config_ ? cuda_config_->stream : nullptr));
  }

  return next;
}
} // namespace model
