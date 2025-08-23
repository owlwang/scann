// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_INDEXING_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_INDEXING_H_

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/projection/chunking_projection.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

// Asymmetric Hashing 2 索引器模板类，支持多种量化方案的哈希编码与重构
template <typename T>
class Indexer {
 public:
  // 浮点类型别名（float/double等）
  using FloatT = FloatingTypeFor<T>;

  // 构造函数，需指定投影、距离度量、模型
  Indexer(shared_ptr<const ChunkingProjection<T>> projector,
          shared_ptr<const DistanceMeasure> quantization_distance,
          shared_ptr<const Model<T>> model);

  // 哈希编码接口，支持多种输入/输出类型
  Status Hash(const DatapointPtr<T>& input, Datapoint<uint8_t>* hashed) const;
  Status Hash(const DatapointPtr<T>& input, std::string* hashed) const;
  Status Hash(const DatapointPtr<T>& input, MutableSpan<uint8_t> hashed) const;
  Status Hash(ConstSpan<T> input, MutableSpan<uint8_t> hashed) const;

  // 噪声整形参数结构体
  struct NoiseShapingParameter {
    double eta = NAN;        // 噪声整形系数
    double threshold = NAN;  // 阈值
  };

  // 支持噪声整形的哈希编码接口
  Status HashWithNoiseShaping(
      const DatapointPtr<T>& input, Datapoint<uint8_t>* hashed,
      const NoiseShapingParameter& noise_shaping_param) const;
  Status HashWithNoiseShaping(
      const DatapointPtr<T>& input, MutableSpan<uint8_t> hashed,
      const NoiseShapingParameter& noise_shaping_param) const;
  Status HashWithNoiseShaping(
      ConstSpan<T> input, MutableSpan<uint8_t> hashed,
      const NoiseShapingParameter& noise_shaping_param) const;
  Status HashWithNoiseShaping(
      const DatapointPtr<T>& maybe_residual, const DatapointPtr<T>& original,
      Datapoint<uint8_t>* hashed,
      const NoiseShapingParameter& noise_shaping_param) const;
  Status HashWithNoiseShaping(
      const DatapointPtr<T>& maybe_residual, const DatapointPtr<T>& original,
      MutableSpan<uint8_t> hashed,
      const NoiseShapingParameter& noise_shaping_param) const;

      ConstSpan<T> maybe_residual, ConstSpan<T> original,
      MutableSpan<uint8_t> hashed,
      const NoiseShapingParameter& noise_shaping_param) const;

  // 对整个数据集进行哈希编码
  StatusOr<DenseDataset<uint8_t>> HashDataset(
      const TypedDataset<T>& dataset) const;

  // 哈希码重构为近似原始向量
  Status Reconstruct(const DatapointPtr<uint8_t>& input,
                     Datapoint<FloatT>* reconstructed) const;
  Status Reconstruct(absl::string_view input,
                     Datapoint<FloatT>* reconstructed) const;
  Status Reconstruct(ConstSpan<uint8_t> input,
                     MutableSpan<FloatT> reconstructed) const;

  // 计算原始向量与哈希码之间的距离
  StatusOr<FloatT> DistanceBetweenOriginalAndHashed(
      ConstSpan<FloatT> original, ConstSpan<uint8_t> hashed,
      shared_ptr<const DistanceMeasure> distance_override = nullptr) const;

  // 获取哈希空间维度
  DimensionIndex hash_space_dimension() const;
  // 获取原始空间维度
  DimensionIndex original_space_dimension() const;

  // 计算原始向量与哈希重构向量的残差
  Status ComputeResidual(const DatapointPtr<T>& original,
                         const DatapointPtr<uint8_t>& hashed,
                         Datapoint<FloatT>* result) const;

  // 获取模型指针
  shared_ptr<const Model<T>> model() { return model_; }

 private:
  // 投影器（分块投影）
  shared_ptr<const ChunkingProjection<T>> projector_;
  // 量化距离度量
  shared_ptr<const DistanceMeasure> quantization_distance_;
  // 训练模型（中心点、量化方案等）
  shared_ptr<const Model<T>> model_;

  // 展平后的中心点数据，便于高效访问
  std::vector<FloatT> flattend_model_;
  // 每个子空间的大小和维度
  std::vector<std::pair<uint32_t, uint32_t>> subspace_sizes_;
};

// 实例化所有支持类型的 Indexer 模板
SCANN_INSTANTIATE_TYPED_CLASS(extern, Indexer);

// asymmetric_hashing2 命名空间结束
}  // namespace asymmetric_hashing2
// research_scann 命名空间结束
}  // namespace research_scann

#endif
