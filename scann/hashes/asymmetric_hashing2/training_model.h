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

#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_MODEL_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_MODEL_H_

#include <cstdint>
#include <optional>

#include "scann/data_format/dataset.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/proto/projection.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

// AH 模型类，包含中心集合、量化方案、投影等
template <typename T>
class Model {
 public:
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  using FloatT = FloatingTypeFor<T>;

  // 从中心集合构造模型
  static StatusOr<unique_ptr<Model<T>>> FromCenters(
      std::vector<DenseDataset<FloatT>> centers,
      AsymmetricHasherConfig::QuantizationScheme quantization_scheme =
          AsymmetricHasherConfig::PRODUCT);

  // 从序列化 proto 构造模型
  static StatusOr<unique_ptr<Model<T>>> FromProto(
      const CentersForAllSubspaces& proto,
      std::optional<ProjectionConfig> projection_config = std::nullopt);

  // 模型序列化为 proto
  CentersForAllSubspaces ToProto() const;

  // 获取所有中心集合
  ConstSpan<DenseDataset<FloatT>> centers() const { return centers_; }

  // 获取转置后的中心数据（SIMD优化）
  ConstSpan<FloatT> block_transposed_centers() const {
    return block_transposed_centers_;
  }

  // 每块聚类数
  uint32_t num_clusters_per_block() const { return num_clusters_per_block_; }

  // 块数
  size_t num_blocks() const { return centers_.size(); }

  // 量化方案类型
  AsymmetricHasherConfig::QuantizationScheme quantization_scheme() const {
    return quantization_scheme_;
  }

  // 判断中心集合是否相等
  bool CentersEqual(const Model& rhs) const;

  // 获取投影对象
  StatusOr<shared_ptr<const ChunkingProjection<T>>> GetProjection(
      const ProjectionConfig& projection_config) const;

  // 设置投影对象
  void SetProjection(shared_ptr<const ChunkingProjection<T>> projection);

 private:
  // 构造函数（私有，仅供静态工厂方法使用）
  explicit Model(
      std::vector<DenseDataset<FloatT>> centers,
      AsymmetricHasherConfig::QuantizationScheme quantization_scheme);

  std::vector<DenseDataset<FloatT>> centers_ = {};

  std::vector<FloatT> block_transposed_centers_;

  uint32_t num_clusters_per_block_ = numeric_limits<uint32_t>::max();

  AsymmetricHasherConfig::QuantizationScheme quantization_scheme_ =
      AsymmetricHasherConfig::PRODUCT;

  shared_ptr<const ChunkingProjection<T>> projection_ = nullptr;

  ProjectionConfig projection_config_;
};

// Model 模板类显式实例化声明
SCANN_INSTANTIATE_TYPED_CLASS(extern, Model);

// asymmetric_hashing2 命名空间结束
}  // namespace asymmetric_hashing2
// research_scann 命名空间结束
}  // namespace research_scann

#endif
