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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_BASE_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_OPTIONS_BASE_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/projection/chunking_projection.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

// AH 训练参数基类，包含量化距离、配置、权重等
class TrainingOptionsBase {
 public:
  explicit TrainingOptionsBase(
      shared_ptr<const DistanceMeasure> quantization_distance)
      : quantization_distance_(std::move(quantization_distance)) {}

  TrainingOptionsBase(const AsymmetricHasherConfig& config,
                      shared_ptr<const DistanceMeasure> quantization_distance)
      : conf_(config),
        quantization_distance_(std::move(quantization_distance)) {}

  const shared_ptr<const DistanceMeasure>& quantization_distance() const {
    return quantization_distance_;
  }

  const AsymmetricHasherConfig& config() const { return conf_; }

  AsymmetricHasherConfig* mutable_config() { return &conf_; }

  ConstSpan<float> weights() const { return weights_; }
  void set_weights(vector<float> weights) { weights_ = std::move(weights); }

 protected:
  AsymmetricHasherConfig conf_;
  shared_ptr<const DistanceMeasure> quantization_distance_;
  vector<float> weights_;
};

// 类型化训练参数类，包含投影、预处理函数等
template <typename T>
class TrainingOptionsTyped : public TrainingOptionsBase {
 public:
  TrainingOptionsTyped(const AsymmetricHasherConfig& config,
                       shared_ptr<const DistanceMeasure> quantization_distance)
      : TrainingOptionsBase(config, std::move(quantization_distance)) {}

  TrainingOptionsTyped(shared_ptr<const ChunkingProjection<T>> projector,
                       shared_ptr<const DistanceMeasure> quantization_distance)
      : TrainingOptionsBase(std::move(quantization_distance)),
        projector_(std::move(projector)) {}

  const shared_ptr<const ChunkingProjection<T>>& projector() const {
    return projector_;
  }

  // 预处理函数类型定义
  using PreprocessingFunction =
      std::function<StatusOr<Datapoint<T>>(const DatapointPtr<T>&)>;
  void set_preprocessing_function(PreprocessingFunction fn) {
    preprocessing_function_ = std::move(fn);
  }
  const PreprocessingFunction& preprocessing_function() const {
    return preprocessing_function_;
  }

 protected:
  shared_ptr<const ChunkingProjection<T>> projector_;

  PreprocessingFunction preprocessing_function_;
};

// TrainingOptionsTyped 模板类显式实例化声明
SCANN_INSTANTIATE_TYPED_CLASS(extern, TrainingOptionsTyped);

// asymmetric_hashing2 命名空间结束
}  // namespace asymmetric_hashing2
// research_scann 命名空间结束
}  // namespace research_scann

#endif
