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



#ifndef SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_H_
#define SCANN_HASHES_ASYMMETRIC_HASHING2_TRAINING_H_

#include <utility>

#include "scann/data_format/dataset.h"
#include "scann/hashes/asymmetric_hashing2/training_model.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/hashes/internal/asymmetric_hashing_impl.h"
#include "scann/hashes/internal/stacked_quantizers.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing2 {

// 单机训练主流程，根据量化方案分支处理，训练中心并设置投影
template <typename T>
StatusOr<unique_ptr<Model<T>>> TrainSingleMachine(
    const TypedDataset<T>& dataset, const TrainingOptions<T>& params,
    shared_ptr<ThreadPool> pool = nullptr) {
  unique_ptr<Model<T>> result;
  if (params.config().quantization_scheme() ==
      AsymmetricHasherConfig::STACKED) {
    // 堆叠量化器，仅支持稠密数据集
    if (!dataset.IsDense())
      return InvalidArgumentError(
          "Stacked quantizers can only process dense datasets.");
    const auto& dense = down_cast<const DenseDataset<T>&>(dataset);
    SCANN_ASSIGN_OR_RETURN(
        auto centers,
        ::research_scann::asymmetric_hashing_internal::StackedQuantizers<
            T>::Train(dense, params, pool));
    SCANN_ASSIGN_OR_RETURN(
        result, Model<T>::FromCenters(std::move(centers),
                                      params.config().quantization_scheme()));
  } else if (params.config().quantization_scheme() ==
             AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    // PRODUCT_AND_BIAS 量化，去除 bias 维度
    const auto& dense = down_cast<const DenseDataset<T>&>(dataset);
    DenseDataset<T> dataset_no_bias;
    dataset_no_bias.set_dimensionality(dense.dimensionality() - 1);
    dataset_no_bias.Reserve(dense.size());
    for (const auto& dp : dense) {
      SCANN_RETURN_IF_ERROR(dataset_no_bias.Append(
          MakeDatapointPtr(dp.values(), dp.dimensionality() - 1)));
    }

    SCANN_ASSIGN_OR_RETURN(
        auto centers,
        ::research_scann::asymmetric_hashing_internal::TrainAsymmetricHashing(
            dataset_no_bias, params, pool));
    auto converted = asymmetric_hashing_internal::ConvertCentersIfNecessary<T>(
        std::move(centers));
    SCANN_ASSIGN_OR_RETURN(
        result, Model<T>::FromCenters(std::move(converted),
                                      params.config().quantization_scheme()));
  } else {
    // 普通 PRODUCT 量化流程
    SCANN_ASSIGN_OR_RETURN(
        auto centers,
        ::research_scann::asymmetric_hashing_internal::TrainAsymmetricHashing(
            dataset, params, pool));
    auto converted = asymmetric_hashing_internal::ConvertCentersIfNecessary<T>(
        std::move(centers));
    SCANN_ASSIGN_OR_RETURN(
        result, Model<T>::FromCenters(std::move(converted),
                                      params.config().quantization_scheme()));
  }
  // 设置投影对象
  result->SetProjection(params.projector());
  return {std::move(result)};
}

// asymmetric_hashing2 命名空间结束
}  // namespace asymmetric_hashing2
// research_scann 命名空间结束
}  // namespace research_scann

#endif
