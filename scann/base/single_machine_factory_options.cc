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

#include "scann/base/single_machine_factory_options.h"

#include "scann/utils/input_data_utils.h"

namespace research_scann {

// 计算工厂选项下所有相关数据集的维度一致性
StatusOr<DimensionIndex>
SingleMachineFactoryOptions::ComputeConsistentDimensionality(
    const ScannConfig& config, const Dataset* dataset) const {
  // 调用工具函数，检查主数据集、哈希数据集、预量化数据集、bfloat16数据集的维度是否一致
  return ComputeConsistentDimensionalityFromIndex(
      config, dataset, hashed_dataset.get(), pre_quantized_fixed_point.get(),
      bfloat16_dataset.get());
}

// research_scann 命名空间结束
}  // namespace research_scann
