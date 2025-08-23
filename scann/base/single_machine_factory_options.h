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

#ifndef SCANN_BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_
#define SCANN_BASE_SINGLE_MACHINE_FACTORY_OPTIONS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "scann/data_format/dataset.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/partitioning/partitioner.pb.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"
#include "scann/utils/types.h"

namespace research_scann {
template <typename T>
class DenseDataset;
template <typename T>
class TypedDataset;
class KMeansTree;
template <typename T>
class SingleMachineSearcherBase;
class ScannConfig;

// 单机搜索器工厂选项结构体，包含构建搜索器所需的所有辅助数据和参数
struct SingleMachineFactoryOptions {
  SingleMachineFactoryOptions() = default;

  // 计算一致的维度信息（根据配置和数据集）
  StatusOr<DimensionIndex> ComputeConsistentDimensionality(
      const ScannConfig& config, const Dataset* dataset = nullptr) const;

  // 数据类型标签
  TypeTag type_tag = kInvalidTypeTag;

  // 按 token 分组的数据点索引（分区用）
  shared_ptr<vector<std::vector<DatapointIndex>>> datapoints_by_token;

  // 预量化定点数据（int8 量化相关）
  shared_ptr<PreQuantizedFixedPoint> pre_quantized_fixed_point;

  // 哈希数据集（uint8）
  shared_ptr<DenseDataset<uint8_t>> hashed_dataset;

  // SOAR 哈希数据集（uint8，特殊分区用）
  shared_ptr<DenseDataset<uint8_t>> soar_hashed_dataset;

  // bfloat16 数据集
  shared_ptr<DenseDataset<int16_t>> bfloat16_dataset;

  // AH（Asymmetric Hashing）码本
  std::shared_ptr<CentersForAllSubspaces> ah_codebook;

  // 序列化分区器
  std::shared_ptr<SerializedPartitioner> serialized_partitioner;

  // KMeansTree 分区树
  std::shared_ptr<const KMeansTree> kmeans_tree;

  // 哈希参数（如哈希函数等）
  std::shared_ptr<std::vector<std::string>> hash_parameters;

  // 拥挤度属性（支持 crowding）
  shared_ptr<vector<int64_t>> crowding_attributes;
  shared_ptr<vector<std::string>> crowding_dimension_names;

  // 并行线程池（加速构建/查询）
  shared_ptr<ThreadPool> parallelization_pool;
};

}  // namespace research_scann

#endif
