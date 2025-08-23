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

#ifndef SCANN_BRUTE_FORCE_BFLOAT16_BRUTE_FORCE_H_
#define SCANN_BRUTE_FORCE_BFLOAT16_BRUTE_FORCE_H_

#include <cmath>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// Bfloat16 精度暴力搜索器：用于高效地在 bfloat16 量化空间进行暴力最近邻搜索
class Bfloat16BruteForceSearcher final
    : public SingleMachineSearcherBase<float> {
 public:
  // 构造函数：通过原始 float 数据集和距离度量初始化 bfloat16 搜索器
  Bfloat16BruteForceSearcher(shared_ptr<const DistanceMeasure> distance,
                             shared_ptr<const DenseDataset<float>> dataset,
                             int32_t default_num_neighbors,
                             float default_epsilon,
                             float noise_shaping_threshold = NAN);

  // 构造函数：直接通过 bfloat16 数据集初始化
  Bfloat16BruteForceSearcher(
      shared_ptr<const DistanceMeasure> distance,
      shared_ptr<const DenseDataset<int16_t>> bfloat16_dataset,
      int32_t default_num_neighbors, float default_epsilon,
      float noise_shaping_threshold = NAN);

  // 创建通用暴力搜索器（兼容接口）
  StatusOr<const SingleMachineSearcherBase<float>*> CreateBruteForceSearcher(
      const DistanceMeasureConfig& distance_config,
      unique_ptr<SingleMachineSearcherBase<float>>* storage) const final;

  // 析构函数
  ~Bfloat16BruteForceSearcher() override = default;

  // 是否支持 crowding（多样性约束）
  bool supports_crowding() const final { return true; }

  class Mutator : public SingleMachineSearcherBase<float>::Mutator {
   public:
    // Mutator：支持数据集的动态增删改
    using PrecomputedMutationArtifacts =
        UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    // 创建 Mutator 实例
    static StatusOr<unique_ptr<Mutator>> Create(
        Bfloat16BruteForceSearcher* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final = default;

    // 获取指定索引的数据点
    StatusOr<Datapoint<float>> GetDatapoint(DatapointIndex i) const final;
    // 增加数据点
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr,
                                          string_view docid,
                                          const MutationOptions&) final;
    // 通过 docid 删除数据点
    Status RemoveDatapoint(string_view docid) final;
    // 预分配空间
    void Reserve(size_t size) final;
    // 通过索引删除数据点
    Status RemoveDatapoint(DatapointIndex index) final;
    // 更新数据点（通过 docid 或索引）
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             string_view docid,
                                             const MutationOptions&) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions&) final;

   private:
    // 构造函数（私有）：持有量化器和数据集变更器
    Mutator(Bfloat16BruteForceSearcher* searcher,
            TypedDataset<int16_t>::Mutator* quantized_dataset_mutator)
        : searcher_(searcher),
          quantized_dataset_mutator_(quantized_dataset_mutator) {}
    // 查找 docid 对应索引
    StatusOr<DatapointIndex> LookupDatapointIndexOrError(
        string_view docid) const;

    Bfloat16BruteForceSearcher* searcher_;
    TypedDataset<int16_t>::Mutator* quantized_dataset_mutator_;
  };

    // 获取 Mutator（用于支持动态数据集变更）
    StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
            const final;

    // 提取工厂选项（用于重排序加速等）
    StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
            override;

protected:
    // 实际查找邻居的实现（bfloat16 距离计算）
    Status FindNeighborsImpl(const DatapointPtr<float>& query,
                                                     const SearchParameters& params,
                                                     NNResultsVector* result) const final;

    // crowding 属性使能实现
    Status EnableCrowdingImpl(
            ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
            ConstSpan<std::string> crowding_dimension_names) final;

private:
    // 是否需要原始数据集（bfloat16 暴力搜索不需要）
    bool impl_needs_dataset() const override { return false; }

    // 是否为 dot product 距离类型
    bool is_dot_product_;
    // bfloat16 量化后的数据集
    shared_ptr<const DenseDataset<int16_t>> bfloat16_dataset_;

    // 噪声整形阈值
    const float noise_shaping_threshold_ = NAN;

    // Mutator（支持动态数据集变更）
    mutable unique_ptr<Mutator> mutator_ = nullptr;
};

}  // namespace research_scann

#endif
