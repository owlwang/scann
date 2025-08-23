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



#ifndef SCANN_BRUTE_FORCE_BRUTE_FORCE_H_
#define SCANN_BRUTE_FORCE_BRUTE_FORCE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"

namespace research_scann {


// BruteForceSearcher：暴力搜索器，支持稠密/稀疏数据集的最近邻搜索
template <typename T>
class BruteForceSearcher final : public SingleMachineSearcherBase<T> {
 public:
  // 构造函数，初始化距离度量、数据集、默认参数
  BruteForceSearcher(shared_ptr<const DistanceMeasure> distance,
                     shared_ptr<const TypedDataset<T>> dataset,
                     const int32_t default_pre_reordering_num_neighbors,
                     const float default_pre_reordering_epsilon);

  ~BruteForceSearcher() final;

  // 是否支持Crowding（始终返回true）
  bool supports_crowding() const final { return true; }

  // 推荐批量搜索的batch size
  DatapointIndex optimal_batch_size() const final {
    return supports_low_level_batching_ ? 128 : 1;
  }

  // 设置线程池（用于并发批量搜索）
  void set_thread_pool(std::shared_ptr<ThreadPool> p) { pool_ = std::move(p); }

  // 设置最小距离过滤阈值
  void set_min_distance(float min_distance) { min_distance_ = min_distance; }

  // 工厂方法，创建BruteForceSearcher
  StatusOr<const SingleMachineSearcherBase<T>*> CreateBruteForceSearcher(
      const DistanceMeasureConfig&,
      unique_ptr<SingleMachineSearcherBase<T>>* storage) const final {
    return this;
  }

  using PrecomputedMutationArtifacts =
      UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;

  // Mutator：支持数据集动态变更（增删改）的辅助类
  class Mutator : public SingleMachineSearcherBase<T>::Mutator {
   public:
    using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    // 创建Mutator对象
    static StatusOr<unique_ptr<typename BruteForceSearcher<T>::Mutator>> Create(
        BruteForceSearcher<T>* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final {}
    // 获取数据点
    absl::StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex i) const final;
    // 增加数据点
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<T>& dptr,
                                          string_view docid,
                                          const MutationOptions& mo) final;
    // 根据docid删除数据点
    Status RemoveDatapoint(string_view docid) final;
    // 预分配空间
    void Reserve(size_t size) final;
    // 根据索引删除数据点
    Status RemoveDatapoint(DatapointIndex index) final;
    // 更新数据点（docid或索引）
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             string_view docid,
                                             const MutationOptions& mo) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<T>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions& mo) final;

    // 增量维护（如索引重建）
    StatusOr<std::optional<ScannConfig>> IncrementalMaintenance() final;

   private:
    explicit Mutator(BruteForceSearcher<T>* searcher) : searcher_(searcher) {}

    // docid查找数据点索引
    StatusOr<DatapointIndex> LookupDatapointIndexOrError(
        string_view docid) const;

    BruteForceSearcher<T>* searcher_;
  };

  // 获取Mutator对象
  StatusOr<typename SingleMachineSearcherBase<T>::Mutator*> GetMutator()
      const final;

 protected:
  // 单点邻居搜索主入口
  Status FindNeighborsImpl(const DatapointPtr<T>& query,
                           const SearchParameters& params,
                           NNResultsVector* result) const final;

  // 批量邻居搜索主入口
  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const final;

  // 支持索引映射的批量邻居搜索
  Status FindNeighborsBatchedImpl(
      const TypedDataset<T>& queries, ConstSpan<SearchParameters> params,
      MutableSpan<FastTopNeighbors<float>*> results,
      ConstSpan<DatapointIndex> datapoint_index_mapping) const final;

  // 启用Crowding功能
  Status EnableCrowdingImpl(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
      ConstSpan<std::string> crowding_dimension_names) final;

 private:
  // 单点邻居搜索内部实现（支持MinDistance）
  template <bool kUseMinDistance, typename TopN>
  Status FindNeighborsInternal(const DatapointPtr<T>& query,
                               const SearchParameters& params,
                               TopN* top_n_ptr) const;

  // 单点邻居搜索（支持限制列表）
  template <bool kUseMinDistance, typename WhitelistIterator, typename TopN>
  void FindNeighborsOneToOneInternal(const DatapointPtr<T>& query,
                                     const SearchParameters& params,
                                     WhitelistIterator* allowlist_iterator,
                                     TopN* top_n_ptr) const;

  // 批量搜索（float/double类型专用）
  template <typename Float>
  enable_if_t<IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  // float类型批量搜索的高效实现
  void FinishBatchedSearchSimple(const DenseDataset<float>& db,
                                 const DenseDataset<float>& queries,
                                 ConstSpan<SearchParameters> params,
                                 MutableSpan<NNResultsVector> results) const;

  // 非float/double类型批量搜索（直接报错）
  template <typename Float>
  enable_if_t<!IsSameAny<Float, float, double>(), void> FinishBatchedSearch(
      const DenseDataset<Float>& db, const DenseDataset<Float>& queries,
      ConstSpan<SearchParameters> params,
      MutableSpan<NNResultsVector> results) const;

  // 距离度量对象
  shared_ptr<const DistanceMeasure> distance_;

  // 是否支持高效批量搜索
  const bool supports_low_level_batching_;

  // 线程池（用于并发批量搜索）
  std::shared_ptr<ThreadPool> pool_;

  // 最小距离过滤阈值
  float min_distance_ = -numeric_limits<float>::infinity();

  // Mutator对象（用于动态数据集变更）
  mutable unique_ptr<typename BruteForceSearcher<T>::Mutator> mutator_ =
      nullptr;

  // 是否不可变（只读）
  bool is_immutable_ = false;
};

SCANN_INSTANTIATE_TYPED_CLASS(extern, BruteForceSearcher);

}  // namespace research_scann

#endif
