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

#ifndef SCANN_BASE_HEALTH_STATS_COLLECTOR_H_
#define SCANN_BASE_HEALTH_STATS_COLLECTOR_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/partitioning/kmeans_tree_like_partitioner.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {


// 用于收集分区健康统计信息的工具类
// 支持分区大小、量化误差、失衡度等统计，便于分析索引分布和性能
template <typename Searcher, typename InDataType,
          typename InAccamulationType = InDataType,
          typename Partitioner =
              KMeansTreeLikePartitioner<typename Searcher::DataType>>
class HealthStatsCollector {
 public:
  using DataType = InDataType;
  using AccamulationType = InAccamulationType;
  using HealthStats = typename Searcher::HealthStats;

  // 初始化统计器，收集分区信息
  Status Initialize(const Searcher& searcher);

  // 获取当前健康统计结果
  absl::StatusOr<HealthStats> GetHealthStats();

  // 是否启用统计
  bool IsEnabled() const { return is_enabled_; }

  // 分区数量
  uint32_t NumTokens() const { return sizes_by_token_.size(); }

  // 分区操作相关接口
  void AddPartition();
  void SwapPartitions(int32_t token1, int32_t token2);
  void RemoveLastPartition();
  void Resize(size_t n);
  void Reserve(size_t n);

  // 添加/移除分区内数据点的统计
  void AddStats(int32_t token, absl::Span<const DatapointIndex> datapoints) {
    AddStats(absl::MakeConstSpan({token}), datapoints);
  }
  template <typename Tokens>
  void AddStats(const Tokens& tokens,
                absl::Span<const DatapointIndex> datapoints);
  void SubtractStats(int32_t token,
                     absl::Span<const DatapointIndex> datapoints) {
    SubtractStats(absl::MakeConstSpan({token}), datapoints);
  }
  template <typename Tokens>
  void SubtractStats(const Tokens& tokens,
                     absl::Span<const DatapointIndex> datapoints);

  // 移除整个分区统计
  void SubtractPartition(int32_t token);

  // 更新分区中心点（如聚类中心变化时）
  void UpdatePartitionCentroid(int32_t token,
                               DatapointPtr<DataType> new_centroid,
                               DatapointPtr<DataType> old_centroid);

 private:
  // 初始化分区中心点
  Status InitializeCentroids(const Searcher& searcher);

  // 统计操作类型
  enum class Op {
    Add,
    Subtract,
  };
  // 批量更新分区统计
  template <typename Tokens>
  void StatsUpdate(const Tokens& tokens, Op op,
                   absl::Span<const DatapointIndex> datapoints);

  // 单数据点统计增/减
  void Add(int32_t token, DatapointPtr<DataType> dp_ptr,
           DatapointPtr<DataType> center);
  void Add(int32_t token);
  void Subtract(int32_t token, DatapointPtr<DataType> dp_ptr,
                DatapointPtr<DataType> center);
  void Subtract(int32_t token);

  // 统计向量增量更新
  static void AddDelta(Datapoint<InAccamulationType>& dst,
                       DatapointPtr<DataType> new_dp,
                       DatapointPtr<DataType> old_dp, int times = 1);

  // 计算分区失衡度
  void ComputeAvgRelativeImbalance();

  // 获取数据点指针
  DatapointPtr<DataType> GetDatapointPtr(
      DatapointIndex i, Datapoint<typename Searcher::DataType>* storage) const {
    return searcher_->GetDatapointPtr(i);
  }

  // 统计相关成员变量
  const Searcher* searcher_ = nullptr;
  InAccamulationType sum_squared_quantization_error_ = 0;
  double partition_weighted_avg_relative_imbalance_ = 0;
  double partition_avg_relative_positive_imbalance_ = 0;
  uint64_t sum_partition_sizes_ = 0;

  std::vector<Datapoint<InAccamulationType>> sum_qe_by_token_; // 每个分区的量化误差和
  std::vector<uint32_t> sizes_by_token_; // 每个分区的大小
  std::vector<InAccamulationType> squared_quantization_error_by_token_; // 每个分区的量化误差
  std::shared_ptr<Partitioner> centroids_; // 分区中心点
  bool is_enabled_ = false;

  // 数据类型是否一致
  static constexpr bool kCentroidAndDPAreSameType =
      std::is_same_v<DataType, typename Searcher::DataType>;
};

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 初始化分区统计信息，包括分区大小、量化误差等
Status HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                            Partitioner>::Initialize(const Searcher& searcher) {
  *this = HealthStatsCollector();
  is_enabled_ = true;
  searcher_ = &searcher;
  SCANN_RETURN_IF_ERROR(InitializeCentroids(searcher));

  // 获取每个分区的数据点索引
  ConstSpan<std::vector<DatapointIndex>> datapoints_by_token =
      searcher.datapoints_by_token();
  Reserve(datapoints_by_token.size());
  for (const auto& dps : datapoints_by_token) {
    sum_partition_sizes_ += dps.size();
    sizes_by_token_.push_back(dps.size());
    sum_qe_by_token_.emplace_back();
    squared_quantization_error_by_token_.emplace_back();
  }

  // 统计每个分区的量化误差和
  const auto* dataset = searcher.dataset();
  if constexpr (kCentroidAndDPAreSameType) {
    if (dataset && !dataset->empty()) {
      const auto& centroids = centroids_->LeafCenters();

      if (dataset[0].dimensionality() == centroids[0].dimensionality()) {
        const auto& ds = *dataset;
        InAccamulationType total_squared_qe = 0.0;

        for (const auto& [token, dps] : Enumerate(datapoints_by_token)) {
          Datapoint<InAccamulationType> sum_dims;
          sum_dims.ZeroFill(ds.dimensionality());
          SCANN_RET_CHECK_EQ(sum_dims.dimensionality(), ds.dimensionality());
          SCANN_RET_CHECK_EQ(sum_dims.values_span().size(),
                             ds.dimensionality());
          DatapointPtr<DataType> centroid = centroids[token];
          InAccamulationType v = 0;
          for (auto dp_idx : dps) {
            v += SquaredL2DistanceBetween(ds[dp_idx], centroid); // 计算量化误差
            AddDelta(sum_dims, ds[dp_idx], centroids[token]);    // 统计误差向量
          }
          squared_quantization_error_by_token_[token] = v;
          sum_qe_by_token_[token] = std::move(sum_dims);
          total_squared_qe += v;
        }

        sum_squared_quantization_error_ = total_squared_qe;
      }
    }
  }

  // 计算分区失衡度
  ComputeAvgRelativeImbalance();
  return OkStatus();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 获取当前分区健康统计结果，包括平均量化误差、分区失衡度等
absl::StatusOr<typename HealthStatsCollector<
  Searcher, InDataType, InAccamulationType, Partitioner>::HealthStats>
HealthStatsCollector<Searcher, InDataType, InAccamulationType,
           Partitioner>::GetHealthStats() {
  HealthStats r;
  if (sum_partition_sizes_ > 0) {
  r.avg_quantization_error =
    sqrt(sum_squared_quantization_error_ / sum_partition_sizes_);
  }

  r.sum_partition_sizes = sum_partition_sizes_;

  ComputeAvgRelativeImbalance();
  r.partition_weighted_avg_relative_imbalance =
    partition_weighted_avg_relative_imbalance_;
  r.partition_avg_relative_positive_imbalance =
    partition_avg_relative_positive_imbalance_;
  return r;
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 新增一个分区（用于动态分区场景）
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::AddPartition() {
  if (!is_enabled_) return;
  sum_qe_by_token_.emplace_back();
  sizes_by_token_.push_back(0);
  squared_quantization_error_by_token_.push_back(0);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 交换两个分区的统计信息
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::SwapPartitions(int32_t token1,
                                                       int32_t token2) {
  if (!is_enabled_) return;
  std::swap(sum_qe_by_token_[token1], sum_qe_by_token_[token2]);
  std::swap(sizes_by_token_[token1], sizes_by_token_[token2]);
  std::swap(squared_quantization_error_by_token_[token1],
            squared_quantization_error_by_token_[token2]);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 移除最后一个分区
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::RemoveLastPartition() {
  if (!is_enabled_) return;
  sum_qe_by_token_.pop_back();
  sizes_by_token_.pop_back();
  squared_quantization_error_by_token_.pop_back();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 调整分区数量
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Resize(size_t n) {
  if (!is_enabled_) return;
  sum_qe_by_token_.resize(n);
  sizes_by_token_.resize(n);
  squared_quantization_error_by_token_.resize(n);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 预分配分区空间
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Reserve(size_t n) {
  if (!is_enabled_) return;
  sum_qe_by_token_.reserve(n);
  sizes_by_token_.reserve(n);
  squared_quantization_error_by_token_.reserve(n);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 批量添加分区统计（如批量插入数据点）
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::AddStats(const Tokens& tokens,
                           absl::Span<const DatapointIndex> datapoints) {
  if (!is_enabled_) return;
  StatsUpdate(tokens, Op::Add, datapoints);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 批量移除分区统计（如批量删除数据点）
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::SubtractStats(const Tokens& tokens,
                                absl::Span<const DatapointIndex> datapoints) {
  if (!is_enabled_) return;
  StatsUpdate(tokens, Op::Subtract, datapoints);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 移除指定分区的所有统计信息
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::SubtractPartition(int32_t token) {
  if (!is_enabled_) return;

  sum_squared_quantization_error_ -=
      squared_quantization_error_by_token_[token];
  sum_partition_sizes_ -= sizes_by_token_[token];

  sum_qe_by_token_[token].ZeroFill(sum_qe_by_token_[token].dimensionality());
  sizes_by_token_[token] = 0;
  squared_quantization_error_by_token_[token] = 0;
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 更新分区中心点后，重新计算分区统计信息
template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::UpdatePartitionCentroid(int32_t token,
                                          DatapointPtr<DataType> new_centroid,
                                          DatapointPtr<DataType> old_centroid) {
  if (!is_enabled_) return;

  if constexpr (kCentroidAndDPAreSameType) {
    if (sizes_by_token_[token] == 0) return;

    if (sum_qe_by_token_[token].dimensionality() == 0) {
      sum_qe_by_token_[token].ZeroFill(new_centroid.dimensionality());
    }
    InAccamulationType delta = 0;
    for (int dim = 0; dim < new_centroid.dimensionality(); ++dim) {
      auto d =
          new_centroid.values_span()[dim] - old_centroid.values_span()[dim];
      InAccamulationType v = sizes_by_token_[token] * d * d -
                             2 * d * sum_qe_by_token_[token].values_span()[dim];
      delta += v;
    }
    sum_squared_quantization_error_ += delta;
    squared_quantization_error_by_token_[token] += delta;

    AddDelta(sum_qe_by_token_[token], old_centroid, new_centroid,
             sizes_by_token_[token]);
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 初始化分区中心点（聚类中心），确保数据库和查询分区一致
Status HealthStatsCollector<Searcher, InDataType, InAccamulationType,
              Partitioner>::InitializeCentroids(const Searcher&
                                  searcher) {
  auto pd = std::dynamic_pointer_cast<const Partitioner>(
    searcher_->database_tokenizer());
  auto pq = std::dynamic_pointer_cast<const Partitioner>(
    searcher_->query_tokenizer());
  SCANN_RET_CHECK(pd != nullptr);
  SCANN_RET_CHECK(pq != nullptr);
  SCANN_RET_CHECK_EQ(pd->kmeans_tree(), pq->kmeans_tree())
    << "Centroids in database partitioner and query partitioner must be "
    << "identical";
  SCANN_RET_CHECK(pq->kmeans_tree()->is_flat())
    << "The query/database partitioner must contain a single flat "
    << "KMeansTree.";

  centroids_ = std::const_pointer_cast<Partitioner>(pq);
  return OkStatus();
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 批量更新分区统计（支持有无中心点的情况）
template <typename Tokens>
void HealthStatsCollector<
    Searcher, InDataType, InAccamulationType,
    Partitioner>::StatsUpdate(const Tokens& tokens, Op op,
                              absl::Span<const DatapointIndex> datapoints) {
  const auto* dataset = searcher_->dataset();
  auto UpdateWithoutCentroids = [&]() {
    for (int32_t token : tokens) {
      if (op == Op::Add) {
        Add(token);
      } else {
        Subtract(token);
      }
    }
  };
  if constexpr (kCentroidAndDPAreSameType) {
    if (dataset && !dataset->empty()) {
      const auto& centroids = centroids_->LeafCenters();
      Datapoint<DataType> dp;
      for (int32_t token : tokens) {
        DatapointPtr<DataType> centroid = centroids[token];
        for (DatapointIndex dp_idx : datapoints) {
          auto d_ptr = GetDatapointPtr(dp_idx, &dp);
          if (op == Op::Add) {
            Add(token, d_ptr, centroid); // 增加统计
          } else {
            Subtract(token, d_ptr, centroid); // 减少统计
          }
        }
      }
    } else {
      UpdateWithoutCentroids();
    }
  } else {
    UpdateWithoutCentroids();
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 增加单个数据点到分区的统计信息
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Add(int32_t token,
                                            DatapointPtr<DataType> dp_ptr,
                                            DatapointPtr<DataType> center) {
  double quantize_err = SquaredL2DistanceBetween(dp_ptr, center);
  sum_squared_quantization_error_ += quantize_err;

  AddDelta(sum_qe_by_token_[token], dp_ptr, center);
  squared_quantization_error_by_token_[token] += quantize_err;
  Add(token);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 增加分区大小计数
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Add(int32_t token) {
  ++sum_partition_sizes_;
  ++sizes_by_token_[token];
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 移除单个数据点的分区统计信息
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Subtract(int32_t token,
                                                 DatapointPtr<DataType> dp_ptr,
                                                 DatapointPtr<DataType>
                                                     center) {
  double quantize_err = SquaredL2DistanceBetween(dp_ptr, center);
  sum_squared_quantization_error_ -= quantize_err;

  AddDelta(sum_qe_by_token_[token], center, dp_ptr);
  squared_quantization_error_by_token_[token] -= quantize_err;
  Subtract(token);
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 减少分区大小计数
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::Subtract(int32_t token) {
  --sum_partition_sizes_;
  --sizes_by_token_[token];
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>

// 统计向量增量更新（用于误差累加）
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::AddDelta(Datapoint<InAccamulationType>&
                                                     dst,
                                                 DatapointPtr<DataType> new_dp,
                                                 DatapointPtr<DataType> old_dp,
                                                 int times) {
  if (dst.dimensionality() == 0) dst.ZeroFill(new_dp.dimensionality());
  for (int dim = 0; dim < dst.dimensionality(); ++dim) {
    dst.mutable_values_span()[dim] +=
        (new_dp.values_span()[dim] - old_dp.values_span()[dim]) * times;
  }
}

template <typename Searcher, typename InDataType, typename InAccamulationType,
          typename Partitioner>
// 计算分区失衡度（衡量分区分布均匀性）
void HealthStatsCollector<Searcher, InDataType, InAccamulationType,
                          Partitioner>::ComputeAvgRelativeImbalance() {
  partition_weighted_avg_relative_imbalance_ = 0;
  partition_avg_relative_positive_imbalance_ = 0;
  if (sum_partition_sizes_ == 0) return;

  // 加权失衡度
  for (const auto& partition_size : sizes_by_token_) {
    partition_weighted_avg_relative_imbalance_ +=
        1.0 * partition_size / sum_partition_sizes_ * partition_size;
  }
  partition_weighted_avg_relative_imbalance_ /=
      1.0 * sum_partition_sizes_ / sizes_by_token_.size();
  partition_weighted_avg_relative_imbalance_ -= 1.0;

  // 正失衡度（只统计大于均值的分区）
  double best = 1.0 * sum_partition_sizes_ / sizes_by_token_.size();
  uint32_t n_positive = 0;
  for (uint32_t partition_size : sizes_by_token_) {
    if (partition_size <= best) continue;
    ++n_positive;
    partition_avg_relative_positive_imbalance_ += partition_size - best;
  }
  if (n_positive > 0 && best > 0) {
    partition_avg_relative_positive_imbalance_ /= n_positive;
    partition_avg_relative_positive_imbalance_ /= best;
  }
}

}  // namespace research_scann
#endif
