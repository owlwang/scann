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



#ifndef SCANN_DATA_FORMAT_SPARSE_LOW_LEVEL_H_
#define SCANN_DATA_FORMAT_SPARSE_LOW_LEVEL_H_

#include "scann/utils/types.h"

namespace research_scann {

// 稀疏数据点底层结构，存储索引、值和非零条目数
template <typename IndexT, typename ValueT>
struct SparseLowLevelDatapoint {
  SparseLowLevelDatapoint(IndexT* indices, ValueT* values,
                          DimensionIndex nonzero_entries)
      : indices(indices), values(values), nonzero_entries(nonzero_entries) {}

  IndexT* indices = nullptr;      // 非零维度索引指针
  ValueT* values = nullptr;       // 非零维度值指针
  DimensionIndex nonzero_entries = 0; // 非零条目数
};

// SparseDatasetLowLevel：稀疏数据集底层实现，支持追加、删除、预分配、内存管理等
template <typename IndexT, typename ValueT, typename StartOffsetT = size_t>
class SparseDatasetLowLevel {
 public:
  SparseDatasetLowLevel() {}

  // 构造函数，使用已有索引、值和起始偏移构造稀疏数据集
  SparseDatasetLowLevel(std::vector<IndexT> indices, std::vector<ValueT> values,
                        std::vector<StartOffsetT> start_offsets)
      : indices_(std::move(indices)),
        values_(std::move(values)),
        start_offsets_(std::move(start_offsets)) {
    if (!values_.empty()) {
      CHECK_EQ(values_.size(), indices_.size());
    }
    if (!indices_.empty()) {
      CHECK_GE(start_offsets_.size(), 2);
    }
  }

  // 追加一个稀疏数据点（索引和值），更新起始偏移
  void Append(ConstSpan<IndexT> indices, ConstSpan<ValueT> values) {
    if (!values.empty() || !values_.empty()) {
      DCHECK_EQ(indices.size(), values.size());
    }
    indices_.insert(indices_.end(), indices.begin(), indices.end());
    values_.insert(values_.end(), values.begin(), values.end());

    CHECK_LE(indices_.size(), numeric_limits<StartOffsetT>::max());
    start_offsets_.push_back(indices_.size());
  }

  // 删除最后一个稀疏数据点
  void PopBack() {
    DCHECK_GT(start_offsets_.size(), 1);
    start_offsets_.pop_back();
    indices_.resize(start_offsets_.back());
    if (!values_.empty()) values_.resize(indices_.size());
  }

  // 获取第 i 个稀疏数据点的底层结构
  SparseLowLevelDatapoint<IndexT, ValueT> Get(size_t i) {
    DCHECK_LT(i + 1, start_offsets_.size());
    const size_t end_offset = start_offsets_[i + 1];
    const size_t start_offset = start_offsets_[i];
    const DimensionIndex nonzero_entries = end_offset - start_offset;
    ValueT* values_ptr =
        (values_.empty()) ? nullptr : (values_.data() + start_offset);
    return SparseLowLevelDatapoint<IndexT, ValueT>(
        indices_.data() + start_offset, values_ptr, nonzero_entries);
  }

  // 获取第 i 个稀疏数据点的非零条目数
  DimensionIndex NonzeroEntriesForDatapoint(size_t i) const {
    DCHECK_LT(i + 1, start_offsets_.size());
    return start_offsets_[i + 1] - start_offsets_[i];
  }

  // 获取所有索引/值/起始偏移的只读视图
  ConstSpan<IndexT> indices() const { return indices_; }
  ConstSpan<ValueT> values() const { return values_; }
  ConstSpan<StartOffsetT> start_offsets() const { return start_offsets_; }

  // 预分配空间（点数/条目数/二值）
  void Reserve(size_t n_points) { start_offsets_.reserve(n_points + 1); }
  void Reserve(size_t n_points, size_t n_entries) {
    ReserveForBinaryData(n_points, n_entries);
    values_.reserve(n_entries);
  }
  void ReserveForBinaryData(size_t n_points, size_t n_entries) {
    Reserve(n_points);
    indices_.reserve(n_entries);
  }

  // 清空所有数据
  void Clear() {
    FreeBackingStorage(&indices_);
    FreeBackingStorage(&values_);
    FreeBackingStorage(&start_offsets_);
    start_offsets_ = {0};
  }

  // 收缩内存到实际大小
  void ShrinkToFit() {
    start_offsets_.shrink_to_fit();
    if (indices_.size() * sizeof(indices_[0]) <
        values_.size() * sizeof(values_[0])) {
      indices_.shrink_to_fit();
      values_.shrink_to_fit();
    } else {
      values_.shrink_to_fit();
      indices_.shrink_to_fit();
    }
  }

  // 计算底层稀疏数据集占用内存
  size_t MemoryUsage() const {
    return sizeof(ValueT) * values_.capacity() +
           sizeof(IndexT) * indices_.capacity() +
           sizeof(StartOffsetT) * start_offsets_.capacity();
  }

  // 预取第 i 个数据点的索引和值到本地缓存（加速访问）
  void Prefetch(size_t i) const {
    const StartOffsetT start_offset = start_offsets_[i];
    absl::PrefetchToLocalCacheNta(
        reinterpret_cast<const char*>(indices_.data() + start_offset));
    absl::PrefetchToLocalCacheNta(
        reinterpret_cast<const char*>(values_.data() + start_offset));
  }

  // 数据点数量
  size_t size() const { return start_offsets_.size() - 1; }

 private:
  std::vector<IndexT> indices_;
  std::vector<ValueT> values_;
  std::vector<StartOffsetT> start_offsets_ = {0};
};

}  // namespace research_scann

#endif
