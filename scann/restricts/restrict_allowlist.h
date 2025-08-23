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



#ifndef SCANN_RESTRICTS_RESTRICT_ALLOWLIST_H_
#define SCANN_RESTRICTS_RESTRICT_ALLOWLIST_H_

#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "gtest/gtest_prod.h"
#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/bit_iterator.h"
#include "scann/utils/bits.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class RestrictTokenMap;
class RestrictAllowlistConstView;

// 限制允许列表主类，支持高效的点级别允许/禁止管理，底层采用位数组实现
class RestrictAllowlist {
 public:
  // 构造与析构
  RestrictAllowlist(DatapointIndex num_points, bool default_allowlisted);
  RestrictAllowlist() : RestrictAllowlist(0, false) {}
  ~RestrictAllowlist();

  // 支持底层存储转移的构造
  RestrictAllowlist(std::vector<size_t>&& allowlist_array,
                    DatapointIndex num_points, bool default_allowlisted);

  // 拷贝与移动语义
  RestrictAllowlist(const RestrictAllowlist& rhs);
  RestrictAllowlist(RestrictAllowlist&& rhs) noexcept = default;
  RestrictAllowlist& operator=(const RestrictAllowlist& rhs);
  RestrictAllowlist& operator=(RestrictAllowlist&& rhs) = default;

  // 支持从只读视图构造
  explicit RestrictAllowlist(const RestrictAllowlistConstView& view);

  // 初始化/扩容/缩容
  void Initialize(DatapointIndex num_points, bool default_allowlisted);

  // 拷贝并扩容底层存储
  RestrictAllowlist CopyWithCapacity(
      DatapointIndex capacity,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  // 拷贝并设置新大小
  RestrictAllowlist CopyWithSize(
      DatapointIndex size, bool default_allowlisted,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  // 追加新点
  void Append(bool is_allowlisted);

  // 调整大小
  void Resize(size_t num_points, bool default_allowlisted);

  // 检查是否有空间追加
  bool CapacityAvailableForAppend(DatapointIndex dp_index) const;
  bool CapacityAvailableForAppend() const {
    return CapacityAvailableForAppend(num_points_);
  }

  // 查询点是否允许
  bool IsAllowlisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return IsAllowlistedNoBoundsCheck(dp_index);
  }

  // 查询点是否允许（超界时返回默认值）
  bool IsAllowlistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsAllowlisted(dp_index);
  }

  // 设置回收函数
  void set_allowlist_recycling_fn(
      std::function<void(std::vector<size_t>&&)> f) {
    allowlist_recycling_fn_ = std::move(f);
  }

  // 统计允许点数量
  DatapointIndex NumPointsAllowlisted() const;

  // 获取点总数
  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }

  // 位迭代器类型
  using Iterator = BitIterator<ConstSpan<size_t>, DatapointIndex>;

  // 允许点迭代器
  Iterator AllowlistedPointIterator() const {
    return Iterator(allowlist_array_);
  }

  // 获取包含指定点的字
  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    return allowlist_array_[dp_index / kBitsPerWord];
  }

  // 获取底层数据指针
  size_t* data() { return allowlist_array_.data(); }
  const size_t* data() const { return allowlist_array_.data(); }

  // 查找最低位 set 的索引
  static uint8_t FindLSBSetNonZeroNative(size_t word) {
    static_assert(sizeof(word) == 8 || sizeof(word) == 4, "");
    return (sizeof(word) == 8) ? bits::FindLSBSetNonZero64(word)
                               : bits::FindLSBSetNonZero(word);
  }

  // 位操作相关常量
  static constexpr size_t kBitsPerWord = sizeof(size_t) * CHAR_BIT;

  static constexpr size_t kOne = 1;

  static constexpr size_t kZero = 0;

  static constexpr size_t kAllOnes = ~kZero;

  static constexpr size_t kRoundDownMask = ~(kBitsPerWord - 1);

 private:
  bool IsAllowlistedNoBoundsCheck(DatapointIndex dp_index) const {
    return GetWordContainingDatapoint(dp_index) &
           (kOne << (dp_index % kBitsPerWord));
  }

  std::vector<size_t> allowlist_array_;

  DatapointIndex num_points_;

  std::function<void(std::vector<size_t>&&)> allowlist_recycling_fn_;

  friend class RestrictTokenMap;

  friend class RestrictAllowlistConstView;
  friend class RestrictAllowlistMutableView;

  FRIEND_TEST(RestrictAllowlist, ResizeUpwardDefaultAllowlisted);
  FRIEND_TEST(RestrictAllowlist, ResizeDownwardDefaultAllowlisted);
  FRIEND_TEST(RestrictAllowlist, ResizeDownwardDefaultNonAllowlisted);
  FRIEND_TEST(RestrictAllowlist, ResizeUpwardDefaultNonAllowlisted);
};

// 全允许列表（所有点均允许），用于特殊场景
class DummyAllowlist {
 public:
  explicit DummyAllowlist(DatapointIndex num_points);

  bool IsAllowlisted(DatapointIndex dp_index) const { return true; }

  class Iterator {
   public:
    DatapointIndex value() const { return value_; }
    bool Done() const { return value_ >= num_points_; }
    void Next() { ++value_; }

   private:
    friend class DummyAllowlist;

    explicit Iterator(DatapointIndex num_points);

    DatapointIndex value_;

    DatapointIndex num_points_;
  };

  Iterator AllowlistedPointIterator() const { return Iterator(num_points_); }

 private:
  DatapointIndex num_points_;
};

// 只读视图类，支持高效只读访问底层允许列表
class RestrictAllowlistConstView {
 public:
  RestrictAllowlistConstView() = default;

  explicit RestrictAllowlistConstView(const RestrictAllowlist& allowlist)
      : allowlist_array_(allowlist.allowlist_array_.data()),
        num_points_(allowlist.num_points_) {}

  explicit RestrictAllowlistConstView(const RestrictAllowlist* allowlist)
      : allowlist_array_(allowlist ? allowlist->allowlist_array_.data()
                                   : nullptr),
        num_points_(allowlist ? allowlist->num_points_ : 0) {}

  RestrictAllowlistConstView(ConstSpan<size_t> storage,
                             DatapointIndex num_points)
      : allowlist_array_(storage.data()), num_points_(num_points) {
    DCHECK_EQ(storage.size(),
              DivRoundUp(num_points, RestrictAllowlist::kBitsPerWord));
  }

  bool IsAllowlisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return GetWordContainingDatapoint(dp_index) &
           (RestrictAllowlist::kOne
            << (dp_index % RestrictAllowlist::kBitsPerWord));
  }

  bool IsAllowlistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsAllowlisted(dp_index);
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    return allowlist_array_[dp_index / RestrictAllowlist::kBitsPerWord];
  }

  const size_t* data() const { return allowlist_array_; }
  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }
  bool empty() const { return !allowlist_array_; }

  operator bool() const { return !empty(); }

  ABSL_DEPRECATED("Use IsAllowlisted instead.")
  bool IsWhitelisted(DatapointIndex dp_index) const {
    return IsAllowlisted(dp_index);
  }
  ABSL_DEPRECATED("Use IsAllowlistedWithDefault instead.")
  bool IsWhitelistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    return IsAllowlistedWithDefault(dp_index, default_value);
  }

 private:
  const size_t* allowlist_array_ = nullptr;
  DatapointIndex num_points_ = 0;
  friend class RestrictAllowlist;
};

// 可写视图类，支持高效批量修改底层允许列表
class RestrictAllowlistMutableView {
 public:
  RestrictAllowlistMutableView() = default;

  explicit RestrictAllowlistMutableView(RestrictAllowlist& allowlist)
      : allowlist_array_(allowlist.allowlist_array_.data()),
        num_points_(allowlist.num_points_) {}

  explicit RestrictAllowlistMutableView(RestrictAllowlist* allowlist)
      : allowlist_array_(allowlist ? allowlist->allowlist_array_.data()
                                   : nullptr),
        num_points_(allowlist ? allowlist->num_points_ : 0) {}

  RestrictAllowlistMutableView(MutableSpan<size_t> storage,
                               DatapointIndex num_points)
      : allowlist_array_(storage.data()), num_points_(num_points) {
    DCHECK_EQ(storage.size(),
              DivRoundUp(num_points, RestrictAllowlist::kBitsPerWord));
  }

  bool IsAllowlisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return GetWordContainingDatapoint(dp_index) &
           (RestrictAllowlist::kOne
            << (dp_index % RestrictAllowlist::kBitsPerWord));
  }

  bool IsAllowlistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsAllowlisted(dp_index);
  }

  // 将指定点加入允许列表
  void AddToAllowlist(DatapointIndex dp_index) {
    DCHECK_LT(dp_index, num_points_);
    allowlist_array_[dp_index / RestrictAllowlist::kBitsPerWord] |=
        RestrictAllowlist::kOne << (dp_index % RestrictAllowlist::kBitsPerWord);
  }

  // 将指定点移出允许列表
  void RemoveFromAllowlist(DatapointIndex dp_index) {
    DCHECK_LT(dp_index, num_points_);
    allowlist_array_[dp_index / RestrictAllowlist::kBitsPerWord] &=
        ~(RestrictAllowlist::kOne
          << (dp_index % RestrictAllowlist::kBitsPerWord));
  }

  // 批量更新允许列表，支持添加/移除/偏移/无效点过滤
  template <bool kIsAddToAllowlist, bool kCheckForInvalidDpIdx,
            bool kWithOffset>
  void MultiUpdateAllowlist(ConstSpan<DatapointIndex> dp_indices,
                            DatapointIndex offset = 0) {
    for (DatapointIndex dp_index : dp_indices) {
      if constexpr (kWithOffset) {
        dp_index -= offset;
      }
      if constexpr (kCheckForInvalidDpIdx) {
        if (ABSL_PREDICT_FALSE(dp_index == kInvalidDatapointIndex)) continue;
      }
      if constexpr (kIsAddToAllowlist) {
        AddToAllowlist(dp_index);
      } else {
        RemoveFromAllowlist(dp_index);
      }
    }
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    return allowlist_array_[dp_index / RestrictAllowlist::kBitsPerWord];
  }

  const size_t* data() const { return allowlist_array_; }
  size_t* data() { return allowlist_array_; }
  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }
  bool empty() const { return !allowlist_array_; }

  operator bool() const { return !empty(); }

 private:
  size_t* allowlist_array_ = nullptr;
  DatapointIndex num_points_ = 0;
  friend class RestrictAllowlist;
};

inline RestrictAllowlistConstView MakeConstView(
    const RestrictAllowlist& allowlist) {
  return RestrictAllowlistConstView(allowlist);
}
inline RestrictAllowlistMutableView MakeMutableView(
    RestrictAllowlist& allowlist) {
  return RestrictAllowlistMutableView(allowlist);
}

// 允许列表回收器，支持底层存储的复用与回收，提升内存利用率
class RestrictAllowlistRecycler {
 public:
  void AddToFreelist(std::vector<size_t>&& v) {
    absl::MutexLock lock(&mutex_);
    freelist_.push(std::move(v));
    VLOG(2) << "Received recyclable at " << freelist_.top().data();
  }

  std::function<void(std::vector<size_t>&&)> AddToFreelistFunctor() {
    return [this](std::vector<size_t>&& v) { AddToFreelist(std::move(v)); };
  }

  std::vector<size_t> MaybeRemoveFromFreelist() {
    absl::MutexLock lock(&mutex_);
    if (freelist_.empty()) return {};
    VLOG(2) << "Available for recycling at " << freelist_.top().data();
    auto result = std::move(freelist_.top());
    freelist_.pop();
    return result;
  }

 private:
  absl::Mutex mutex_;
  std::stack<std::vector<size_t>> freelist_ ABSL_GUARDED_BY(mutex_);
};

// 创建允许列表，支持回收机制
RestrictAllowlist CreateAllowlist(RestrictAllowlistRecycler* recycler,
                                  DatapointIndex num_datapoints,
                                  bool default_allowed);

// 尝试回收允许列表底层存储
inline vector<size_t> MaybeRecycleAllowlist(
    RestrictAllowlistRecycler* recycler) {
  if (!recycler) {
    return {};
  }
  return recycler->MaybeRemoveFromFreelist();
}

// research_scann 命名空间结束
}  // namespace research_scann

#endif
