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



#ifndef SCANN_DATA_FORMAT_INTERNAL_SHORT_STRING_OPTIMIZED_STRING_H_
#define SCANN_DATA_FORMAT_INTERNAL_SHORT_STRING_OPTIMIZED_STRING_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

#include "absl/base/prefetch.h"
#include "absl/types/optional.h"
#include "scann/oss_wrappers/scann_malloc_extension.h"
#include "scann/utils/common.h"

namespace research_scann {

// 短字符串优化存储类，针对长度较短的字符串采用内联存储，超过阈值则堆分配，兼顾性能与空间效率
class ShortStringOptimizedString {
 public:
  // 默认构造函数，初始化存储空间为零
  ShortStringOptimizedString() { memset(storage_, 0, kStorageSize); }

  // 从 string_view 构造，自动选择内联或堆分配
  explicit ShortStringOptimizedString(string_view orig) {
    ConstructFromStringPiece(orig);
  }

  // 拷贝构造与赋值，保证语义正确（深拷贝）
  ShortStringOptimizedString(const ShortStringOptimizedString& rhs)
      : ShortStringOptimizedString(rhs.ToStringPiece()) {}
  ShortStringOptimizedString& operator=(const ShortStringOptimizedString& rhs) {
    this->~ShortStringOptimizedString();
    ConstructFromStringPiece(rhs.ToStringPiece());
    return *this;
  }

  // 支持直接用 string_view 赋值
  ShortStringOptimizedString& operator=(const string_view rhs) {
    this->~ShortStringOptimizedString();
    ConstructFromStringPiece(rhs);
    return *this;
  }

  // 移动构造，直接拷贝存储并清空源对象
  ShortStringOptimizedString(ShortStringOptimizedString&& rhs) noexcept {
    memcpy(storage_, rhs.storage_, kStorageSize);
    rhs.ClearNoFree();
  }

  // 移动赋值，释放自身后拷贝存储并清空源对象
  ShortStringOptimizedString& operator=(
      ShortStringOptimizedString&& rhs) noexcept {
    this->~ShortStringOptimizedString();
    memcpy(storage_, rhs.storage_, kStorageSize);
    rhs.ClearNoFree();
    return *this;
  }

  // 获取字符串数据指针，自动区分内联和堆分配
  const char* data() const {
    return (size() <= kMaxInline) ? storage_ : heap_string();
  }

  // 对堆分配字符串进行预取，提升访问性能
  void prefetch() const {
    if (size() > kMaxInline) {
      absl::PrefetchToLocalCache(heap_string());
    }
  }

  // 获取字符串长度，统一从存储区尾部读取
  uint32_t size() const {
    return *reinterpret_cast<const uint32_t*>(storage_ + kStorageSize -
                                              sizeof(uint32_t));
  }

  // 判断字符串是否为空
  bool empty() const { return !size(); }

  // 转换为 string_view，便于高效访问
  string_view ToStringPiece() const { return string_view(data(), size()); }

  // 支持隐式转换为 std::string
  operator std::string() const { return std::string(ToStringPiece()); }

  // 返回堆分配字符串实际占用内存（仅大于阈值时有效）
  size_t HeapStorageUsed() const {
    if (size() <= kMaxInline) return 0;
    std::optional<size_t> true_size =
        tcmalloc::MallocExtension::GetAllocatedSize(heap_string());
    return *true_size;
  }

  // 支持与 string_view 比较
  bool operator==(string_view s) const { return ToStringPiece() == s; }

  // 析构函数，释放堆分配内存并清零存储区
  ~ShortStringOptimizedString() {
    if (size() > kMaxInline) {
      delete[] heap_string();
      memset(storage_, 0, kStorageSize);
    }
  }

 private:
  // 编译期断言，保证 uint32_t 类型大小正确
  static_assert(sizeof(uint32_t) == 4, "The uint32 typedef is wrong.");

  // 编译期断言，仅支持 32/64 位指针模型
  static_assert(sizeof(char*) == 4 || sizeof(char*) == 8,
                "ScaNN only supports 32- and 64-bit flat memory models.");

  // 存储区大小，依赖于指针宽度
  static constexpr size_t kStorageSize = (sizeof(char*) == 4) ? 8 : 16;

  // 最大内联存储长度，超出则堆分配
  static constexpr size_t kMaxInline = kStorageSize - sizeof(uint32_t);

  // 根据输入字符串构造对象，自动选择内联或堆分配
  void ConstructFromStringPiece(string_view orig) {
    set_size(orig.size());
    if (orig.size() > kMaxInline) {
      char* heap_string = new char[orig.size()];
      memcpy(heap_string, orig.data(), orig.size());
      set_heap_string(heap_string);
    } else {
      memcpy(storage_, orig.data(), orig.size());
    }
  }

  // 清空存储区但不释放堆内存（用于移动语义）
  void ClearNoFree() { memset(storage_, 0, kStorageSize); }

  // 获取堆分配字符串指针
  const char* heap_string() const {
    return *reinterpret_cast<const char* const*>(storage_);
  }

  // 设置堆分配字符串指针
  void set_heap_string(char* s) { *reinterpret_cast<char**>(storage_) = s; }

  // 设置字符串长度到存储区尾部
  void set_size(uint32_t s) {
    *(reinterpret_cast<uint32_t*>(storage_ + kStorageSize - sizeof(uint32_t))) =
        s;
  }

  // 联合体用于存储字符串数据和对齐辅助
  union {
    char storage_[kStorageSize];

    pair<char*, uint32_t> for_alignment_only_;

    static_assert(sizeof(for_alignment_only_) == kStorageSize, "");
  };
};

// 编译期断言，保证类大小与 string_view 一致，便于高效替换
static_assert(sizeof(ShortStringOptimizedString) == sizeof(string_view), "");

}  // namespace research_scann

#endif
