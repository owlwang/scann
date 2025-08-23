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

#ifndef SCANN_DATA_FORMAT_DOCID_COLLECTION_INTERFACE_H_
#define SCANN_DATA_FORMAT_DOCID_COLLECTION_INTERFACE_H_

#include <cstddef>
#include <optional>

#include "absl/base/nullability.h"
#include "scann/data_format/docid_lookup.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// DocidCollectionInterface：docid 集合抽象接口，支持追加、查找、删除、内存管理等
class DocidCollectionInterface {
 public:
  virtual ~DocidCollectionInterface() = default;

  // 追加一个 docid 到集合
  virtual Status Append(string_view docid) = 0;

  // 集合中 docid 数量
  virtual size_t size() const = 0;

  // 判断集合是否为空
  virtual bool empty() const = 0;

  // 固定长度集合的 size（默认无）
  virtual std::optional<size_t> fixed_len_size() const { return std::nullopt; }

  // 获取指定索引的 docid
  virtual string_view Get(size_t i) const = 0;

  // 批量获取 docid，支持自定义 getter/setter
  virtual void MultiGet(size_t num_docids, DpIdxGetter docid_idx_getter,
                        StringSetter docid_setter) const = 0;

  // 集合容量
  virtual size_t capacity() const = 0;

  // 集合占用内存
  virtual size_t MemoryUsage() const = 0;

  // 清空集合
  virtual void Clear() = 0;

  // 预分配集合空间
  virtual void Reserve(DatapointIndex n_elements) = 0;

  // 收缩集合内存到实际大小
  virtual void ShrinkToFit() = 0;

  // 复制集合
  virtual unique_ptr<DocidCollectionInterface> Copy() const = 0;

  // Mutator：docid 集合可变操作接口
  class Mutator : public DocidLookup {
   public:
    ~Mutator() override = default;

  // 追加 docid
  virtual Status AddDatapoint(string_view docid) = 0;
  // 移除指定 docid
  virtual Status RemoveDatapoint(string_view docid) = 0;
  // 预分配空间
  virtual void Reserve(size_t size) = 0;
  // 按索引移除 docid
  virtual Status RemoveDatapoint(DatapointIndex idx) = 0;
  // 返回实现名称
  string_view ImplName() const override = 0;
  };

  // 获取 docid 查找器（默认返回 Mutator）
  virtual StatusOr<DocidLookup*> GetDocidLookup() const {
    auto mutator = GetMutator();
    if (!mutator.ok()) {
      return mutator.status();
    }
    return mutator.value();
  }

  // 获取 Mutator（抽象接口）
  virtual StatusOr<Mutator*> GetMutator() const = 0;
};

// DocidLookupMap：docid 查找 map 抽象，支持添加/删除/查找/内存管理
class DocidLookupMap : public DocidLookup {
 public:
  ~DocidLookupMap() override = default;

  // 清空 map
  virtual void Clear() = 0;
  // 预分配空间
  virtual void Reserve(size_t size) = 0;
  // 添加 docid 与索引映射
  virtual Status AddDatapoint(string_view docid, DatapointIndex dp_idx) = 0;
  // 移除指定 docid
  virtual Status RemoveDatapoint(string_view docid) = 0;
  // 按索引移除 docid
  virtual Status RemoveDatapoint(DatapointIndex dp_idx) = 0;
  // 返回实现名称
  string_view ImplName() const override = 0;

protected:
  // 构造函数，绑定 docid 集合指针
  explicit DocidLookupMap(const DocidCollectionInterface* docids)
      : docids_(docids) {}
  // docid 集合指针
  const DocidCollectionInterface* docids_ = nullptr;
};

// 构建 docid 查找 map
absl::StatusOr<std::unique_ptr<DocidLookupMap>> CreateDocidLookupMap(
  DocidCollectionInterface* docids);

}  // namespace research_scann

#endif
