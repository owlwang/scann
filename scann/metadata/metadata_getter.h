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



#ifndef SCANN_METADATA_METADATA_GETTER_H_
#define SCANN_METADATA_METADATA_GETTER_H_

#include <optional>
#include <string>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/features.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// 模板声明：元数据获取器，按类型特化
template <typename T>
class MetadataGetter;

// 未特化类型的元数据获取器基类，定义通用接口
class UntypedMetadataGetter {
 public:
  // 添加元数据到集合
  virtual Status AppendMetadata(const GenericFeatureVector& gfv);

  // 更新指定索引的数据点元数据
  virtual Status UpdateMetadata(DatapointIndex idx,
                                const GenericFeatureVector& gfv);

  // 移除指定索引的数据点元数据
  virtual Status RemoveMetadata(DatapointIndex removed_idx);

  // 是否需要数据集支持
  virtual bool needs_dataset() const;

  // 返回类型标签（需子类实现）
  virtual research_scann::TypeTag TypeTag() const = 0;

  // 虚析构，保证多态删除
  virtual ~UntypedMetadataGetter();
};

// 类型特化的元数据获取器，支持具体数据类型T
template <typename T>
class MetadataGetter : public UntypedMetadataGetter {
 public:
  // 默认构造
  MetadataGetter() = default;

  // 禁止拷贝构造和赋值
  MetadataGetter(const MetadataGetter&) = delete;
  MetadataGetter& operator=(const MetadataGetter&) = delete;

  // 返回类型标签，特化为T
  research_scann::TypeTag TypeTag() const final { return TagForType<T>(); }

  // 返回固定长度元数据size（默认无实现）
  virtual std::optional<size_t> fixed_len_size(
      const TypedDataset<T>* dataset, const DatapointPtr<T>& query) const {
    return std::nullopt;
  }

  // 获取指定邻居的数据点元数据（需子类实现）
  virtual Status GetMetadata(const TypedDataset<T>* dataset,
                             const DatapointPtr<T>& query,
                             DatapointIndex neighbor_index,
                             std::string* result) const = 0;

  // 批量获取邻居元数据，结果通过回调设置
  virtual Status GetMetadatas(const TypedDataset<T>* dataset,
                              const DatapointPtr<T>& query,
                              size_t num_neighbors,
                              DpIdxGetter neighbor_dp_idx_getter,
                              StringSetter metadata_setter) const {
    for (size_t i : Seq(num_neighbors)) {
      std::string result;
      SCANN_RETURN_IF_ERROR(
          GetMetadata(dataset, query, neighbor_dp_idx_getter(i), &result));
      metadata_setter(i, result);
    }
    return OkStatus();
  }

  // 批量获取并转换元数据，结果通过回调输出
  virtual Status TransformAndCopyMetadatas(
      const TypedDataset<T>* dataset, const DatapointPtr<T>& query,
      size_t num_neighbors, DpIdxGetter neighbor_dp_idx_getter,
      OutputStringGetter output_string_getter) const {
    for (size_t i : Seq(num_neighbors)) {
      SCANN_RETURN_IF_ERROR(GetMetadata(
          dataset, query, neighbor_dp_idx_getter(i), output_string_getter(i)));
    }
    return OkStatus();
  }

  // 按数据点索引获取元数据（默认未实现）
  virtual StatusOr<std::string> GetByDatapointIndex(
      const TypedDataset<T>* dataset, DatapointIndex dp_idx) const {
    return UnimplementedError(
        StrCat("Cannot get metadata by datapoint index for "
               "metadata getter type ",
               typeid(*this).name(), "."));
  }
};

}  // namespace research_scann

#endif
