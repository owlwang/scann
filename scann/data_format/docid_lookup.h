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

#ifndef SCANN_DATA_FORMAT_DOCID_LOOKUP_H_
#define SCANN_DATA_FORMAT_DOCID_LOOKUP_H_

#include <cstddef>

#include "absl/functional/any_invocable.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// DocidLookup：docid 查找抽象接口，支持单个和批量查找
class DocidLookup {
 public:
  virtual ~DocidLookup() = default;

  // 查找 docid 对应的数据点索引，返回是否找到
  virtual bool LookupDatapointIndex(string_view docid,
                                    DatapointIndex* idx) const = 0;

  // DocidGetter：批量查找时获取 docid 的函数类型
  using DocidGetter = absl::AnyInvocable<string_view(size_t) const>;

  // LookupCallback：批量查找时回调，返回 docid 索引和数据点索引
  using LookupCallback =
      absl::AnyInvocable<void(size_t docids_idx, DatapointIndex dp_idx)>;

  // 批量查找 docid 索引，支持自定义 getter 和回调
  virtual void LookupDatapointIndices(size_t num_docids,
                                      DocidGetter docid_getter,
                                      LookupCallback callback) const {
    for (size_t i = 0; i < num_docids; ++i) {
      DatapointIndex dp_idx;
      callback(i, LookupDatapointIndex(docid_getter(i), &dp_idx)
                      ? dp_idx
                      : kInvalidDatapointIndex);
    }
  }

  // 返回实现名称
  virtual string_view ImplName() const = 0;
};

}  // namespace research_scann

#endif
