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



#include "scann/metadata/metadata_getter.h"

#include "scann/data_format/features.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// 追加元数据，默认实现为空操作
Status UntypedMetadataGetter::AppendMetadata(const GenericFeatureVector& gfv) {
  return OkStatus();
}

// 是否需要数据集，默认返回 true
bool UntypedMetadataGetter::needs_dataset() const { return true; }

// 更新元数据，默认未实现
Status UntypedMetadataGetter::UpdateMetadata(DatapointIndex idx,
                                             const GenericFeatureVector& gfv) {
  return UnimplementedError("UpdateMetadata not implemented by default.");
}

// 移除元数据，默认未实现
Status UntypedMetadataGetter::RemoveMetadata(DatapointIndex removed_idx) {
  return UnimplementedError("UpdateMetadata not implemented by default.");
}

// 析构函数，默认实现
UntypedMetadataGetter::~UntypedMetadataGetter() {}

// research_scann 命名空间结束
}  // namespace research_scann
