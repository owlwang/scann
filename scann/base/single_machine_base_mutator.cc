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

#include <cmath>
#include <utility>

#include "absl/base/optimization.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

namespace {

template <typename T>
bool SameDocidsInstance(
    const shared_ptr<const DocidCollectionInterface>& docids,
    const TypedDataset<T>* dataset) {
  if (!dataset) return false;
  return docids == dataset->docids();
}

}  // namespace

// 变异器准备，获取各数据结构的 mutator
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::PrepareForBaseMutation(
    SingleMachineSearcherBase<T>* searcher) {
  searcher_ = searcher;
  searcher_->mutator_outstanding_ = true;
  // 获取原始数据集 mutator
  if (searcher->dataset_) {
    SCANN_ASSIGN_OR_RETURN(dataset_mutator_, searcher->dataset_->GetMutator());
  }
  // 获取哈希数据集 mutator
  if (searcher->hashed_dataset_) {
    SCANN_ASSIGN_OR_RETURN(hashed_dataset_mutator_,
                           searcher->hashed_dataset_->GetMutator());
  }
  // 获取重排序辅助 mutator
  if (searcher_->reordering_helper_ &&
      searcher_->reordering_helper_->owns_mutation_data_structures()) {
    SCANN_ASSIGN_OR_RETURN(reordering_mutator_,
                           searcher->reordering_helper_->GetMutator());
  }
  // 获取 docid mutator（需排除与数据集绑定的情况）
  if (searcher->docids_ &&
      !SameDocidsInstance(searcher->docids_, searcher->dataset_.get()) &&
      !SameDocidsInstance(searcher->docids_, searcher->hashed_dataset_.get())) {
    SCANN_ASSIGN_OR_RETURN(docid_mutator_, searcher->docids_->GetMutator());
  }
  return OkStatus();
}

// 获取下一个可用的数据点索引，确保所有相关结构大小一致
template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::GetNextDatapointIndex() const {
  DatapointIndex result = kInvalidDatapointIndex;
  if (searcher_->dataset_) {
    result = searcher_->dataset_->size();

    // 校验 docids 和哈希数据集大小一致
    if (searcher_->docids_)
      SCANN_RET_CHECK_EQ(result, searcher_->docids_->size());
    if (searcher_->hashed_dataset_) {
      SCANN_RET_CHECK_EQ(result, searcher_->hashed_dataset_->size());
    }
  } else if (searcher_->hashed_dataset_) {
    result = searcher_->hashed_dataset()->size();
    if (searcher_->docids_)
      SCANN_RET_CHECK_EQ(result, searcher_->docids_->size());
  } else if (searcher_->docids_) {
    result = searcher_->docids_->size();
  }
  return result;
}

// 校验数据点是否有效（无 NaN/Inf）
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::ValidateForUpdateOrAdd(
    const DatapointPtr<T>& dptr, string_view docid,
    const MutationOptions& mo) const {
  if constexpr (std::is_floating_point_v<T>) {
    auto vs = dptr.values_span();
    for (size_t i : IndicesOf(vs)) {
      // 检查每个值是否为有限数
      if (!ABSL_PREDICT_TRUE(std::isfinite(vs[i]))) {
        return InvalidArgumentError(absl::StrCat(
            "NaN or infinity found in ScaNN update.   value = ", vs[i],
            " dim idx = ", (dptr.indices()) ? dptr.indices()[i] : i,
            " Docid = ", docid));
      }
    }
  }

  return OkStatus();
}

// 校验更新操作的索引和数据点有效性
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::ValidateForUpdate(
    const DatapointPtr<T>& dptr, DatapointIndex idx,
    const MutationOptions& mo) const {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex next_idx, GetNextDatapointIndex());
  // 检查索引是否越界
  if (idx >= next_idx) {
    return InvalidArgumentError(absl::StrCat(
        "Datapoint index ", idx,
        " is out of range for update.  This index's size is ", next_idx, "."));
  }

  StatusOr<string_view> docid = searcher_->GetDocid(idx);

  // 校验数据点内容
  return ValidateForUpdateOrAdd(dptr, docid.ok() ? *docid : "<UNKNOWN DOCID>",
                                mo);
}

// 校验新增操作的 docid 唯一性和数据点有效性
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::ValidateForAdd(
    const DatapointPtr<T>& dptr, string_view docid,
    const MutationOptions& mo) const {
  DatapointIndex dp_idx = kInvalidDatapointIndex;
  // 检查 docid 是否已存在
  if (LookupDatapointIndex(docid, &dp_idx)) {
    return FailedPreconditionError(
        absl::StrCat("Cannot add docid that already exists: ", docid));
  }

  SCANN_RETURN_IF_ERROR(GetNextDatapointIndex().status());
  // 校验数据点内容
  return ValidateForUpdateOrAdd(dptr, docid, mo);
}

// 校验移除操作的索引有效性
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::ValidateForRemove(
    DatapointIndex idx) const {
  SCANN_ASSIGN_OR_RETURN(DatapointIndex next_idx, GetNextDatapointIndex());
  // 检查索引是否越界
  if (idx >= next_idx) {
    return InvalidArgumentError(absl::StrCat(
        "Datapoint index ", idx,
        " is out of range for removal.  This index's size is ", next_idx, "."));
  }
  return OkStatus();
}

// 校验新增操作的哈希数据点选项
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::CheckAddDatapointToBaseOptions(
    const MutateBaseOptions& opts) const {
  if (hashed_dataset_mutator_ && !opts.hashed) {
    return InternalError(
        "Hashed datapoint must be specified in MutateBaseOptions if a hashed "
        "dataset exists in the searcher.");
  }
  return OkStatus();
}

// 获取基础数据集中的数据点
template <typename T>
absl::StatusOr<Datapoint<T>>
SingleMachineSearcherBase<T>::Mutator::GetDatapointFromBase(
    DatapointIndex i) const {
  if (dataset_mutator_) {
    return dataset_mutator_->GetDatapoint(i);
  }
  // 哈希数据集暂不支持直接获取数据点
  if (hashed_dataset_mutator_) {
    return UnimplementedError(
        "GetDatapointFromBase not implemented for hashed dataset.");
  }
  return UnimplementedError("GetDatapointFromBase not implemented.");
}

// 新增数据点到基础数据结构
template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::AddDatapointToBase(
    const DatapointPtr<T>& dptr, string_view docid,
    const MutateBaseOptions& opts) {
  SCANN_RETURN_IF_ERROR(CheckAddDatapointToBaseOptions(opts));
  SCANN_ASSIGN_OR_RETURN(const DatapointIndex result, GetNextDatapointIndex());
  // 原始数据集新增
  if (dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->AddDatapoint(dptr, docid));
  }
  // 哈希数据集新增
  if (hashed_dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(
        hashed_dataset_mutator_->AddDatapoint(*opts.hashed, docid));
  }
  // docid 集合新增
  if (docid_mutator_) {
    SCANN_RETURN_IF_ERROR(docid_mutator_->AddDatapoint(docid));
  }
  // 重排序辅助新增
  if (reordering_mutator_) {
    SCANN_ASSIGN_OR_RETURN(auto idx, reordering_mutator_->AddDatapoint(dptr));
    SCANN_RET_CHECK_EQ(result, idx);
  }
  return result;
}

// 更新基础数据结构中的数据点
template <typename T>
Status SingleMachineSearcherBase<T>::Mutator::UpdateDatapointInBase(
    const DatapointPtr<T>& dptr, DatapointIndex idx,
    const MutateBaseOptions& opts) {
  SCANN_RETURN_IF_ERROR(CheckAddDatapointToBaseOptions(opts));
  const bool mutate_values_vector = true;
  // 原始数据集更新
  if (dataset_mutator_ && mutate_values_vector) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->UpdateDatapoint(dptr, idx));
  }
  // 哈希数据集更新
  if (hashed_dataset_mutator_ && mutate_values_vector) {
    SCANN_RETURN_IF_ERROR(
        hashed_dataset_mutator_->UpdateDatapoint(*opts.hashed, idx));
  }
  // 重排序辅助更新
  if (reordering_mutator_ && mutate_values_vector) {
    SCANN_RETURN_IF_ERROR(reordering_mutator_->UpdateDatapoint(dptr, idx));
  }
  return OkStatus();
}

// 移除基础数据结构中的数据点
template <typename T>
StatusOr<DatapointIndex>
SingleMachineSearcherBase<T>::Mutator::RemoveDatapointFromBase(
    DatapointIndex idx) {
  SCANN_RETURN_IF_ERROR(GetNextDatapointIndex().status());

  DatapointIndex result = kInvalidDatapointIndex;
  // 原始数据集移除
  if (dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(dataset_mutator_->RemoveDatapoint(idx));
    result = searcher_->dataset_->size();
  }
  // 哈希数据集移除
  if (hashed_dataset_mutator_) {
    SCANN_RETURN_IF_ERROR(hashed_dataset_mutator_->RemoveDatapoint(idx));
    result = searcher_->hashed_dataset_->size();
  }
  // docid 集合移除
  if (docid_mutator_) {
    SCANN_RETURN_IF_ERROR(docid_mutator_->RemoveDatapoint(idx));
    result = searcher_->docids_->size();
  }
  // 重排序辅助移除
  if (reordering_mutator_) {
    SCANN_ASSIGN_OR_RETURN(auto swapped_from,
                           reordering_mutator_->RemoveDatapoint(idx));
    if (result != kInvalidDatapointIndex) {
      SCANN_RET_CHECK_EQ(swapped_from, result);
    }
  }
  return result;
}

// 预留各基础数据结构空间
template <typename T>
void SingleMachineSearcherBase<T>::Mutator::ReserveInBase(
    DatapointIndex num_datapoints) {
  if (dataset_mutator_) dataset_mutator_->Reserve(num_datapoints);
  if (hashed_dataset_mutator_) hashed_dataset_mutator_->Reserve(num_datapoints);
  if (reordering_mutator_) reordering_mutator_->Reserve(num_datapoints);
  if (docid_mutator_) docid_mutator_->Reserve(num_datapoints);
}

// 查找 docid 对应的数据点索引
template <typename T>
bool SingleMachineSearcherBase<T>::Mutator::LookupDatapointIndex(
    string_view docid, DatapointIndex* index) const {
  if (dataset_mutator_) {
    return dataset_mutator_->LookupDatapointIndex(docid, index);
  }
  if (hashed_dataset_mutator_) {
    return hashed_dataset_mutator_->LookupDatapointIndex(docid, index);
  }
  if (docid_mutator_) return docid_mutator_->LookupDatapointIndex(docid, index);
  return false;
}

SCANN_INSTANTIATE_TYPED_CLASS(, SingleMachineSearcherBase);

}  // namespace research_scann
