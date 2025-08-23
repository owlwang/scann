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

#ifndef SCANN_SCANN_OPS_CC_KERNELS_SCANN_OPS_UTILS_H_
#define SCANN_SCANN_OPS_CC_KERNELS_SCANN_OPS_UTILS_H_

#include "absl/types/span.h"
#include "scann/data_format/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"


// ScaNN TensorFlow算子工具函数
namespace tensorflow {
namespace scann_ops {

// 将protobuf序列化内容写入Tensor
absl::Status TensorFromProto(OpKernelContext* context, absl::string_view name,
                             const protobuf::MessageLite* proto);
// 检查状态的TensorFromProto
void TensorFromProtoRequireOk(OpKernelContext* context, absl::string_view name,
                              const protobuf::MessageLite* proto);

// 创建空Tensor
absl::Status EmptyTensor(OpKernelContext* context, absl::string_view name);
void EmptyTensorRequireOk(OpKernelContext* context, absl::string_view name);

// TensorFlow Status与absl::Status转换
Status ConvertStatus(const Status& status);

// 将Tensor转为DenseDataset（支持类型转换）
template <typename DstType, typename SrcType = DstType>
absl::Status PopulateDenseDatasetFromTensor(
    const Tensor& tensor, research_scann::DenseDataset<DstType>* dataset);

// 实现：将二维Tensor转为DenseDataset
template <typename DstType, typename SrcType>
absl::Status PopulateDenseDatasetFromTensor(
    const Tensor& tensor, research_scann::DenseDataset<DstType>* dataset) {
  if (tensor.dims() != 2) {
    return errors::InvalidArgument("Dataset must be 2-dimensional",
                                   tensor.DebugString());
  }
  auto tensor_t = tensor.matrix<SrcType>();
  int num_dim = tensor_t.dimension(1);
  int num_datapoint = tensor_t.dimension(0);

  if (!num_dim) return OkStatus();

  dataset->clear();
  dataset->set_dimensionality(num_dim);
  dataset->Reserve(num_datapoint);

  for (int i = 0; i < num_datapoint; ++i) {
    // 构造DatapointPtr并加入Dataset
    const research_scann::DatapointPtr<DstType> dptr(
        nullptr, reinterpret_cast<const DstType*>(&tensor_t(i, 0)), num_dim,
        num_dim);
    TF_RETURN_IF_ERROR(ConvertStatus(dataset->Append(dptr, "")));
  }
  return OkStatus();
}


// DenseDataset转Tensor
template <typename T>
absl::Status TensorFromDenseDataset(
    OpKernelContext* context, absl::string_view name,
    const research_scann::DenseDataset<T>* dataset) {
  if (dataset == nullptr) return EmptyTensor(context, name);
  Tensor* tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      name,
      TensorShape({static_cast<int64_t>(dataset->size()),
                   static_cast<int64_t>(dataset->dimensionality())}),
      &tensor));
  auto tensor_flat = tensor->flat<T>();
  std::copy(dataset->data().begin(), dataset->data().end(), tensor_flat.data());
  return OkStatus();
}


// 检查状态的DenseDataset转Tensor
template <typename T>
void TensorFromDenseDatasetRequireOk(
    OpKernelContext* context, absl::string_view name,
    const research_scann::DenseDataset<T>* dataset) {
  OP_REQUIRES_OK(context, TensorFromDenseDataset(context, name, dataset));
}


// ConstSpan转Tensor
template <typename T>
absl::Status TensorFromSpan(OpKernelContext* context, absl::string_view name,
                            research_scann::ConstSpan<T> span) {
  if (span.empty()) return EmptyTensor(context, name);
  Tensor* tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      name, TensorShape({static_cast<int64_t>(span.size())}), &tensor));
  auto tensor_flat = tensor->flat<T>();
  std::copy(span.begin(), span.end(), tensor_flat.data());
  return OkStatus();
}


// 检查状态的ConstSpan转Tensor
template <typename T>
void TensorFromSpanRequireOk(OpKernelContext* context, absl::string_view name,
                             research_scann::ConstSpan<T> span) {
  OP_REQUIRES_OK(context, TensorFromSpan(context, name, span));
}


// Tensor转ConstSpan（用于高效访问Tensor数据）
template <typename T>
research_scann::ConstSpan<T> TensorToConstSpan(const Tensor* t) {
  return absl::MakeConstSpan(t->flat<T>().data(), t->NumElements());
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
