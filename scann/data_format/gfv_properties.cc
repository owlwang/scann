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

#include "scann/data_format/gfv_properties.h"

#include "absl/log/check.h"
#include "scann/data_format/features.pb.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// 获取 GFV 特征类型名称（如 INT64、FLOAT、DOUBLE、STRING）
string_view GfvFeatureTypeName(int gfv_feature_type) {
  switch (gfv_feature_type) {
    case GenericFeatureVector::INT64:
      return "INT64";
    case GenericFeatureVector::FLOAT:
      return "FLOAT";
    case GenericFeatureVector::DOUBLE:
      return "DOUBLE";
    case GenericFeatureVector::STRING:
      return "STRING";
    default:
      return "INVALID_GFV_FEATURE_TYPE";
  }
}

// 获取 GFV 向量实际存储长度（不同类型字段）
StatusOr<size_t> GetGfvVectorSize(const GenericFeatureVector& gfv) {
  switch (gfv.feature_type()) {
    case GenericFeatureVector::INT64:
    case GenericFeatureVector::BINARY:
      return gfv.feature_value_int64_size();
    case GenericFeatureVector::FLOAT:
      return gfv.feature_value_float_size();
    case GenericFeatureVector::DOUBLE:
      return gfv.feature_value_double_size();
    case GenericFeatureVector::STRING:
      return 1;
    default:
      return InvalidArgumentError("Unknown feature type:  %d",
                                  gfv.feature_type());
  }
}

// 获取 GFV 维度（稀疏/稠密自动判定）
StatusOr<DimensionIndex> GetGfvDimensionality(const GenericFeatureVector& gfv) {
  if (gfv.feature_dim() == 0) {
    return InvalidArgumentError(
        "GenericFeatureVector dimensionality cannot be == 0.");
  }

  SCANN_ASSIGN_OR_RETURN(bool is_sparse, IsGfvSparse(gfv));
  if (is_sparse) {
    return gfv.feature_dim();
  } else {
    return GetGfvVectorSize(gfv);
  }
}

// 判定 GFV 是否为稀疏类型
StatusOr<bool> IsGfvSparse(const GenericFeatureVector& gfv) {
  if (gfv.feature_type() == GenericFeatureVector::STRING) {
    return false;
  }

  if (gfv.feature_index_size() > 0) {
    return true;
  }

  SCANN_ASSIGN_OR_RETURN(DimensionIndex vector_size, GetGfvVectorSize(gfv));
  return vector_size == 0;
}

// 判定 GFV 是否为稠密类型
StatusOr<bool> IsGfvDense(const GenericFeatureVector& gfv) {
  if (gfv.feature_type() == GenericFeatureVector::STRING) {
    return false;
  }

  SCANN_ASSIGN_OR_RETURN(bool is_sparse, IsGfvSparse(gfv));
  return !is_sparse;
}

// 获取 GFV 向量长度（辅助接口，带输出参数）
Status GetGfvVectorSize(const GenericFeatureVector& gfv,
                        DimensionIndex* result) {
  DCHECK(result);
  SCANN_ASSIGN_OR_RETURN(*result, GetGfvVectorSize(gfv));
  return OkStatus();
}

// 获取 GFV 维度（辅助接口，带输出参数）
Status GetGfvDimensionality(const GenericFeatureVector& gfv,
                            DimensionIndex* result) {
  DCHECK(result);
  SCANN_ASSIGN_OR_RETURN(*result, GetGfvDimensionality(gfv));
  return OkStatus();
}

// 判定 GFV 是否稀疏（辅助接口，带输出参数）
Status IsGfvSparse(const GenericFeatureVector& gfv, bool* result) {
  DCHECK(result);
  SCANN_ASSIGN_OR_RETURN(*result, IsGfvSparse(gfv));
  return OkStatus();
}

// 判定 GFV 是否稠密（辅助接口，带输出参数）
Status IsGfvDense(const GenericFeatureVector& gfv, bool* result) {
  DCHECK(result);
  SCANN_ASSIGN_OR_RETURN(*result, IsGfvDense(gfv));
  return OkStatus();
}

// 获取 GFV 维度（失败直接崩溃）
size_t GetGfvDimensionalityOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(GetGfvDimensionality(gfv));
}

// 判定 GFV 是否稀疏（失败直接崩溃）
bool IsGfvSparseOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(IsGfvSparse(gfv));
}

// 判定 GFV 是否稠密（失败直接崩溃）
bool IsGfvDenseOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(IsGfvDense(gfv));
}

}  // namespace research_scann
