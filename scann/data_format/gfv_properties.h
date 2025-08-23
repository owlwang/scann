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



#ifndef SCANN_DATA_FORMAT_GFV_PROPERTIES_H_
#define SCANN_DATA_FORMAT_GFV_PROPERTIES_H_

#include <string>

#include "scann/data_format/features.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// 获取类型 T 对应的 GFV 特征类型枚举值
template <typename T>
inline int GfvFeatureType() {
  return GenericFeatureVector::INT64;
}
template <>
inline int GfvFeatureType<float>() {
  return GenericFeatureVector::FLOAT;
}
template <>
inline int GfvFeatureType<double>() {
  return GenericFeatureVector::DOUBLE;
}
template <>
inline int GfvFeatureType<std::string>() {
  return GenericFeatureVector::STRING;
}

// 获取 GFV 特征类型名称（如 INT64、FLOAT、DOUBLE、STRING）
string_view GfvFeatureTypeName(int gfv_feature_type);

// 获取 GFV 向量实际存储长度（不同类型字段）
StatusOr<size_t> GetGfvVectorSize(const GenericFeatureVector& gfv);

// 获取 GFV 维度（稀疏/稠密自动判定）
StatusOr<DimensionIndex> GetGfvDimensionality(const GenericFeatureVector& gfv);

// 判定 GFV 是否为稀疏类型
StatusOr<bool> IsGfvSparse(const GenericFeatureVector& gfv);
// 判定 GFV 是否为稠密类型
StatusOr<bool> IsGfvDense(const GenericFeatureVector& gfv);

// 判定 GFV 是否为非二值数值类型
inline bool IsNonBinaryNumeric(const GenericFeatureVector& gfv) {
  return gfv.feature_type() == GenericFeatureVector::INT64 ||
         gfv.feature_type() == GenericFeatureVector::FLOAT ||
         gfv.feature_type() == GenericFeatureVector::DOUBLE;
}

// 获取 GFV 向量长度（辅助接口，带输出参数）
Status GetGfvVectorSize(const GenericFeatureVector& gfv,
                        DimensionIndex* result);

// 获取 GFV 维度（辅助接口，带输出参数）
Status GetGfvDimensionality(const GenericFeatureVector& gfv,
                            DimensionIndex* result);

// 判定 GFV 是否稀疏（辅助接口，带输出参数）
Status IsGfvSparse(const GenericFeatureVector& gfv, bool* result);

// 判定 GFV 是否稠密（辅助接口，带输出参数）
Status IsGfvDense(const GenericFeatureVector& gfv, bool* result);

// 获取 GFV 维度（失败直接崩溃）
size_t GetGfvDimensionalityOrDie(const GenericFeatureVector& gfv);

// 判定 GFV 是否稀疏（失败直接崩溃）
bool IsGfvSparseOrDie(const GenericFeatureVector& gfv);

// 判定 GFV 是否稠密（失败直接崩溃）
bool IsGfvDenseOrDie(const GenericFeatureVector& gfv);

}  // namespace research_scann

#endif
