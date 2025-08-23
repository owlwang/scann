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

#ifndef SCANN_DISTANCE_MEASURES_DISTANCE_MEASURE_BASE_H_
#define SCANN_DISTANCE_MEASURES_DISTANCE_MEASURE_BASE_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/types.h"


// 距离度量相关代码均在 research_scann 命名空间下
namespace research_scann {

// 距离度量基类，定义了所有距离计算相关的虚方法接口
class DistanceMeasure : public VirtualDestructor {
 public:
  // 返回距离度量的名称（如 L2、Cosine 等），需由子类实现
  virtual string_view name() const = 0;

  // 返回该距离度量是否需要归一化，默认不需要
  virtual Normalization NormalizationRequired() const { return NONE; }

  // 标记特殊优化的距离类型，便于后续针对常见距离做高效实现
  enum SpeciallyOptimizedDistanceTag {
    L1,
    L2,
    SQUARED_L2,
    COSINE,
    DOT_PRODUCT,
    ABS_DOT_PRODUCT,
    LIMITED_INNER_PRODUCT,
    GENERAL_HAMMING,
    NEGATED_SQUARED_L2,
    NOT_SPECIALLY_OPTIMIZED
  };

  // 返回该距离度量的特殊优化标签，默认无特殊优化
  virtual SpeciallyOptimizedDistanceTag specially_optimized_distance_tag()
      const {
    return NOT_SPECIALLY_OPTIMIZED;
  }

  // 通用距离计算入口，根据数据点稠密/稀疏类型自动分派到对应实现
  template <typename T>
  double GetDistance(const DatapointPtr<T>& a, const DatapointPtr<T>& b) const {
    const bool a_is_dense = a.IsDense();
    const bool b_is_dense = b.IsDense();
    const int n_dense = a_is_dense + b_is_dense;
    if (n_dense == 0) {
      // 两个数据点均为稀疏，调用稀疏距离计算
      return GetDistanceSparse(a, b);
    } else if (n_dense == 1) {
      // 一个稠密一个稀疏，调用混合距离计算
      return GetDistanceHybrid(a, b);
    } else {
      // 两个均为稠密，调用稠密距离计算
      DCHECK_EQ(n_dense, 2);
      return GetDistanceDense(a, b);
    }
  }

  // 带阈值的距离计算入口，部分实现可利用阈值做提前终止优化
  template <typename T>
  double GetDistance(const DatapointPtr<T>& a, const DatapointPtr<T>& b,
                     double threshold) const {
    const bool a_is_dense = a.IsDense();
    const bool b_is_dense = b.IsDense();
    const int n_dense = a_is_dense + b_is_dense;
    if (n_dense == 0) {
      // 两个数据点均为稀疏
      return GetDistanceSparse(a, b);
    } else if (n_dense == 1) {
      // 一个稠密一个稀疏
      return GetDistanceHybrid(a, b);
    } else {
      // 两个均为稠密，带阈值
      DCHECK_EQ(n_dense, 2);
      return GetDistanceDense(a, b, threshold);
    }
  }

// 声明距离度量的所有虚方法接口，支持多种数据类型（int/float/double等）
#define SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(T)                     \
  /* 稠密数据点距离计算 */                                                    \
  virtual double GetDistanceDense(const DatapointPtr<T>& a,                   \
                                  const DatapointPtr<T>& b) const = 0;        \
  /* 稠密数据点距离计算（带阈值）*/                                            \
  virtual double GetDistanceDense(const DatapointPtr<T>& a,                   \
                                  const DatapointPtr<T>& b, double threshold) \
      const = 0;                                                              \
  /* 稀疏数据点距离计算 */                                                    \
  virtual double GetDistanceSparse(const DatapointPtr<T>& a,                  \
                                   const DatapointPtr<T>& b) const = 0;       \
  /* 混合（稠密+稀疏）数据点距离计算 */                                      \
  virtual double GetDistanceHybrid(const DatapointPtr<T>& a,                  \
                                   const DatapointPtr<T>& b) const = 0;

  // 支持所有主流数值类型的数据点距离计算
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int8_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint8_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int16_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint16_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int32_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint32_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(int64_t);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(uint64_t);

  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(float);
  SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS(double);

#undef SCANN_DECLARE_DISTANCE_MEASURE_VIRTUAL_METHODS

 private:
  // 预留的私有虚方法，防止类无虚方法导致的编译器优化问题
  virtual void UnusedKeyMethod();
};

// research_scann 命名空间结束
}  // namespace research_scann

#endif
