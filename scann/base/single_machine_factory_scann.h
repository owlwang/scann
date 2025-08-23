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



#ifndef SCANN_BASE_SINGLE_MACHINE_FACTORY_SCANN_H_
#define SCANN_BASE_SINGLE_MACHINE_FACTORY_SCANN_H_

#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/utils/factory_helpers.h"

namespace research_scann {


// 数据集类型模板声明
template <typename T>
class TypedDataset;
// ScaNN 配置类声明
class ScannConfig;

// 类型安全的搜索器返回类型
template <typename T>
using StatusOrSearcher = StatusOr<unique_ptr<SingleMachineSearcherBase<T>>>;

// 无类型搜索器返回类型
using StatusOrSearcherUntyped =
    StatusOr<unique_ptr<UntypedSingleMachineSearcherBase>>;

// 单机搜索器工厂主入口（类型安全）
template <typename T>
StatusOr<unique_ptr<SingleMachineSearcherBase<T>>> SingleMachineFactoryScann(
    const ScannConfig& config, shared_ptr<TypedDataset<T>> dataset,
    SingleMachineFactoryOptions opts = SingleMachineFactoryOptions());

// 单机搜索器工厂主入口（无类型）
StatusOrSearcherUntyped SingleMachineFactoryUntypedScann(
    const ScannConfig& config, shared_ptr<Dataset> dataset,
    SingleMachineFactoryOptions opts);

namespace internal {

// 叶子搜索器工厂（分区/暴力/哈希等具体实现分派）
template <typename T>
StatusOrSearcherUntyped SingleMachineFactoryLeafSearcherScann(
    const ScannConfig& config, const shared_ptr<TypedDataset<T>>& dataset,
    const GenericSearchParameters& params, SingleMachineFactoryOptions* opts);

}


// 工厂模板实例化宏（为每种类型生成工厂函数声明）
#define SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(          \
    extern_keyword, Type)                                                 \
  extern_keyword template StatusOr<                                       \
      unique_ptr<SingleMachineSearcherBase<Type>>>                        \
  SingleMachineFactoryScann<Type>(const ScannConfig& config,              \
                                  shared_ptr<TypedDataset<Type>> dataset, \
                                  SingleMachineFactoryOptions opts);      \
  extern_keyword template StatusOrSearcherUntyped                         \
  internal::SingleMachineFactoryLeafSearcherScann<Type>(                  \
      const ScannConfig& config,                                          \
      const shared_ptr<TypedDataset<Type>>& dataset,                      \
      const GenericSearchParameters& params,                              \
      SingleMachineFactoryOptions* opts);


// 工厂宏：为所有支持类型生成工厂声明
#define SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN(extern_keyword)    \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    int8_t);        \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    uint8_t);       \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    int16_t);       \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    int32_t);       \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    uint32_t);      \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    int64_t);       \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    float);         \
    SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN_FOR_TYPE(extern_keyword, \
                                                                                                                    double);

SCANN_INSTANTIATE_SINGLE_MACHINE_FACTORY_SCANN(extern);

}  // namespace research_scann

#endif
