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

#include "scann/distance_measures/distance_measure_factory.h"

namespace research_scann {

// 根据配置 proto 创建距离度量对象，支持多种距离类型
StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(
    const DistanceMeasureConfig& config) {
  // 检查配置是否指定了距离类型
  if (config.distance_measure().empty()) {
    // 未指定距离类型，返回参数错误
    return InvalidArgumentError(
        "Empty DistanceMeasureConfig proto! Must specify distance_measure.");
  }
  // 根据距离类型名称分派到具体实现
  return GetDistanceMeasure(config.distance_measure());
}

// 根据距离类型名称创建对应的距离度量对象
StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(string_view name) {
  // 针对每种支持的距离类型，分派到具体实现类
  if (name == "DotProductDistance")
    return shared_ptr<DistanceMeasure>(new DotProductDistance());
  if (name == "BinaryDotProductDistance")
    return shared_ptr<DistanceMeasure>(new BinaryDotProductDistance());
  if (name == "AbsDotProductDistance")
    return shared_ptr<DistanceMeasure>(new AbsDotProductDistance());
  if (name == "L2Distance")
    return shared_ptr<DistanceMeasure>(new L2Distance());
  if (name == "SquaredL2Distance")
    return shared_ptr<DistanceMeasure>(new SquaredL2Distance());
  if (name == "NegatedSquaredL2Distance")
    return shared_ptr<DistanceMeasure>(new NegatedSquaredL2Distance());
  if (name == "L1Distance")
    return shared_ptr<DistanceMeasure>(new L1Distance());
  if (name == "CosineDistance")
    return shared_ptr<DistanceMeasure>(new CosineDistance());
  if (name == "BinaryCosineDistance")
    return shared_ptr<DistanceMeasure>(new BinaryCosineDistance());
  if (name == "GeneralJaccardDistance")
    return shared_ptr<DistanceMeasure>(new GeneralJaccardDistance());
  if (name == "BinaryJaccardDistance")
    return shared_ptr<DistanceMeasure>(new BinaryJaccardDistance());
  if (name == "LimitedInnerProductDistance")
    return shared_ptr<DistanceMeasure>(new LimitedInnerProductDistance());
  if (name == "GeneralHammingDistance")
    return shared_ptr<DistanceMeasure>(new GeneralHammingDistance());
  if (name == "BinaryHammingDistance")
    return shared_ptr<DistanceMeasure>(new BinaryHammingDistance());
  if (name == "NonzeroIntersectDistance")
    return shared_ptr<DistanceMeasure>(new NonzeroIntersectDistance());
  // 未知距离类型，返回参数错误
  return InvalidArgumentError("Invalid distance_measure: '%s'", name);
}

// research_scann 命名空间结束
}  // namespace research_scann
