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

#ifndef SCANN_BASE_SEARCH_PARAMETERS_H_
#define SCANN_BASE_SEARCH_PARAMETERS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/base/optimization.h"
#include "scann/base/restrict_allowlist.h"
#include "scann/data_format/features.pb.h"
#include "scann/oss_wrappers/scann_aligned_malloc.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class SearcherSpecificOptionalParameters : public VirtualDestructor {};

class SearchParameters {
 public:
  // 声明只允许移动的类，禁止拷贝
  SCANN_DECLARE_MOVE_ONLY_CLASS(SearchParameters);

  // 默认构造函数
  SearchParameters() = default;

  // 构造函数，设置预/后重排序的邻居数和epsilon
  SearchParameters(
      int32_t pre_reordering_num_neighbors, float pre_reordering_epsilon,
      int32_t post_reordering_num_neighbors = numeric_limits<int32_t>::max(),
      float post_reordering_epsilon = numeric_limits<float>::infinity())
      : pre_reordering_num_neighbors_(pre_reordering_num_neighbors),
        post_reordering_num_neighbors_(post_reordering_num_neighbors),
        pre_reordering_epsilon_(pre_reordering_epsilon),
        post_reordering_epsilon_(post_reordering_epsilon) {}

  ~SearchParameters() {}

  // 用默认参数填充未指定的参数
  void SetUnspecifiedParametersFrom(const SearchParameters& defaults);

  // 校验参数合法性
  Status Validate(bool reordering_enabled) const;

  // 是否对结果排序
  bool sort_results() const { return sort_results_; }
  void set_sort_results(bool val) { sort_results_ = val; }

  // 获取/设置预重排序和后重排序的邻居数
  int32_t pre_reordering_num_neighbors() const {
    return pre_reordering_num_neighbors_;
  }
  int32_t post_reordering_num_neighbors() const {
    return post_reordering_num_neighbors_;
  }
  void set_pre_reordering_num_neighbors(int32_t val) {
    pre_reordering_num_neighbors_ = val;
  }
  void set_post_reordering_num_neighbors(int32_t val) {
    post_reordering_num_neighbors_ = val;
  }

  // 获取/设置预重排序和后重排序的epsilon
  float pre_reordering_epsilon() const { return pre_reordering_epsilon_; }
  float post_reordering_epsilon() const { return post_reordering_epsilon_; }
  void set_pre_reordering_epsilon(float val) { pre_reordering_epsilon_ = val; }
  void set_post_reordering_epsilon(float val) {
    post_reordering_epsilon_ = val;
  }

  // 获取/设置crowding属性的邻居数（用于多属性聚类）
  int32_t per_crowding_attribute_pre_reordering_num_neighbors() const {
    return per_crowding_attribute_pre_reordering_num_neighbors_;
  }
  void set_per_crowding_attribute_pre_reordering_num_neighbors(int32_t val) {
    per_crowding_attribute_pre_reordering_num_neighbors_ = val;
  }
  int32_t per_crowding_attribute_post_reordering_num_neighbors() const {
    return per_crowding_attribute_post_reordering_num_neighbors_;
  }
  void set_per_crowding_attribute_post_reordering_num_neighbors(int32_t val) {
    per_crowding_attribute_post_reordering_num_neighbors_ = val;
  }

  // 多维crowding属性结构体
  struct CrowdingDimensionAttributeNumNeighbor {
    std::string dimension; // 维度名
    std::optional<int64_t> attribute; // 可选属性值
    int32_t num_neighbors; // 邻居数
  };
  // 获取/设置多维crowding属性的邻居数
  ConstSpan<CrowdingDimensionAttributeNumNeighbor>
  per_crowding_dimension_attribute_post_reordering_num_neighbors() const {
    return per_crowding_dimension_attribute_post_reordering_num_neighbors_;
  }
  void set_per_crowding_dimension_attribute_post_reordering_num_neighbors(
      ConstSpan<CrowdingDimensionAttributeNumNeighbor> val) {
    per_crowding_dimension_attribute_post_reordering_num_neighbors_ = {
        val.begin(), val.end()};
  }

  // 判断是否启用crowding（预重排序和后重排序）
  bool pre_reordering_crowding_enabled() const {
    return pre_reordering_num_neighbors_ >
           per_crowding_attribute_pre_reordering_num_neighbors_;
  }
  bool post_reordering_crowding_enabled() const {
    return (post_reordering_num_neighbors_ >
            per_crowding_attribute_post_reordering_num_neighbors_) ||
           post_reordering_multi_dimensional_crowding_enabled();
  }

  // 判断是否启用多维crowding
  bool post_reordering_multi_dimensional_crowding_enabled() const {
    for (const auto& entry :
         per_crowding_dimension_attribute_post_reordering_num_neighbors_) {
      if (post_reordering_num_neighbors_ > entry.num_neighbors) return true;
    }
    return false;
  }

  // 是否启用crowding（任意阶段）
  bool crowding_enabled() const {
    return pre_reordering_crowding_enabled() ||
           post_reordering_crowding_enabled();
  }

  // 是否启用restricts（目前总是false）
  bool restricts_enabled() const { return false; }

  // 获取restrict白名单（目前总是nullptr）
  const RestrictAllowlist* restrict_whitelist() const { return nullptr; }

  // 判断某个数据点是否在白名单（目前总是false）
  bool IsWhitelisted(DatapointIndex dp_index) const { return false; }

  // 获取可修改的restrict白名单（目前总是nullptr）
  RestrictAllowlist* mutable_restrict_whitelist() { return nullptr; }

  // 启用restricts（空实现）
  void EnableRestricts(DatapointIndex database_size, bool default_whitelisted) {
  }

  // 禁用restricts（空实现）
  void DisableRestricts() {}

  // 获取searcher特定的可选参数
  const SearcherSpecificOptionalParameters*
  searcher_specific_optional_parameters() const {
    return searcher_specific_optional_parameters_.get();
  }

  // 设置searcher特定的可选参数
  void set_searcher_specific_optional_parameters(
      shared_ptr<const SearcherSpecificOptionalParameters> params) {
    searcher_specific_optional_parameters_ = std::move(params);
  }

  // 获取指定类型的searcher特定可选参数
  template <typename T>
  shared_ptr<const T> searcher_specific_optional_parameters() const {
    return std::dynamic_pointer_cast<const T>(
        searcher_specific_optional_parameters_);
  }

  // 查询预处理结果基类
  class UnlockedQueryPreprocessingResults : public VirtualDestructor {};

  // 设置查询预处理结果
  void set_unlocked_query_preprocessing_results(
      unique_ptr<UnlockedQueryPreprocessingResults> r) {
    unlocked_query_preprocessing_results_ = std::move(r);
  }

  // 获取指定类型的查询预处理结果
  template <typename Subclass>
  Subclass* unlocked_query_preprocessing_results() const {
    return dynamic_cast<Subclass*>(unlocked_query_preprocessing_results_.get());
  }

  // 设置/获取随机邻居数（用于随机采样）
  void set_num_random_neighbors(int32_t num_random_neighbors) {
    num_random_neighbors_ = num_random_neighbors;
  }
  int32_t num_random_neighbors() const { return num_random_neighbors_; }

private:
  bool sort_results_ = true; // 是否排序结果
  int32_t pre_reordering_num_neighbors_ = -1; // 预重排序邻居数
  int32_t post_reordering_num_neighbors_ = -1; // 后重排序邻居数
  float pre_reordering_epsilon_ = NAN; // 预重排序epsilon
  float post_reordering_epsilon_ = NAN; // 后重排序epsilon
  int per_crowding_attribute_pre_reordering_num_neighbors_ =
    numeric_limits<int32_t>::max(); // crowding属性预重排序邻居数
  int per_crowding_attribute_post_reordering_num_neighbors_ =
    numeric_limits<int32_t>::max(); // crowding属性后重排序邻居数
  std::vector<CrowdingDimensionAttributeNumNeighbor>
    per_crowding_dimension_attribute_post_reordering_num_neighbors_; // 多维crowding属性邻居数
  int32_t num_random_neighbors_ = 0; // 随机邻居数

  shared_ptr<const SearcherSpecificOptionalParameters>
    searcher_specific_optional_parameters_; // searcher特定可选参数

  unique_ptr<UnlockedQueryPreprocessingResults>
    unlocked_query_preprocessing_results_; // 查询预处理结果
};

}  // namespace research_scann

#endif
