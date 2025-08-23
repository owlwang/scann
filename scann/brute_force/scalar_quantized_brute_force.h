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



#ifndef SCANN_BRUTE_FORCE_SCALAR_QUANTIZED_BRUTE_FORCE_H_
#define SCANN_BRUTE_FORCE_SCALAR_QUANTIZED_BRUTE_FORCE_H_

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/tree_x_hybrid/leaf_searcher_optional_parameter_creator.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

// 标量量化暴力搜索器：用于高效地在量化后的向量空间中进行暴力最近邻搜索
class ScalarQuantizedBruteForceSearcher final
    : public SingleMachineSearcherBase<float> {
 public:
    // 量化参数选项：控制量化比例和噪声阈值
    struct Options {
        float multiplier_quantile = 1.0f; // 量化比例分位数
        float noise_shaping_threshold = NAN; // 噪声整形阈值
    };

    // 构造函数：通过原始数据集和距离度量初始化量化暴力搜索器
    ScalarQuantizedBruteForceSearcher(
            shared_ptr<const DistanceMeasure> distance,
            shared_ptr<const DenseDataset<float>> dataset,
            const int32_t default_pre_reordering_num_neighbors,
            const float default_pre_reordering_epsilon, Options opts);

    // 构造函数重载：默认使用 Options()
    ScalarQuantizedBruteForceSearcher(
            shared_ptr<const DistanceMeasure> distance,
            shared_ptr<const DenseDataset<float>> dataset,
            const int32_t default_pre_reordering_num_neighbors,
            const float default_pre_reordering_epsilon)
            : ScalarQuantizedBruteForceSearcher(
                        std::move(distance), std::move(dataset),
                        default_pre_reordering_num_neighbors,
                        default_pre_reordering_epsilon, Options()) {}

    // 析构函数
    ~ScalarQuantizedBruteForceSearcher() override;

    // 是否支持 crowding（多样性约束）
    bool supports_crowding() const final { return true; }

    // 设置最小距离阈值（用于筛选结果）
    void set_min_distance(float min_distance) { min_distance_ = min_distance; }

    // 构造函数：直接通过量化数据集和参数初始化
    ScalarQuantizedBruteForceSearcher(
            shared_ptr<const DistanceMeasure> distance,
            shared_ptr<vector<float>> squared_l2_norms,
            shared_ptr<const DenseDataset<int8_t>> quantized_dataset,
            shared_ptr<const vector<float>> inverse_multiplier_by_dimension,
            int32_t default_num_neighbors, float default_epsilon);

    // 工具函数：从量化数据和反乘数恢复 L2 范数
    static StatusOr<vector<float>> ComputeSquaredL2NormsFromQuantizedDataset(
            const DenseDataset<int8_t>& quantized,
            absl::Span<const float> inverse_multipliers);

    // 工厂方法：通过量化数据和参数创建搜索器
    static StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
    CreateFromQuantizedDatasetAndInverseMultipliers(
            shared_ptr<const DistanceMeasure> distance,
            shared_ptr<const DenseDataset<int8_t>> quantized,
            shared_ptr<const vector<float>> inverse_multipliers,
            shared_ptr<vector<float>> squared_l2_norms, int32_t default_num_neighbors,
            float default_epsilon);

    // 工厂方法：通过每个维度的绝对阈值构造量化器和搜索器
    static StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
    CreateWithFixedRange(shared_ptr<const DistanceMeasure> distance,
                                             shared_ptr<const DenseDataset<float>> dataset,
                                             ConstSpan<float> abs_thresholds_for_each_dimension,
                                             int32_t default_num_neighbors, float default_epsilon);

    // 创建通用暴力搜索器（兼容接口）
    StatusOr<const SingleMachineSearcherBase<float>*> CreateBruteForceSearcher(
            const DistanceMeasureConfig& distance_config,
            unique_ptr<SingleMachineSearcherBase<float>>* storage) const final;

  class Mutator : public SingleMachineSearcherBase<float>::Mutator {
    public:
     // Mutator：支持数据集的动态增删改
    using PrecomputedMutationArtifacts =
        UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;
    using MutateBaseOptions =
        UntypedSingleMachineSearcherBase::UntypedMutator::MutateBaseOptions;

    // 创建 Mutator 实例
    static StatusOr<unique_ptr<Mutator>> Create(
        ScalarQuantizedBruteForceSearcher* searcher);
    Mutator(const Mutator&) = delete;
    Mutator& operator=(const Mutator&) = delete;
    ~Mutator() final {}
    // 增加数据点
    StatusOr<DatapointIndex> AddDatapoint(const DatapointPtr<float>& dptr,
                                          string_view docid,
                                          const MutationOptions&) final;
    // 通过 docid 删除数据点
    Status RemoveDatapoint(string_view docid) final;
    // 预分配空间
    void Reserve(size_t size) final;
    // 通过索引删除数据点
    Status RemoveDatapoint(DatapointIndex index) final;
    // 更新数据点（通过 docid 或索引）
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             string_view docid,
                                             const MutationOptions&) final;
    StatusOr<DatapointIndex> UpdateDatapoint(const DatapointPtr<float>& dptr,
                                             DatapointIndex index,
                                             const MutationOptions&) final;

   private:
    // 构造函数（私有）：持有量化器和数据集变更器
    Mutator(ScalarQuantizedBruteForceSearcher* searcher,
            TypedDataset<int8_t>::Mutator* quantized_dataset_mutator,
            std::vector<float> multipliers)
        : searcher_(searcher),
          quantized_dataset_mutator_(quantized_dataset_mutator),
          multipliers_(std::move(multipliers)),
          quantized_datapoint_(multipliers_.size()) {}
    // 查找 docid 对应索引
    StatusOr<DatapointIndex> LookupDatapointIndexOrError(
        string_view docid) const;
    // 对数据点进行标量量化
    DatapointPtr<int8_t> ScalarQuantize(const DatapointPtr<float>& dptr);

    ScalarQuantizedBruteForceSearcher* searcher_;
    TypedDataset<int8_t>::Mutator* quantized_dataset_mutator_;
    std::vector<float> multipliers_;
    std::vector<int8_t> quantized_datapoint_;
  };

    // 获取 Mutator（用于支持动态数据集变更）
    StatusOr<typename SingleMachineSearcherBase<float>::Mutator*> GetMutator()
            const final;

    // 提取工厂选项（用于重排序加速等）
    StatusOr<SingleMachineFactoryOptions> ExtractSingleMachineFactoryOptions()
            override;

    // 工厂方法（已废弃）：通过量化数据和参数创建搜索器
    ABSL_DEPRECATED("Use shared_ptr overload instead.")
    static StatusOr<unique_ptr<ScalarQuantizedBruteForceSearcher>>
    CreateFromQuantizedDatasetAndInverseMultipliers(
            shared_ptr<const DistanceMeasure> distance,
            DenseDataset<int8_t> quantized, vector<float> inverse_multipliers,
            vector<float> squared_l2_norms, int32_t default_num_neighbors,
            float default_epsilon);

protected:
    // 实际查找邻居的实现（量化距离计算）
    Status FindNeighborsImpl(const DatapointPtr<float>& query,
                                                     const SearchParameters& params,
                                                     NNResultsVector* result) const final;

    // crowding 属性使能实现
    Status EnableCrowdingImpl(
            ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
            ConstSpan<std::string> crowding_dimension_names) final;

    // 距离传播实现（将 dot product 转换为实际距离）
    Status PropagateDistances(const DatapointPtr<float>& query,
                                                        const SearchParameters& params,
                                                        NNResultsVector* result) const override;

private:
    // 距离后处理模板（支持 min_distance、TopN 筛选等）
    template <bool kUseMinDistance, typename ResultElem>
    Status PostprocessDistances(const DatapointPtr<float>& query,
                                                            const SearchParameters& params,
                                                            ConstSpan<ResultElem> dot_products,
                                                            NNResultsVector* result) const;

    // 距离后处理实现（支持距离类型转换）
    template <bool kUseMinDistance, typename DistanceFunctor, typename ResultElem>
    Status PostprocessDistancesImpl(const DatapointPtr<float>& query,
                                                                    const SearchParameters& params,
                                                                    ConstSpan<ResultElem> dot_products,
                                                                    DistanceFunctor distance_functor,
                                                                    NNResultsVector* result) const;

    // TopN筛选实现（不支持 restricts）
    template <bool kUseMinDistance, typename DistanceFunctor, typename TopN>
    Status PostprocessTopNImpl(const DatapointPtr<float>& query,
                                                         const SearchParameters& params,
                                                         ConstSpan<float> dot_products,
                                                         DistanceFunctor distance_functor,
                                                         TopN* top_n_ptr) const;

    // TopN筛选实现（支持 restricts）
    template <bool kUseMinDistance, typename DistanceFunctor, typename TopN>
    Status PostprocessTopNImpl(
            const DatapointPtr<float>& query, const SearchParameters& params,
            ConstSpan<pair<DatapointIndex, float>> dot_products,
            DistanceFunctor distance_functor, TopN* top_n_ptr) const;

    // 是否需要原始数据集（量化暴力搜索不需要）
    bool impl_needs_dataset() const override { return false; }

    // 量化数据集每个点的 L2 范数
    shared_ptr<vector<float>> squared_l2_norms_ = make_shared<vector<float>>();

    // 最小距离阈值
    float min_distance_ = -numeric_limits<float>::infinity();

    // 量化参数选项
    Options opts_;

    // Mutator（支持动态数据集变更）
    mutable unique_ptr<Mutator> mutator_ = nullptr;

    // 每个维度的反乘数（用于量化/反量化）
    shared_ptr<const vector<float>> inverse_multiplier_by_dimension_;

    // 量化后的数据集
    shared_ptr<const DenseDataset<int8_t>> quantized_dataset_;

    // 距离度量对象
    shared_ptr<const DistanceMeasure> distance_;
};

class TreeScalarQuantizationPreprocessedQuery final
    : public SearcherSpecificOptionalParameters {
 public:
  // 树结构量化查询的预处理结果
  explicit TreeScalarQuantizationPreprocessedQuery(
      unique_ptr<float[]> preprocessed_query)
      : preprocessed_query_(std::move(preprocessed_query)) {}

  // 获取预处理后的查询向量
  const float* PreprocessedQuery() const { return preprocessed_query_.get(); }

 private:
  // 持有预处理后的查询向量
  const unique_ptr<float[]> preprocessed_query_;
};

class TreeScalarQuantizationPreprocessedQueryCreator final
    : public LeafSearcherOptionalParameterCreator<float> {
 public:
  // 创建器：用于生成树结构量化查询的预处理参数
  explicit TreeScalarQuantizationPreprocessedQueryCreator(
      vector<float> inverse_multipliers)
      : inverse_multipliers_(std::move(inverse_multipliers)) {}

  // 生成预处理参数
  StatusOr<unique_ptr<SearcherSpecificOptionalParameters>>
  CreateLeafSearcherOptionalParameters(
      const DatapointPtr<float>& query) const override;

  // 获取反乘数
  ConstSpan<float> inverse_multipliers() const;

 private:
  // 每个维度的反乘数
  const vector<float> inverse_multipliers_;
};

}  // namespace research_scann

#endif
