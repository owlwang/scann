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

#ifndef SCANN_SCANN_OPS_CC_SCANN_NPY_H_
#define SCANN_SCANN_OPS_CC_SCANN_NPY_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
using np_row_major_arr =
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

class ScannNumpy {
public:
    // 构造函数：从资产目录和pbtxt初始化
    ScannNumpy(const std::string& artifacts_dir,
                         const std::string& scann_assets_pbtxt);
    // 构造函数：从numpy数据集和配置初始化
    ScannNumpy(const np_row_major_arr<float>& np_dataset,
                         const std::string& config, int training_threads);
    // 单条查询接口，返回索引和距离
    std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>> Search(
            const np_row_major_arr<float>& query, int final_nn, int pre_reorder_nn,
            int leaves);
    // 批量查询接口，支持并行和批量大小设置
    std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
    SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                                int pre_reorder_nn, int leaves, bool parallel = false,
                                int batch_size = 256);
    // 序列化ScaNN资产到文件
    void Serialize(std::string path, bool relative_path = false);

    // 插入/更新数据点，返回实际插入的索引
    vector<DatapointIndex> Upsert(
            std::vector<std::optional<DatapointIndex>> indices,
            std::vector<np_row_major_arr<float>>& vecs, int batch_size = 256);
    // 删除数据点，返回实际删除的索引
    vector<DatapointIndex> Delete(std::vector<DatapointIndex> indices);

    // 重新平衡索引结构
    int Rebalance(const string& config = "");

    // 获取数据点数量
    size_t Size() const;

    // 设置并行线程数
    void SetNumThreads(int num_threads);

    // 预分配数据点空间
    void Reserve(size_t num_datapoints);

    // 自动推荐配置（根据数据规模和维度）
    static string SuggestAutopilot(absl::string_view config, DatapointIndex n,
                                                                 DimensionIndex dim);

    // 获取当前配置
    string Config();

    // 获取健康状态
    pybind11::dict GetHealthStats() const;
    void InitializeHealthStats();

private:
    ScannInterface scann_; // 底层ScaNN接口对象
};

}  // namespace research_scann

#endif
