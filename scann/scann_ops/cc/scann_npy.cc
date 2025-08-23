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

#include "scann/scann_ops/cc/scann_npy.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/common.h"
#include "scann/utils/io_oss_wrapper.h"
#include "scann/utils/single_machine_autopilot.h"
#include "scann/utils/types.h"

namespace research_scann {
using MutationOptions = UntypedSingleMachineSearcherBase::MutationOptions;
using PrecomputedMutationArtifacts =
    UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts;

void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  // 如果状态不正常则抛出异常
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.message());
    throw std::runtime_error(msg);
  }
}

template <typename T>
// 获取StatusOr的值，不正常则抛异常
template <typename T>
T ValueOrRuntimeError(StatusOr<T> status_or, const char* prefix) {
  RuntimeErrorIfNotOk(prefix, status_or.status());
  return status_or.value();
}

ScannNumpy::ScannNumpy(const std::string& artifacts_dir,
                       const std::string& scann_assets_pbtxt) {
  // 从资产目录和pbtxt初始化ScaNN对象
  auto status_or =
      ScannInterface::LoadArtifacts(artifacts_dir, scann_assets_pbtxt);
  RuntimeErrorIfNotOk("Error loading artifacts: ", status_or.status());
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(status_or.value()));
}

ScannNumpy::ScannNumpy(const np_row_major_arr<float>& np_dataset,
                       const std::string& config, int training_threads) {
  // 从numpy数据集和配置初始化ScaNN对象
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());
  pybind11::gil_scoped_release gil_release;
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, np_dataset.shape()[0], config,
                                        training_threads));
}

vector<DatapointIndex> ScannNumpy::Upsert(
    vector<std::optional<DatapointIndex>> indices,
    vector<np_row_major_arr<float>>& vecs, int batch_size) {
  // 批量插入/更新数据点，支持多线程和预计算
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  if (batch_size > 1)
    mutator->set_mutation_threadpool(scann_.parallel_query_pool());
  if (indices.size() != vecs.size())
    throw std::runtime_error("Upsert input size must match.");

  DatapointIndex n = vecs.size();
  vector<DatapointIndex> result;

  for (size_t b : Seq(DivRoundUp(n, batch_size))) {
    size_t begin = batch_size * b;
    size_t bs = std::min<DatapointIndex>(n - begin, batch_size);
    DenseDataset<float> ds;
    // 构建本批次数据集
    for (size_t i : Seq(bs))
      RuntimeErrorIfNotOk("Error appending datapoint.",
                          ds.Append(MakeDatapointPtr(vecs[begin + i].data(),
                                                     vecs[begin + i].size())));
    auto precomputed = mutator->ComputePrecomputedMutationArtifacts(
        ds, scann_.parallel_query_pool());

    // 插入或更新每个数据点
    for (size_t i : Seq(bs)) {
      auto& index = indices[begin + i];
      auto& vec = vecs[begin + i];
      auto mo = MutationOptions{.precomputed_mutation_artifacts =
                                    precomputed[i].get()};
      if (!index.has_value()) {
        // 新增数据点
        result.push_back(ValueOrRuntimeError(
            mutator->AddDatapoint(MakeDatapointPtr(vec.data(), vec.size()), "",
                                  mo),
            "Failed to add datapoint: "));
      } else {
        // 更新已有数据点
        result.push_back(ValueOrRuntimeError(
            mutator->UpdateDatapoint(MakeDatapointPtr(vec.data(), vec.size()),
                                     index.value(), mo),
            "Failed to update datapoint: "));
      }
    }
    // 增量维护索引结构
    auto statusor = mutator->IncrementalMaintenance();
    RuntimeErrorIfNotOk("Error performing incremental maintenance ",
                        statusor.status());
    if (statusor.value().has_value()) {
      // 需要重平衡时自动重建索引
      Rebalance();
      mutator =
          ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
      mutator->set_mutation_threadpool(scann_.parallel_query_pool());
    }
  }
  return result;
}

vector<DatapointIndex> ScannNumpy::Delete(vector<DatapointIndex> indices) {
  // 批量删除数据点，支持增量维护和自动重建
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  mutator->set_mutation_threadpool(scann_.parallel_query_pool());
  vector<DatapointIndex> result;
  for (const auto& index : indices) {
    RuntimeErrorIfNotOk("Failed to delete datapoint: ",
                        mutator->RemoveDatapoint(index));
    auto statusor = mutator->IncrementalMaintenance();
    RuntimeErrorIfNotOk("Error performing incremental maintenance ",
                        statusor.status());
    if (statusor.value().has_value()) {
      // 需要重平衡时自动重建索引
      Rebalance();
      mutator =
          ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
    }
    result.push_back(scann_.n_points());
  }
  return result;
}

int ScannNumpy::Rebalance(const string& config) {
  // 重新训练和重建索引结构
  auto statusor = scann_.RetrainAndReindex(config);
  if (!statusor.ok()) {
    RuntimeErrorIfNotOk("Failed to retrain searcher: ", statusor.status());
    return -1;
  }

  return scann_.n_points();
}

size_t ScannNumpy::Size() const { return scann_.n_points(); }
// 获取数据点数量

void ScannNumpy::Reserve(size_t num_datapoints) {
  // 预分配数据点空间
  auto mutator =
      ValueOrRuntimeError(scann_.GetMutator(), "Failed to fetch mutator: ");
  mutator->Reserve(num_datapoints);
}

void ScannNumpy::SetNumThreads(int num_threads) {
  // 设置并行线程数
  scann_.SetNumThreads(num_threads);
}

string ScannNumpy::SuggestAutopilot(absl::string_view config_str,
                                    DatapointIndex n, DimensionIndex dim) {
  // 自动推荐配置（根据数据规模和维度）
  ScannConfig config;
  RuntimeErrorIfNotOk("Failed to parse config: ",
                      ParseTextProto(&config, config_str));
  auto status_or = Autopilot(config, nullptr, n, dim);
  RuntimeErrorIfNotOk("Failed to suggest autopilot config: ",
                      status_or.status());
  std::string result;
  google::protobuf::TextFormat::PrintToString(status_or.value(), &result);
  return result;
}

string ScannNumpy::Config() {
  // 获取当前配置字符串
  std::string config_str;
  google::protobuf::TextFormat::PrintToString(*scann_.config(), &config_str);
  return config_str;
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::Search(const np_row_major_arr<float>& query, int final_nn,
                   int pre_reorder_nn, int leaves) {
  // 单条查询，返回索引和距离
  if (query.ndim() != 1)
    throw std::invalid_argument("Query must be one-dimensional");

  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  NNResultsVector res;
  {
    pybind11::gil_scoped_release gil_release;
    auto status = scann_.Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);
  }

  pybind11::array_t<DatapointIndex> indices(res.size());
  pybind11::array_t<float> distances(res.size());
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeNNResult(res, idx_ptr, dis_ptr);
  return {indices, distances};
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                          int pre_reorder_nn, int leaves, bool parallel,
                          int batch_size) {
  // 批量查询，支持并行和批量大小设置，返回索引和距离
  if (queries.ndim() != 2)
    throw std::invalid_argument("Queries must be in two-dimensional array");

  vector<float> queries_vec(queries.data(), queries.data() + queries.size());
  auto query_dataset =
      DenseDataset<float>(std::move(queries_vec), queries.shape()[0]);

  std::vector<NNResultsVector> res(query_dataset.size());
  {
    pybind11::gil_scoped_release gil_release;
    Status status;
    if (parallel)
      status = scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res),
                                            final_nn, pre_reorder_nn, leaves,
                                            batch_size);
    else
      status = scann_.SearchBatched(query_dataset, MakeMutableSpan(res),
                                    final_nn, pre_reorder_nn, leaves);
    RuntimeErrorIfNotOk("Error during search: ", status);
  }

  // 结果补齐到final_nn
  for (const auto& nn_res : res)
    final_nn = std::max<int>(final_nn, nn_res.size());
  pybind11::array_t<DatapointIndex> indices(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  pybind11::array_t<float> distances(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeBatchedNNResult(MakeConstSpan(res), idx_ptr, dis_ptr, final_nn);
  return {indices, distances};
}

void ScannNumpy::Serialize(std::string path, bool relative_path) {
  // 序列化ScaNN资产到文件
  StatusOr<ScannAssets> assets_or = scann_.Serialize(path, relative_path);
  RuntimeErrorIfNotOk("Failed to extract SingleMachineFactoryOptions: ",
                      assets_or.status());
  std::string assets_or_text;
  google::protobuf::TextFormat::PrintToString(*assets_or, &assets_or_text);
  RuntimeErrorIfNotOk("Failed to write ScannAssets proto: ",
                      OpenSourceableFileWriter(path + "/scann_assets.pbtxt")
                          .Write(assets_or_text));
}

pybind11::dict ScannNumpy::GetHealthStats() const {
  // 获取健康状态，返回Python字典
  auto r = scann_.GetHealthStats();
  RuntimeErrorIfNotOk("Error getting health stats: ", r.status());

  using namespace pybind11::literals;

  return pybind11::dict("avg_quantization_error"_a = r->avg_quantization_error,
            "partition_weighted_avg_relative_imbalance"_a =
              r->partition_weighted_avg_relative_imbalance,
            "partition_avg_relative_positive_imbalance"_a =
              r->partition_avg_relative_positive_imbalance,
            "sum_partition_sizes"_a = r->sum_partition_sizes);
}

void ScannNumpy::InitializeHealthStats() {
  // 初始化健康状态
  Status status = scann_.InitializeHealthStats();
  RuntimeErrorIfNotOk("Error initializing health stats: ", status);
}

}  // namespace research_scann
