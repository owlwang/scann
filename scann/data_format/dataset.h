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



#ifndef SCANN_DATA_FORMAT_DATASET_H_
#define SCANN_DATA_FORMAT_DATASET_H_

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/prefetch.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "scann/data_format/datapoint.h"
#include "scann/data_format/docid_collection.h"
#include "scann/data_format/docid_collection_interface.h"
#include "scann/data_format/features.pb.h"
#include "scann/data_format/sparse_low_level.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/proto/hashed.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/iterators.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"

namespace research_scann {


// TypedDataset：数据集模板基类，支持不同类型数据点
template <typename T>
class TypedDataset;
// 稠密数据集，所有数据点均有完整维度
template <typename T>
class DenseDataset;
// 稀疏数据集，仅存储非零维度
template <typename T>
class SparseDataset;

class Dataset : public VirtualDestructor {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(Dataset);

  // 构造函数，默认使用空 docid 集合
  Dataset() : docids_(make_shared<VariableLengthDocidCollection>()) {}

  // 构造函数，使用指定 docid 集合
  explicit Dataset(unique_ptr<DocidCollectionInterface> docids)
      : docids_(std::move(docids)) {
    DCHECK(docids_);
  }

  // 数据集大小（数据点数量）
  DatapointIndex size() const { return docids_->size(); }

  // 判断数据集是否为空
  bool empty() const { return size() == 0; }

  // 数据集维度
  DimensionIndex dimensionality() const { return dimensionality_; }

  // 活跃维度数（抽象接口，稠密/稀疏实现不同）
  virtual DimensionIndex NumActiveDimensions() const = 0;

  // 是否为稠密数据集（抽象接口）
  virtual bool IsDense() const = 0;

  // 是否为稀疏数据集
  bool IsSparse() const { return !IsDense(); }

  // 设置数据集维度（抽象接口）
  virtual void set_dimensionality(DimensionIndex dimensionality) = 0;

  // 预分配数据点空间（可选实现）
  virtual void Reserve(size_t n_points) {}

  // 预分配 docid 空间
  void ReserveDocids(size_t n_docids) { docids_->Reserve(n_docids); }

  // 获取指定索引的 docid
  string_view GetDocid(size_t index) const { return docids_->Get(index); }

  // 获取 docid 集合智能指针
  const shared_ptr<DocidCollectionInterface>& docids() const { return docids_; }

  // 释放 docid 集合，返回原有 docids 并重置为空 docids
  virtual shared_ptr<DocidCollectionInterface> ReleaseDocids();

  // 清空数据集（抽象接口）
  virtual void clear() = 0;

  // 获取数据类型标签（float/double/int 等）
  virtual research_scann::TypeTag TypeTag() const = 0;

  // 获取指定索引的数据点（double/float），以及稠密数据点接口
  virtual void GetDatapoint(size_t index, Datapoint<double>* result) const = 0;
  virtual void GetDatapoint(size_t index, Datapoint<float>* result) const = 0;
  virtual void GetDenseDatapoint(size_t index,
                                 Datapoint<double>* result) const = 0;
  virtual void GetDenseDatapoint(size_t index,
                                 Datapoint<float>* result) const = 0;
  // 预取指定索引数据点到本地缓存（加速访问）
  virtual void Prefetch(size_t index) const = 0;
  // 计算两个数据点之间的距离（抽象接口）
  virtual double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                             size_t vec2_index) const = 0;
  // 按维度计算均值/方差（全量/子集）
  virtual Status MeanByDimension(Datapoint<double>* result) const = 0;
  virtual Status MeanByDimension(ConstSpan<DatapointIndex> subset,
                                 Datapoint<double>* result) const = 0;
  virtual void MeanVarianceByDimension(Datapoint<double>* means,
                                       Datapoint<double>* variances) const = 0;
  virtual void MeanVarianceByDimension(ConstSpan<DatapointIndex> subset,
                                       Datapoint<double>* means,
                                       Datapoint<double>* variances) const = 0;
  // 数据归一化（单位 L2/零均值单位方差）
  virtual Status NormalizeUnitL2() = 0;
  virtual Status NormalizeZeroMeanUnitVariance() = 0;

  // 当前归一化类型
  Normalization normalization() const { return normalization_; }
  // 按指定归一化类型归一化
  Status NormalizeByTag(Normalization tag);
  // 设置归一化类型标签
  void set_normalization_tag(Normalization tag) { normalization_ = tag; }

  // 获取/设置数据集打包策略（如二值、半字节等）
  HashedItem::PackingStrategy packing_strategy() const {
    return packing_strategy_;
  }
  virtual void set_packing_strategy(
      HashedItem::PackingStrategy packing_strategy) {
    packing_strategy_ = packing_strategy;
  }

  // 是否为浮点型数据集
  virtual bool is_float() const = 0;
  // 是否为二值数据集
  bool is_binary() const { return packing_strategy_ == HashedItem::BINARY; }
  // 设置为二值数据集
  virtual void set_is_binary(bool val) {
    packing_strategy_ = val ? HashedItem::BINARY : HashedItem::NONE;
  }

  // 收缩内存到实际大小
  virtual void ShrinkToFit() {}

  // docid 数组容量
  size_t DocidArrayCapacity() const { return docids_->capacity(); }
  // 计算数据集占用内存（不含 docids）
  virtual size_t MemoryUsageExcludingDocids() const = 0;
  // docid 占用内存
  size_t DocidMemoryUsage() const { return docids_->MemoryUsage(); }

  // 绑定 docid 集合到数据集（需与当前 size 匹配）
  void AttachDocidCollection(shared_ptr<DocidCollectionInterface> docids) {
    DCHECK(docids);
    DCHECK_EQ(docids->size(), size());
    set_docids_no_checks(docids);
  }

  // Mutator：数据集可变操作接口
  class Mutator;
  // 获取未指定类型的 Mutator
  virtual StatusOr<typename Dataset::Mutator*> GetUntypedMutator() const = 0;

protected:
  // 直接设置维度（不做一致性检查）
  void set_dimensionality_no_checks(DimensionIndex dim) {
    dimensionality_ = dim;
  }
  // 直接设置 docid 集合（不做一致性检查）
  void set_docids_no_checks(shared_ptr<DocidCollectionInterface> docids) {
    docids_ = std::move(docids);
  }
  // 设置归一化类型
  void set_normalization(Normalization norm) { normalization_ = norm; }
  // 追加 docid 到集合
  Status AppendDocid(string_view docid) { return docids_->Append(docid); }

 private:
  shared_ptr<DocidCollectionInterface> docids_;

  DimensionIndex dimensionality_ = 0;

  Normalization normalization_ = NONE;

  HashedItem::PackingStrategy packing_strategy_ = HashedItem::NONE;
};

class Dataset::Mutator : public VirtualDestructor {
 public:
  // 移除指定 docid 的数据点
  virtual Status RemoveDatapoint(string_view docid) = 0;
  // 查找 docid 对应的数据点索引
  virtual bool LookupDatapointIndex(string_view docid,
                                    DatapointIndex* index) const = 0;
  // 预分配空间
  virtual void Reserve(size_t size) = 0;
  // 按索引移除数据点
  virtual Status RemoveDatapoint(DatapointIndex index) = 0;
};

template <typename T>
// TypedDataset：数据集模板基类，支持泛型数据点操作
class TypedDataset : public Dataset {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(TypedDataset);

  TypedDataset() = default;

  explicit TypedDataset(unique_ptr<DocidCollectionInterface> docids)
      : Dataset(std::move(docids)) {}

  research_scann::TypeTag TypeTag() const final { return TagForType<T>(); }

  bool is_float() const final { return std::is_floating_point<T>::value; }

  using const_iterator = RandomAccessIterator<const TypedDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  virtual DatapointPtr<T> operator[](size_t datapoint_index) const = 0;

  DatapointPtr<T> at(size_t datapoint_index) const {
    CHECK_LT(datapoint_index, size());
    return operator[](datapoint_index);
  }

  virtual Status Append(const DatapointPtr<T>& dptr, string_view docid) = 0;
  Status Append(const DatapointPtr<T>& dptr);

  virtual Status Append(const GenericFeatureVector& gfv, string_view docid) = 0;
  Status Append(const GenericFeatureVector& gfv);

  void AppendOrDie(const DatapointPtr<T>& dptr, string_view docid);
  void AppendOrDie(const GenericFeatureVector& gfv, string_view docid);
  void AppendOrDie(const DatapointPtr<T>& dptr);
  void AppendOrDie(const GenericFeatureVector& gfv);

  void GetDatapoint(size_t index, Datapoint<double>* result) const final;
  void GetDatapoint(size_t index, Datapoint<float>* result) const final;
  Status MeanByDimension(Datapoint<double>* result) const final;
  Status MeanByDimension(ConstSpan<DatapointIndex> subset,
                         Datapoint<double>* result) const final;
  void MeanVarianceByDimension(Datapoint<double>* means,
                               Datapoint<double>* variances) const final;
  void MeanVarianceByDimension(ConstSpan<DatapointIndex> subset,
                               Datapoint<double>* means,
                               Datapoint<double>* variances) const final;
  Status NormalizeUnitL2() final;
  Status NormalizeZeroMeanUnitVariance() final;

  class Mutator;
  virtual StatusOr<typename TypedDataset::Mutator*> GetMutator() const = 0;
  StatusOr<typename Dataset::Mutator*> GetUntypedMutator() const override {
    SCANN_ASSIGN_OR_RETURN(Dataset::Mutator * result, GetMutator());
    return result;
  }
};

template <typename T>
class TypedDataset<T>::Mutator : public Dataset::Mutator {
 public:
  virtual StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex index) const = 0;

  virtual Status AddDatapoint(const DatapointPtr<T>& dptr,
                              string_view docid) = 0;

  Status RemoveDatapoint(string_view docid) override = 0;

  virtual Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                                 string_view docid) = 0;

  bool LookupDatapointIndex(string_view docid,
                            DatapointIndex* index) const override = 0;

  void Reserve(size_t size) override = 0;

  Status RemoveDatapoint(DatapointIndex index) override = 0;
  virtual Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                                 DatapointIndex index) = 0;
};

template <typename T>
// DenseDataset：稠密数据集实现，所有数据点均有完整维度
class DenseDataset final : public TypedDataset<T> {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(DenseDataset);

  DenseDataset() {}

  explicit DenseDataset(unique_ptr<DocidCollectionInterface> docids)
      : TypedDataset<T>(std::move(docids)) {}

  DenseDataset(std::vector<T> datapoint_vec,
               unique_ptr<DocidCollectionInterface> docids);

  DenseDataset(std::vector<T>&& datapoint_vec, size_t num_dp);

  DenseDataset<T> Copy() const {
    auto result = DenseDataset<T>(data_, this->docids()->Copy());
    result.set_normalization_tag(this->normalization());

    result.set_dimensionality(this->dimensionality());
    return result;
  }

  bool IsDense() const final { return true; }

  using const_iterator = RandomAccessIterator<const DenseDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  size_t n_elements() const {
    return static_cast<size_t>(this->size()) *
           static_cast<size_t>(this->dimensionality());
  }

  void Reserve(size_t n) final;
  void ReserveImpl(size_t n);

  void Resize(size_t n);

  template <typename Real>
  void ConvertType(DenseDataset<Real>* target) const {
    ConvertType(target, this->size());
  }
  template <typename Real>
  void ConvertType(DenseDataset<Real>* target, DatapointIndex first_n) const;

  ConstSpan<T> data() const { return data_; }
  ConstSpan<T> data(size_t index) const {
    return MakeConstSpan(data_.data() + index * stride_, stride_);
  }
  MutableSpan<T> mutable_data() { return MakeMutableSpan(data_); }
  MutableSpan<T> mutable_data(size_t index) {
    return MakeMutableSpan(data_.data() + index * stride_, stride_);
  }

  vector<T> ClearRecyclingDataVector() {
    vector<T> result = std::move(data_);
    this->clear();
    return result;
  }

  void clear() final;
  DimensionIndex NumActiveDimensions() const final;
  void ShrinkToFit() final;
  size_t MemoryUsageExcludingDocids() const final;
  inline DatapointPtr<T> operator[](size_t i) const final;
  void set_dimensionality(DimensionIndex dimensionality) final;
  void set_is_binary(bool val) final;
  void GetDenseDatapoint(size_t index, Datapoint<double>* result) const final;
  void GetDenseDatapoint(size_t index, Datapoint<float>* result) const final;
  inline void Prefetch(size_t index) const final;
  double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                     size_t vec2_index) const final;
  using TypedDataset<T>::Append;
  Status Append(const DatapointPtr<T>& dptr, string_view docid) final;
  Status Append(const GenericFeatureVector& gfv, string_view docid) final;
  shared_ptr<DocidCollectionInterface> ReleaseDocids() final;

  using TypedDataset<T>::AppendOrDie;

  void AppendOrDie(ConstSpan<T> values, string_view docid) {
    AppendOrDie(MakeDatapointPtr<T>(values), docid);
  }
  void AppendOrDie(ConstSpan<T> values) {
    AppendOrDie(MakeDatapointPtr<T>(values), absl::StrCat(this->size()));
  }

  class Mutator : public TypedDataset<T>::Mutator {
   public:
    SCANN_DECLARE_MOVE_ONLY_CLASS(Mutator);

    static StatusOr<unique_ptr<typename DenseDataset<T>::Mutator>> Create(
        DenseDataset<T>* dataset);

    ~Mutator() final {}

    StatusOr<Datapoint<T>> GetDatapoint(DatapointIndex index) const final;

    Status AddDatapoint(const DatapointPtr<T>& dptr, string_view docid) final;

    Status RemoveDatapoint(string_view docid) final;
    Status RemoveDatapoint(DatapointIndex index) final;

    Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                           string_view docid) final;
    Status UpdateDatapoint(const DatapointPtr<T>& dptr,
                           DatapointIndex index) final;

    bool LookupDatapointIndex(string_view docid,
                              DatapointIndex* index) const final;

    void Reserve(size_t size) final;

   private:
    explicit Mutator(DenseDataset<T>* dataset,
                     DocidCollectionInterface::Mutator* docid_mutator)
        : dataset_(dataset), docid_mutator_(docid_mutator) {}
    DenseDataset<T>* dataset_ = nullptr;
    DocidCollectionInterface::Mutator* docid_mutator_ = nullptr;
  };

  StatusOr<typename TypedDataset<T>::Mutator*> GetMutator() const final;

 private:
  void SetStride();

  std::vector<T> data_;

  DimensionIndex stride_ = 0;

  mutable unique_ptr<typename DenseDataset<T>::Mutator> mutator_;

  template <typename U>
  friend class DenseDataset;
};

template <typename T>
class DenseDatasetSubView;
template <typename T>
class RandomDatapointsSubView;

template <typename T>
// DenseDatasetView：稠密数据集视图抽象，支持子视图/随机子集
class DenseDatasetView : VirtualDestructor {
 public:
  DenseDatasetView() = default;

  virtual const T* GetPtr(size_t i) const = 0;

  SCANN_INLINE ConstSpan<T> GetDatapointSpan(size_t i) const {
    return MakeConstSpan(GetPtr(i), dimensionality());
  }

  virtual size_t dimensionality() const = 0;

  virtual size_t size() const = 0;

  virtual research_scann::TypeTag TypeTag() const { return TagForType<T>(); }

  virtual bool IsConsecutiveStorage() const { return false; }

  virtual std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                                       size_t size) const {
    return std::make_unique<DenseDatasetSubView<T>>(this, offset, size);
  }

  virtual std::unique_ptr<DenseDatasetView<T>> random_datapoints_subview(
      ConstSpan<DatapointIndex> dp_idxs) const {
    return std::make_unique<RandomDatapointsSubView<T>>(this, dp_idxs);
  }
};

template <typename T>
// DefaultDenseDatasetView：默认稠密数据集视图实现
class DefaultDenseDatasetView : public DenseDatasetView<T> {
 public:
  DefaultDenseDatasetView() = default;

  DefaultDenseDatasetView(const DenseDataset<T>& ds)
      : ptr_(ds.data().data()), size_(ds.size()) {
    if (ds.packing_strategy() == HashedItem::BINARY) {
      dims_ = ds.dimensionality() / 8 + (ds.dimensionality() % 8 > 0);
    } else if (ds.packing_strategy() == HashedItem::NIBBLE) {
      dims_ = ds.dimensionality() / 2 + (ds.dimensionality() % 2 > 0);
    } else {
      dims_ = ds.dimensionality();
    }
  }

  explicit DefaultDenseDatasetView(ConstSpan<T> span, size_t dimensionality)
      : ptr_(span.data()),
        dims_(dimensionality),
        size_(span.size() / dimensionality) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return ptr_ + i * dims_;
  }

  SCANN_INLINE size_t dimensionality() const final { return dims_; }

  SCANN_INLINE size_t size() const final { return size_; }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    return absl::WrapUnique(
        new DefaultDenseDatasetView<T>(ptr_ + offset * dims_, dims_, size));
  }

  bool IsConsecutiveStorage() const override { return true; }

  ConstSpan<T> data() const { return ConstSpan<T>(ptr_, dims_ * size_); }

 private:
  DefaultDenseDatasetView(const T* ptr, size_t dim, size_t size)
      : ptr_(ptr), dims_(dim), size_(size) {}

  const T* __restrict__ ptr_ = nullptr;
  size_t dims_ = 0;
  size_t size_ = 0;
};

template <typename T>
// DenseDatasetSubView：稠密数据集子视图实现
class DenseDatasetSubView : public DenseDatasetView<T> {
 public:
  DenseDatasetSubView(const DenseDatasetView<T>* parent, size_t offset,
                      size_t size)
      : parent_view_(parent), offset_(offset), size_(size) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return parent_view_->GetPtr(offset_ + i);
  }

  SCANN_INLINE size_t dimensionality() const final {
    return parent_view_->dimensionality();
  };

  SCANN_INLINE size_t size() const final { return size_; }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    return std::make_unique<DenseDatasetSubView<T>>(parent_view_,
                                                    offset + offset_, size);
  }

  bool IsConsecutiveStorage() const override {
    return parent_view_->IsConsecutiveStorage();
  }

 private:
  const DenseDatasetView<T>* __restrict__ parent_view_ = nullptr;
  const size_t offset_ = 0;
  const size_t size_ = 0;
};

template <typename T>
// RandomDatapointsSubView：稠密数据集随机子集视图实现
class RandomDatapointsSubView : public DenseDatasetView<T> {
 public:
  RandomDatapointsSubView(const DenseDatasetView<T>* parent,
                          ConstSpan<DatapointIndex> dp_idxs)
      : parent_view_(parent), dp_idxs_(dp_idxs.begin(), dp_idxs.end()) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return parent_view_->GetPtr(dp_idxs_[i]);
  }

  SCANN_INLINE size_t dimensionality() const final {
    return parent_view_->dimensionality();
  };

  SCANN_INLINE size_t size() const final { return dp_idxs_.size(); }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    return std::make_unique<DenseDatasetSubView<T>>(this, offset, size);
  }

  bool IsConsecutiveStorage() const override { return false; }

 private:
  const DenseDatasetView<T>* __restrict__ parent_view_ = nullptr;
  const std::vector<DatapointIndex> dp_idxs_;
};

template <typename T>
// StridedDatasetView：支持 stride 步长的稠密数据集视图
class StridedDatasetView final : public DenseDatasetView<T> {
 public:
  StridedDatasetView(const T* ptr, size_t dimension, size_t stride, size_t size)
      : ptr_(ptr), dims_(dimension), stride_(stride), size_(size) {}

  SCANN_INLINE const T* GetPtr(size_t i) const final {
    return ptr_ + i * stride_;
  }

  SCANN_INLINE size_t dimensionality() const final { return dims_; }

  SCANN_INLINE size_t size() const final { return size_; }

  std::unique_ptr<DenseDatasetView<T>> subview(size_t offset,
                                               size_t size) const final {
    CHECK_LE(offset + size, size_);
    return absl::WrapUnique(new StridedDatasetView<T>(ptr_ + offset * stride_,
                                                      dims_, stride_, size));
  }

 private:
  const T* __restrict__ ptr_ = nullptr;
  size_t dims_ = 0;
  size_t stride_ = 0;
  size_t size_ = 0;
};

template <typename T>
// SpanDenseDatasetView：基于 Span 的稠密数据集视图
class SpanDenseDatasetView final : public DenseDatasetView<T> {
 public:
  SpanDenseDatasetView(ConstSpan<T> span, size_t dimension)
      : ptr_(span.data()),
        dimension_(dimension),
        size_(span.size() / dimension) {
    CHECK_EQ(span.size() % dimension, 0);
  }

  SCANN_INLINE const T* GetPtr(size_t i) const override {
    return ptr_ + i * dimension_;
  }
  SCANN_INLINE size_t dimensionality() const override { return dimension_; }
  SCANN_INLINE size_t size() const override { return size_; }
  SCANN_INLINE bool IsConsecutiveStorage() const override { return true; }

 private:
  const T* __restrict__ ptr_ = nullptr;
  uint32_t dimension_;
  uint32_t size_;
};

template <typename T>
// SparseDataset：稀疏数据集实现，仅存储非零维度
class SparseDataset final : public TypedDataset<T> {
 public:
  SCANN_DECLARE_MOVE_ONLY_CLASS(SparseDataset);

  SparseDataset() {}

  explicit SparseDataset(unique_ptr<DocidCollectionInterface> docids)
      : TypedDataset<T>(std::move(docids)) {}

  explicit SparseDataset(DimensionIndex dimensionality) : SparseDataset() {
    this->set_dimensionality(dimensionality);
  }

  bool IsDense() const final { return false; }

  using const_iterator = RandomAccessIterator<const SparseDataset<T>>;
  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, this->size()); }

  using TypedDataset<T>::Append;

  Status Append(const GenericFeatureVector& gfv, string_view docid) final;
  Status Append(const DatapointPtr<T>& dptr, string_view docid) final;

  using TypedDataset<T>::AppendOrDie;

  void AppendOrDie(ConstSpan<DimensionIndex> indices, ConstSpan<T> values,
                   string_view docid = "") {
    AppendOrDie(MakeDatapointPtr<T>(indices, values, this->dimensionality()),
                docid);
  }

  void Reserve(size_t n_points) final;

  void Reserve(size_t n_points, size_t n_entries);

  DimensionIndex NonzeroEntriesForDatapoint(DatapointIndex i) const {
    return repr_.NonzeroEntriesForDatapoint(i);
  }

  size_t num_entries() const { return repr_.indices().size(); }

  bool AllValuesNonNegative() const {
    return std::is_unsigned<T>::value || repr_.values().empty() ||
           *std::min_element(repr_.values().begin(), repr_.values().end()) >= 0;
  }

  void clear() final;

  void ConvertType(SparseDataset<double>* target);

  inline DatapointPtr<T> operator[](size_t i) const final;
  void set_dimensionality(DimensionIndex dimensionality) final;
  DimensionIndex NumActiveDimensions() const final;
  void GetDenseDatapoint(size_t index, Datapoint<double>* result) const final;
  void GetDenseDatapoint(size_t index, Datapoint<float>* result) const final;
  inline void Prefetch(size_t index) const final;
  double GetDistance(const DistanceMeasure& dist, size_t vec1_index,
                     size_t vec2_index) const final;
  size_t MemoryUsageExcludingDocids() const final;
  void ShrinkToFit() final;

  StatusOr<typename TypedDataset<T>::Mutator*> GetMutator() const final {
    return UnimplementedError("Sparse dataset does not support mutation.");
  }

 private:
  Status AppendImpl(const GenericFeatureVector& gfv, string_view docid);
  Status AppendImpl(const DatapointPtr<T>& dptr, string_view docid);

  template <typename OutT>
  void GetDenseDatapointImpl(size_t index, Datapoint<OutT>* result) const;

  mutable SparseDatasetLowLevel<DimensionIndex, T> repr_;

  template <typename U>
  friend class SparseDataset;
};

template <typename T>
DatapointPtr<T> DenseDataset<T>::operator[](size_t i) const {
  DCHECK_LT(i, this->size());
  return MakeDatapointPtr(nullptr, data_.data() + i * stride_, stride_,
                          this->dimensionality());
}

template <typename T>
void DenseDataset<T>::Prefetch(size_t i) const {
  DCHECK_LT(i, this->size());
  absl::PrefetchToLocalCacheNta(
      reinterpret_cast<const char*>(data_.data() + i * stride_));
}

template <typename T>
template <typename Real>
void DenseDataset<T>::ConvertType(DenseDataset<Real>* target,
                                  DatapointIndex first_n) const {
  static_assert(std::is_floating_point<Real>(),
                "Real template parameter must be either float or double for "
                "DenseDataset::ConvertType.");
  CHECK(!this->is_binary()) << "Not implemented for binary datasets.";
  DCHECK(target);
  target->clear();
  target->set_dimensionality_no_checks(this->dimensionality());
  target->stride_ = stride_;
  first_n = std::min(first_n, this->size());
  if (first_n == this->size()) {
    target->set_docids_no_checks(this->docids()->Copy());
  } else {
    target->set_docids_no_checks(make_unique<VariableLengthDocidCollection>(
        VariableLengthDocidCollection::CreateWithEmptyDocids(first_n)));
  }
  target->data_.insert(target->data_.begin(), data_.begin(),
                       data_.begin() + first_n * stride_);
}

template <typename T>
DatapointPtr<T> SparseDataset<T>::operator[](size_t i) const {
  DCHECK_LT(i, this->size());
  auto low_level_result = repr_.Get(i);
  return MakeDatapointPtr(low_level_result.indices, low_level_result.values,
                          low_level_result.nonzero_entries,
                          this->dimensionality());
}

template <typename T>
void SparseDataset<T>::Prefetch(size_t i) const {
  DCHECK_LT(i, this->size());
  repr_.Prefetch(i);
}

SCANN_INSTANTIATE_TYPED_CLASS(extern, TypedDataset);
SCANN_INSTANTIATE_TYPED_CLASS(extern, SparseDataset);
SCANN_INSTANTIATE_TYPED_CLASS(extern, DenseDataset);

}  // namespace research_scann

#endif
