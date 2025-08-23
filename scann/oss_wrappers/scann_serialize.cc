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

#include "scann/oss_wrappers/scann_serialize.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "absl/base/casts.h"
#include "absl/base/internal/endian.h"
#include "absl/strings/string_view.h"

namespace research_scann {
namespace strings {
namespace {

// 将IEEE754浮点数转换为无符号整数表示（用于序列化）
template <typename UintType, typename FloatType>
UintType UintFromIEEE754(FloatType f) {
  const UintType n = absl::bit_cast<UintType>(f);
  const UintType sign_bit = ~(~static_cast<UintType>(0) >> 1);
  if ((n & sign_bit) == 0) return n + sign_bit;
  return 0 - n;
}

// 将无符号整数还原为IEEE754浮点数（用于反序列化）
template <typename FloatType, typename UintType>
FloatType IEEE754FromUint(UintType n) {
  const UintType sign_bit = ~(~static_cast<UintType>(0) >> 1);
  if (n & sign_bit) {
    n -= sign_bit;
  } else {
    n = 0 - n;
  }
  return absl::bit_cast<FloatType>(n);
}
}  // namespace

// uint32转为序列化key字符串
inline std::string Uint32ToKey(uint32_t u32) {
  std::string key;
  KeyFromUint32(u32, &key);
  return key;
}

// int32转为序列化key字符串
std::string Int32ToKey(int32_t i32) { return Uint32ToKey(i32); }

// uint64转为序列化key字符串
inline std::string Uint64ToKey(uint64_t u64) {
  std::string key;
  KeyFromUint64(u64, &key);
  return key;
}

// uint32转key字符串，处理字节序
inline void KeyFromUint32(uint32_t u32, std::string* key) {
  uint32_t norder = absl::ghtonl(u32);
  key->assign(reinterpret_cast<const char*>(&norder), sizeof(norder));
}

// uint64转key字符串，处理字节序
inline void KeyFromUint64(uint64_t u64, std::string* key) {
  uint64_t norder = absl::ghtonll(u64);
  key->assign(reinterpret_cast<const char*>(&norder), sizeof(norder));
}

// key字符串还原为uint32，处理字节序
inline uint32_t KeyToUint32(absl::string_view key) {
  uint32_t value;
  memcpy(&value, key.data(), sizeof(value));
  return absl::gntohl(value);
}

// key字符串还原为int32
int32_t KeyToInt32(absl::string_view key) { return KeyToUint32(key); }

// key字符串还原为uint64，处理字节序
inline uint64_t KeyToUint64(absl::string_view key) {
  uint64_t value;
  memcpy(&value, key.data(), sizeof(value));
  return absl::gntohll(value);
}

// float转key字符串，先转IEEE754再转uint32
void KeyFromFloat(float x, std::string* key) {
  const uint32_t n = UintFromIEEE754<uint32_t>(x);
  KeyFromUint32(n, key);
}

// float转为序列化key字符串
std::string FloatToKey(float x) {
  std::string key;
  KeyFromFloat(x, &key);
  return key;
}

// key字符串还原为float
float KeyToFloat(absl::string_view key) {
  const uint32_t n = KeyToUint32(key);
  return IEEE754FromUint<float>(n);
}

}  // namespace strings
}  // namespace research_scann
