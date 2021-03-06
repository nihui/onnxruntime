// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/endian_utils.h"

#include <cassert>
#include <cstring>

#include "core/framework/endian.h"

namespace onnxruntime {
namespace utils {

namespace {

// analogous to std::reverse_copy
template <typename BidirIt, typename OutputIt>
OutputIt ReverseCopy(BidirIt first, BidirIt last, OutputIt d_first) {
  while (last != first) {
    --last;
    *d_first = *last;
    ++d_first;
  }
  return d_first;
}

}  // namespace

void SwapByteOrderCopy(
    size_t element_size_in_bytes,
    gsl::span<const char> source_bytes, gsl::span<char> destination_bytes) {
  assert(element_size_in_bytes > 0);
  assert(source_bytes.size_bytes() % element_size_in_bytes == 0);
  assert(source_bytes.size_bytes() == destination_bytes.size_bytes());

  for (size_t element_offset = 0, element_offset_end = source_bytes.size_bytes();
       element_offset < element_offset_end;
       element_offset += element_size_in_bytes) {
    const auto source_element_bytes =
        source_bytes.subspan(element_offset, element_size_in_bytes);
    const auto dest_element_bytes =
        destination_bytes.subspan(element_offset, element_size_in_bytes);
    ReverseCopy(
        source_element_bytes.data(),
        source_element_bytes.data() + source_element_bytes.size_bytes(),
        dest_element_bytes.data());
  }
}

namespace detail {

void CopyLittleEndian(size_t element_size_in_bytes, gsl::span<const char> source_bytes, gsl::span<char> destination_bytes) {
  if (endian::native == endian::little) {
    std::memcpy(destination_bytes.data(), source_bytes.data(), source_bytes.size_bytes());
  } else {
    SwapByteOrderCopy(element_size_in_bytes, source_bytes, destination_bytes);
  }
}

}  // namespace detail

}  // namespace utils
}  // namespace onnxruntime
