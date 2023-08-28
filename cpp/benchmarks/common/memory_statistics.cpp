/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "memory_statistics.hpp"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>

#include <numeric>

uint64_t required_bytes(const cudf::column_view& column)
{
  uint64_t read_bytes = column.size() * cudf::size_of(column.type());
  if (column.nullable()) { read_bytes += cudf::bitmask_allocation_size_bytes(column.size()); }

  return read_bytes;
}

uint64_t required_bytes(const cudf::table_view& table)
{
  return std::accumulate(table.begin(), table.end(), 0, [](uint64_t acc, const auto& col) {
    return acc + required_bytes(col);
  });
}

uint64_t required_bytes(
  const cudf::host_span<cudf::groupby::aggregation_result>& aggregation_results)
{
  uint64_t read_bytes = 0;

  for (auto const& aggregation : aggregation_results) {  // vector of aggregation results
    for (auto const& col : aggregation.results) {        // vector of columns per result
      read_bytes += required_bytes(col->view());
    }
  }

  return read_bytes;
}
