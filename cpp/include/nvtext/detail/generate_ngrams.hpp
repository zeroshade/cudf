/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#pragma once

#include <nvtext/generate_ngrams.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace nvtext {
namespace detail {

/**
 * @copydoc hash_character_ngrams(cudf::strings_column_view const&,
 * cudf::size_type, rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for allocating/copying device memory and launching kernels
 */
std::unique_ptr<cudf::column> hash_character_ngrams(cudf::strings_column_view const& strings,
                                                    cudf::size_type ngrams,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace nvtext
