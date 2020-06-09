//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Vectorized implementation of the transpose operation
 */

#pragma once

namespace etl::impl::vec {

template <typename V, typename T>
inline void transpose_block_4x4_kernel(size_t N, size_t M, const T* A2, T* C2, size_t i2, size_t j2) {
    C2[(j2 + 0) * N + (i2 + 0)] = A2[(i2 + 0) * M + (j2 + 0)];
    C2[(j2 + 1) * N + (i2 + 0)] = A2[(i2 + 0) * M + (j2 + 1)];
    C2[(j2 + 2) * N + (i2 + 0)] = A2[(i2 + 0) * M + (j2 + 2)];
    C2[(j2 + 3) * N + (i2 + 0)] = A2[(i2 + 0) * M + (j2 + 3)];

    C2[(j2 + 0) * N + (i2 + 1)] = A2[(i2 + 1) * M + (j2 + 0)];
    C2[(j2 + 1) * N + (i2 + 1)] = A2[(i2 + 1) * M + (j2 + 1)];
    C2[(j2 + 2) * N + (i2 + 1)] = A2[(i2 + 1) * M + (j2 + 2)];
    C2[(j2 + 3) * N + (i2 + 1)] = A2[(i2 + 1) * M + (j2 + 3)];

    C2[(j2 + 0) * N + (i2 + 2)] = A2[(i2 + 2) * M + (j2 + 0)];
    C2[(j2 + 1) * N + (i2 + 2)] = A2[(i2 + 2) * M + (j2 + 1)];
    C2[(j2 + 2) * N + (i2 + 2)] = A2[(i2 + 2) * M + (j2 + 2)];
    C2[(j2 + 3) * N + (i2 + 2)] = A2[(i2 + 2) * M + (j2 + 3)];

    C2[(j2 + 0) * N + (i2 + 3)] = A2[(i2 + 3) * M + (j2 + 0)];
    C2[(j2 + 1) * N + (i2 + 3)] = A2[(i2 + 3) * M + (j2 + 1)];
    C2[(j2 + 2) * N + (i2 + 3)] = A2[(i2 + 3) * M + (j2 + 2)];
    C2[(j2 + 3) * N + (i2 + 3)] = A2[(i2 + 3) * M + (j2 + 3)];
}

#ifdef __SSE3__
// sse_vec will only be defined if __SSE3__is enabled

// SSE Version optimized for float
template <>
inline void transpose_block_4x4_kernel<sse_vec>(size_t N, size_t M, const float* A2, float* C2, size_t i2, size_t j2) {
    using vec_type = sse_vec;

    auto r1 = vec_type::loadu(A2 + (i2 + 0) * M + j2);
    auto r2 = vec_type::loadu(A2 + (i2 + 1) * M + j2);
    auto r3 = vec_type::loadu(A2 + (i2 + 2) * M + j2);
    auto r4 = vec_type::loadu(A2 + (i2 + 3) * M + j2);

    _MM_TRANSPOSE4_PS(r1.value, r2.value, r3.value, r4.value);

    vec_type::storeu(C2 + (j2 + 0) * N + i2, r1);
    vec_type::storeu(C2 + (j2 + 1) * N + i2, r2);
    vec_type::storeu(C2 + (j2 + 2) * N + i2, r3);
    vec_type::storeu(C2 + (j2 + 3) * N + i2, r4);
}

#endif

template <typename V, typename A, typename C>
void transpose_impl(const A& a, C&& c) {
    const size_t N = etl::dim<0>(a);
    const size_t M = etl::dim<1>(a);

    const auto* A2 = a.memory_start();
    auto*       C2 = c.memory_start();

    if constexpr (decay_traits<A>::storage_order == order::RowMajor) {
        constexpr size_t block_size        = 16;
        constexpr size_t kernel_block_size = 4;


        auto batch_fun_i = [&](const size_t ifirst, const size_t ilast) {
            size_t i = ifirst;

            for (; i + block_size - 1 < ilast; i += block_size) {
                size_t j = 0;

                // Compute blocks of 16x16
                for (; j + block_size - 1 < M; j += block_size) {
                    for (size_t i2 = i; i2 < i + block_size; i2 += kernel_block_size) {
                        for (size_t j2 = j; j2 < j + block_size; j2 += kernel_block_size) {
                            transpose_block_4x4_kernel<V>(N, M, A2, C2, i2, j2);
                        }
                    }
                }

                // Compute blocks of 16x4
                for (; j + kernel_block_size - 1 < M; j += kernel_block_size) {
                    for (size_t i2 = i; i2 < i + block_size; i2 += kernel_block_size) {
                        transpose_block_4x4_kernel<V>(N, M, A2, C2, i2, j);
                    }
                }

                // Compute the left overs
                for (; j < M; ++j) {
                    for (size_t i2 = i; i2 < i + block_size; ++i2) {
                        C2[j * N + i2] = A2[i2 * M + j];
                    }
                }
            }

            for (; i + kernel_block_size - 1 < ilast; i += kernel_block_size) {
                size_t j = 0;

                // Compute blocks of 4x4
                for (; j + kernel_block_size - 1 < M; j += kernel_block_size) {
                    transpose_block_4x4_kernel<V>(N, M, A2, C2, i, j);
                }

                // Compute the leftovers
                for (; j < M; ++j) {
                    for (size_t i2 = i; i2 < i + kernel_block_size; ++i2) {
                        C2[j * N + i] = A2[i2 * M + j];
                    }
                }
            }

            for (; i < ilast; ++i) {
                size_t j = 0;

                for (; j < M; ++j) {
                    C2[j * N + i] = A2[i * M + j];
                }
            }
        };

        engine_dispatch_1d(batch_fun_i, 0, N, engine_select_parallel(N, threads * 2 * block_size));
    } else {
        //TODO Optimize properly for column major
        for (size_t j = 0; j < M; ++j) {
            for (size_t i = 0; i < N; ++i) {
                C2[i * M + j] = A2[j * N + i];
            }
        }
    }
}

template <typename A, typename C>
void transpose([[maybe_unused]] A&& a, [[maybe_unused]] C&& c) {
    if constexpr (all_vectorizable<vector_mode, A, C> && sse3_enabled) {
#ifdef __SSE3__
// sse_vec will only be defined if __SSE3__is enabled
        transpose_impl<sse_vec>(a, c);
#endif
    } else {
        cpp_unreachable("Invalid call to vec::batch_outer");
    }
}

} //end of namespace etl::impl::vec
