//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Traits indicating if vectorized 1D convolution is possible
 * for the given configuration.
 *
 * A 1D convolution can be optimized if vectorization is enabled,
 * vectorization of algorithms is enabled, all the types are the
 * same and all the types are vectorizable.
 *
 * \param V The vector mode
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 * \param C The type of the output matrix
 */
template <vector_mode_t V, typename I, typename K, typename C>
constexpr bool conv1_possible = vec_enabled&& vectorize_impl&& all_homogeneous<I, K, C>&& all_vectorizable<V, I, K, C>;

/*!
 * \brief Traits indicating if vectorized 2D convolution is possible
 * for the given configuration.
 *
 * A 2D convolution can be optimized if vectorization is enabled,
 * vectorization of algorithms is enabled, all the types are the
 * same and all the types are vectorizable.
 *
 * \param V The vector mode
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 * \param C The type of the output matrix
 */
template <vector_mode_t V, typename I, typename K, typename C>
constexpr bool conv2_possible = vec_enabled&& vectorize_impl&& all_homogeneous<I, K, C>&& all_vectorizable<V, I, K, C>&& all_row_major<I, K, C>;

namespace detail {

/*!
 * \brief Indicates if SSE should be preferred for the given kernel size
 * \param n The kernel size
 * \return true if SSE should be preferred, false otherwise.
 */
template <typename T>
constexpr bool prefer_sse(const size_t n) {
    return !avx_enabled || (sse3_enabled && (std::is_same<T, float>::value ? (n % 4 < n % 8) : (n % 2 < n % 4)));
}

/*!
 * \brief Pad the given input into the given output matrix
 * \param in The input matrix
 * \param out The output matrix
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I, typename C>
void pad_2d_input(const I& in, C& out, size_t p1, size_t p2) {
    in.ensure_cpu_up_to_date();

    auto in_m  = in.memory_start();
    auto out_m = out.memory_start();

    for (size_t i = 0; i < etl::dim<0>(in); ++i) {
        direct_copy_n(in_m + i * etl::dim<1>(in), out_m + (i + p1) * etl::dim<1>(out) + p2, etl::dim<1>(in));
    }

    out.invalidate_gpu();
}

#ifdef __AVX__
/*!
 * \brief Safe AVX vectorization utility.
 *
 * If AVX is enabled, this is directly AVX, otherwise, no vectorization.
 */
using safe_avx_vec = avx_vec;
#else
/*!
 * \brief Safe AVX vectorization utility.
 *
 * If AVX is enabled, this is directly AVX, otherwise, no vectorization.
 */
using safe_avx_vec = no_vec;
#endif

#ifdef __SSE3__
/*!
 * \brief Safe SSE vectorization utility.
 *
 * If AVX is enabled, this is directly SSE, otherwise, no vectorization.
 */
using safe_sse_vec = sse_vec;
#else
/*!
 * \brief Safe SSE vectorization utility.
 *
 * If AVX is enabled, this is directly SSE, otherwise, no vectorization.
 */
using safe_sse_vec = no_vec;
#endif

} //end of namespace detail
} //end of namespace etl::impl::vec

// Note: Valid must be included first!

#include "etl/impl/common/conv.hpp"
#include "etl/impl/vec/conv_valid_1d.hpp"
#include "etl/impl/vec/conv_valid_2d.hpp"
#include "etl/impl/vec/conv_valid_4d.hpp"
#include "etl/impl/vec/conv_full.hpp"
#include "etl/impl/vec/conv_same.hpp"
