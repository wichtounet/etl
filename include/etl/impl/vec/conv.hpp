//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {
namespace impl {
namespace vec {
namespace detail {

/*!
 * \brief Indicates if SSE should be preferred for the given kernel size
 * \param n The kernel size
 * \return true if SSE should be preferred, false otherwise.
 */
template <typename T>
constexpr bool prefer_sse(const size_t n) {
    return !avx_enabled || (sse3_enabled && (std::is_same<T, float>::value
                                                 ? (n % 4 < n % 8)
                                                 : (n % 2 < n % 4)));
}

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
using safe_avx_vec = avx_vec;
#else
using safe_avx_vec = no_vec;
#endif

#ifdef __SSE3__
using safe_sse_vec = sse_vec;
#else
using safe_sse_vec = no_vec;
#endif

} //end of namespace detail
} //end of namespace vec
} //end of namespace impl
} //end of namespace etl

// Note: Valid must be included first!

#include "etl/impl/common/conv.hpp"
#include "etl/impl/vec/conv_valid_1d.hpp"
#include "etl/impl/vec/conv_valid_2d.hpp"
#include "etl/impl/vec/conv_valid_4d.hpp"
#include "etl/impl/vec/conv_full.hpp"
#include "etl/impl/vec/conv_same.hpp"
