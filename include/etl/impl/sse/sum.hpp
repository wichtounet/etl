//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*
 * \file
 * \brief SSE implementation of "sum" reduction
 */

namespace etl {

namespace impl {

namespace sse {

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)

template<typename E>
double dsum_kernel(const E& in, std::size_t n){
    double tmp_res[2] __attribute__((aligned(16)));

    __m128d ymm1;
    __m128d ymm2;

    //Compute the sum, 2 doubles at a time

    ymm2 = _mm_setzero_pd();

    for (std::size_t i = 0; i + 1 < n; i += 2) {
        ymm1 = in.template load<sse_vec>(i);
        ymm2 = _mm_add_pd(ymm2, ymm1);
    }

    _mm_store_pd(tmp_res, ymm2);

    auto acc = tmp_res[0] + tmp_res[1];

    if (n % 2) {
        acc += in[n - 1];
    }

    return acc;
}

template<typename E>
float ssum_kernel(const E& in, std::size_t n){
    float tmp_res[4] __attribute__((aligned(16)));

    __m128 ymm1;
    __m128 ymm2;

    //Compute the sum, 4 floats at a time

    ymm2 = _mm_setzero_ps();

    for (std::size_t i = 0; i + 3 < n; i += 4) {
        ymm1 = in.template load<sse_vec>(i);
        ymm2 = _mm_add_ps(ymm2, ymm1);
    }

    _mm_store_ps(tmp_res, ymm2);

    auto acc = tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];

    if (n % 4) {
        auto rem = n % 4;
        for(std::size_t i = n - rem; i < n; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template <typename I, cpp_enable_if(all_single_precision<I>::value && has_direct_access<I>::value)>
value_t<I> sum(const I& input) {
    return ssum_kernel(input, etl::size(input));
}

template <typename I, cpp_enable_if(all_double_precision<I>::value && has_direct_access<I>::value)>
value_t<I> sum(const I& input) {
    return dsum_kernel(input, etl::size(input));
}

template <typename I, cpp_disable_if(has_direct_access<I>::value)>
value_t<I> sum(const I& /*input*/) {
    cpp_unreachable("SSE not available/enabled");
    return value_t<I>(0.0);
}

#else

template <typename I>
double sum(const I& /*input*/) {
    cpp_unreachable("SSE not available/enabled");
    return 0.0;
}

#endif

} //end of namespace sse
} //end of namespace impl
} //end of namespace etl
