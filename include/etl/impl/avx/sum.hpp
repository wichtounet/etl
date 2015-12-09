//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*
 * \file
 * \brief AVX implementation of "sum" reduction
 */

namespace etl {

namespace impl {

namespace avx {

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)

inline double sum(const double* in, std::size_t n){
    double tmp_res[4] __attribute__((aligned(32)));

    __m256d ymm1;
    __m256d ymm2;

    //Compute the sum, 4 doubles at a time

    ymm2 = _mm256_setzero_pd();

    for (std::size_t i = 0; i + 3 < n; i += 4) {
        ymm1 = _mm256_loadu_pd(in + i);
        ymm2 = _mm256_add_pd(ymm2, ymm1);
    }

    _mm256_store_pd(tmp_res, ymm2);

    auto acc = tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];

    if (n % 4) {
        auto rem = n % 4;
        for(std::size_t i = n - rem; i < n; ++i){
            acc += in[i];
        }
    }

    return acc;
}

inline float sum(const float* in, std::size_t n){
    float tmp_res[8] __attribute__((aligned(32)));

    __m256 ymm1;
    __m256 ymm2;

    //Compute the sum, 8 floats at a time

    ymm2 = _mm256_setzero_ps();

    for (std::size_t i = 0; i + 7 < n; i += 8) {
        ymm1 = _mm256_loadu_ps(in + i);
        ymm2 = _mm256_add_ps(ymm2, ymm1);
    }

    _mm256_store_ps(tmp_res, ymm2);

    auto acc = tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];

    if (n % 8) {
        auto rem = n % 8;
        for(std::size_t i = n - rem; i < n; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template <typename I, cpp_enable_if(has_direct_access<I>::value)>
value_t<I> sum(const I& input) {
    return sum(input.memory_start(), etl::size(input));
}

template <typename I, cpp_disable_if(has_direct_access<I>::value)>
value_t<I> sum(const I& /*input*/) {
    cpp_unreachable("AVX not available/enabled");
    return value_t<I>(0.0);
}

#else

template <typename I>
double sum(const I& /*input*/) {
    cpp_unreachable("AVX not available/enabled");
    return 0.0;
}

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
