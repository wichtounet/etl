//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

template<typename E>
double dsum_kernel(const E& in, std::size_t first, std::size_t last){
    double acc = 0.0;

    while(first < last && first % 4 != 0){
        acc += in[first++];
    }

    double tmp_res[4] __attribute__((aligned(32)));

    __m256d ymm1;
    __m256d ymm2;

    //Compute the sum, 4 doubles at a time

    ymm2 = _mm256_setzero_pd();

    for (std::size_t i = first; i + 3 < last; i += 4) {
        ymm1 = in.template load<avx_vec>(i);
        ymm2 = _mm256_add_pd(ymm2, ymm1);
    }

    _mm256_store_pd(tmp_res, ymm2);

    acc += tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];

    auto n = last - first;
    if (n % 4) {
        auto rem = n % 4;
        for(std::size_t i = last - rem; i < last; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template<typename E>
float ssum_kernel(const E& in, std::size_t first, std::size_t last){
    float acc = 0.0;

    while(first < last && first % 8 != 0){
        acc += in[first++];
    }

    float tmp_res[8] __attribute__((aligned(32)));

    __m256 ymm1;
    __m256 ymm2;

    //Compute the sum, 8 floats at a time

    ymm2 = _mm256_setzero_ps();

    for (std::size_t i = first; i + 7 < last; i += 8) {
        ymm1 = in.template load<avx_vec>(i);
        ymm2 = _mm256_add_ps(ymm2, ymm1);
    }

    _mm256_store_ps(tmp_res, ymm2);

    acc += tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];

    auto n = last - first;
    if (n % 8) {
        auto rem = n % 8;
        for(std::size_t i = last - rem; i < last; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template <typename I, cpp_enable_if(all_single_precision<I>::value, decay_traits<I>::vectorizable)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return ssum_kernel(input, first, last);
}

template <typename I, cpp_enable_if(all_double_precision<I>::value, decay_traits<I>::vectorizable)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return dsum_kernel(input, first, last);
}

template <typename I, cpp_disable_if(decay_traits<I>::vectorizable)>
value_t<I> sum(const I& /*input*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
    return value_t<I>(0.0);
}

#else

template <typename I>
double sum(const I& /*input*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
    return 0.0;
}

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
