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

    __m256d ymm1;
    __m256d ymm2;

    //Compute the sum, 4 doubles at a time

    ymm2 = _mm256_setzero_pd();

    for (std::size_t i = first; i + 3 < last; i += 4) {
        ymm1 = in.template load<avx_vec>(i);
        ymm2 = _mm256_add_pd(ymm2, ymm1);
    }

    //Horizontal sum of the result
    ymm2 = _mm256_hadd_pd(ymm2, ymm2);
    __m128d sum_low = _mm256_extractf128_pd(ymm2, 0);
    __m128d sum_high = _mm256_extractf128_pd(ymm2, 1);
    __m128d result = _mm_add_pd(sum_low, sum_high);

    acc += _mm_cvtsd_f64(result);

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

    __m256 ymm1;
    __m256 ymm2;

    //Compute the sum, 8 floats at a time

    ymm2 = _mm256_setzero_ps();

    for (std::size_t i = first; i + 7 < last; i += 8) {
        ymm1 = in.template load<avx_vec>(i);
        ymm2 = _mm256_add_ps(ymm2, ymm1);
    }

    // Horizontal sum of the vector...
    __m128 hiQuad = _mm256_extractf128_ps(ymm2, 1);
    __m128 loQuad = _mm256_castps256_ps128(ymm2);
    __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    __m128 loDual = sumQuad;
    __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    __m128 sumDual = _mm_add_ps(loDual, hiDual);
    __m128 lo = sumDual;
    __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    __m128 sum = _mm_add_ss(lo, hi);

    acc += _mm_cvtss_f32(sum);

    auto n = last - first;
    if (n % 8) {
        auto rem = n % 8;
        for(std::size_t i = last - rem; i < last; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template <typename I, cpp_enable_if(all_single_precision<I>::value, decay_traits<I>::template vectorizable<vector_mode_t::AVX>::value)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return ssum_kernel(input, first, last);
}

template <typename I, cpp_enable_if(all_double_precision<I>::value, decay_traits<I>::template vectorizable<vector_mode_t::AVX>::value)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return dsum_kernel(input, first, last);
}

template <typename I, cpp_disable_if(decay_traits<I>::template vectorizable<vector_mode_t::AVX>::value)>
value_t<I> sum(const I& /*input*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
    return value_t<I>(0.0);
}

#else

/*!
 * \brief Compute the of the input in the given range
 * \param input The input expression
 * \param first The beginning of the range
 * \param last The end of the range
 * \return the sum
 */
template <typename I>
double sum(const I& input, std::size_t first, std::size_t last) {
    cpp_unused(input);
    cpp_unused(first);
    cpp_unused(last);
    cpp_unreachable("AVX not available/enabled");
    return 0.0;
}

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
