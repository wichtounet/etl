//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
double dsum_kernel(const E& in, std::size_t first, std::size_t last){
    double acc = 0.0;

    while(first < last && first % 2 != 0){
        acc += in[first++];
    }

    __m128d ymm1;
    __m128d ymm2;

    //Compute the sum, 2 doubles at a time

    ymm2 = _mm_setzero_pd();

    for (std::size_t i = first; i + 1 < last; i += 2) {
        ymm1 = in.template load<sse_vec>(i);
        ymm2 = _mm_add_pd(ymm2, ymm1);
    }

    //Horizontal sum of the result
    ymm2 = _mm_hadd_pd(ymm2, ymm2);

    acc += _mm_cvtsd_f64(ymm2);

    auto n = last - first;
    if (n % 2) {
        acc += in[last - 1];
    }

    return acc;
}

template<typename E>
float ssum_kernel(const E& in, std::size_t first, std::size_t last){
    float acc = 0.0;

    while(first < last && first % 4 != 0){
        acc += in[first++];
    }

    __m128 ymm1;
    __m128 ymm2;

    //Compute the sum, 4 floats at a time

    ymm2 = _mm_setzero_ps();

    for (std::size_t i = first; i + 3 < last; i += 4) {
        ymm1 = in.template load<sse_vec>(i);
        ymm2 = _mm_add_ps(ymm2, ymm1);
    }

    //Horizontal sum of the result
    ymm2 = _mm_hadd_ps(ymm2, ymm2);
    ymm2 = _mm_hadd_ps(ymm2, ymm2);

    acc += _mm_cvtss_f32(ymm2);

    auto n = last - first;
    if (n % 4) {
        auto rem = n % 4;
        for(std::size_t i = last - rem; i < last; ++i){
            acc += in[i];
        }
    }

    return acc;
}

template <typename I, cpp_enable_if(all_single_precision<I>::value, decay_traits<I>::template vectorizable<vector_mode_t::SSE3>::value)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return ssum_kernel(input, first, last);
}

template <typename I, cpp_enable_if(all_double_precision<I>::value, decay_traits<I>::template vectorizable<vector_mode_t::SSE3>::value)>
value_t<I> sum(const I& input, std::size_t first, std::size_t last) {
    return dsum_kernel(input, first, last);
}

template <typename I, cpp_disable_if(decay_traits<I>::template vectorizable<vector_mode_t::SSE3>::value)>
value_t<I> sum(const I& /*input*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("SSE not available/enabled");
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
    cpp_unreachable("SSE not available/enabled");
    return 0.0;
}

#endif

} //end of namespace sse
} //end of namespace impl
} //end of namespace etl
