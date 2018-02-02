//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief CUBLAS implementation of the dot product
 */

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Compute the batch_outer product of a and b and store the result in c
 * \param a The lhs expression
 * \param b The rhs expression
 * \param c The output expression
 */
template <typename A, typename B, cpp_enable_iff(all_dma<A, B> && all_single_precision<A, B>)>
float dot(const A& a, const B& b) {
    decltype(auto) handle = start_cublas();

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();

    float prod = 0.0;
    cublas_check(cublasSdot(handle.get(), etl::size(a), a.gpu_memory(), 1, b.gpu_memory(), 1, &prod));
    return prod;
}

/*!
 * \copydoc batch_outer
 */
template <typename A, typename B, cpp_enable_iff(all_dma<A, B> && all_double_precision<A, B>)>
double dot(const A& a, const B& b) {
    decltype(auto) handle = start_cublas();

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();

    double prod = 0.0;
    cublas_check(cublasDdot(handle.get(), etl::size(a), a.gpu_memory(), 1, b.gpu_memory(), 1, &prod));
    return prod;
}

/*!
 * \copydoc dot
 */
template <typename A, typename B, cpp_enable_iff(!is_dma<A>)>
value_t<A> dot(const A& /*a*/, const B& /*b*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

#else

/*!
 * \copydoc batch_outer
 */
template <typename A, typename B>
value_t<A> dot(const A& /*a*/, const B& /*b*/) {
    cpp_unreachable("CUBLAS not enabled/available");
    return 0.0;
}

#endif

} //end of namespace etl::impl::cublas
