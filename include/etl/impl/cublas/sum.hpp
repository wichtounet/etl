//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief CUBLAS implementation of the sum product.
 *
 * Unfortunately, CUBLAS has no sum implementation, therefore, it
 * uses a dot product with a vector of 1...
 *
 * This must be improved!
 */

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#ifdef ETL_EGBLAS_MODE
#include "egblas.hpp"
#endif

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Compute the sum of a
 * \param a The lhs expression
 */
template <typename A, cpp_enable_iff(is_gpu_computable<A>&& is_single_precision<A>)>
float sum(const A& a) {
    auto t1 = smart_forward_gpu(a);

    t1.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_SSUM
    return egblas_ssum(t1.gpu_memory(), etl::size(a), 1);
#else
    decltype(auto) handle = start_cublas();

    auto ones = etl::impl::cuda::cuda_allocate_only<float>(etl::size(a));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

    // Set a vector of float one... :(
    float alpha        = 1.0;
    int32_t alpha_bits = *reinterpret_cast<int32_t*>(&alpha);
    cuMemsetD32(CUdeviceptr(ones.get()), alpha_bits, etl::size(a));

#pragma GCC diagnostic pop

    float prod = 0.0;
    cublas_check(cublasSdot(handle.get(), etl::size(a), t1.gpu_memory(), 1, ones.get(), 1, &prod));
    return prod;
#endif
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_iff(is_gpu_computable<A>&& is_double_precision<A>)>
double sum(const A& a) {
    auto t1 = smart_forward_gpu(a);

    t1.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_DSUM
    return egblas_dsum(t1.gpu_memory(), etl::size(a), 1);
#else
    decltype(auto) handle = start_cublas();

    etl::dyn_vector<value_t<A>> ones(etl::size(a), 1.0);

    ones.ensure_gpu_up_to_date();

    double prod = 0.0;
    cublas_check(cublasDdot(handle.get(), etl::size(a), t1.gpu_memory(), 1, ones.gpu_memory(), 1, &prod));
    return prod;
#endif
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_iff(!is_gpu_computable<A> || !is_floating<A>)>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("CUBLAS not enabled/available");
    return 0.0;
}

/*!
 * \brief Compute the sum of a
 * \param a The lhs expression
 */
template <typename A, cpp_enable_iff(is_gpu_computable<A>&& is_single_precision<A>)>
float asum(const A& a) {
    decltype(auto) handle = start_cublas();

    auto t1 = smart_forward_gpu(a);

    t1.ensure_gpu_up_to_date();

    float prod = 0.0;
    cublas_check(cublasSasum(handle.get(), etl::size(a), t1.gpu_memory(), 1, &prod));
    return prod;
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_iff(is_gpu_computable<A>&& is_double_precision<A>)>
double asum(const A& a) {
    decltype(auto) handle = start_cublas();

    auto t1 = smart_forward_gpu(a);

    t1.ensure_gpu_up_to_date();

    double prod = 0.0;
    cublas_check(cublasDasum(handle.get(), etl::size(a), t1.gpu_memory(), 1, &prod));
    return prod;
}

/*!
 * \copydoc asum
 */
template <typename A, cpp_enable_iff(!is_gpu_computable<A> || !is_floating<A>)>
value_t<A> asum(const A& /*a*/) {
    cpp_unreachable("CUBLAS not enabled/available");
    return 0.0;
}

#else

/*!
 * \copydoc sum
 */
template <typename A>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("CUBLAS not enabled/available");
    return 0.0;
}

/*!
 * \copydoc asum
 */
template <typename A>
value_t<A> asum(const A& /*a*/) {
    cpp_unreachable("CUBLAS not enabled/available");
    return 0.0;
}

#endif

} //end of namespace etl::impl::cublas
