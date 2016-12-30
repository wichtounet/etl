//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

#endif

namespace etl {

namespace impl {

namespace cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Compute the sum of a
 * \param a The lhs expression
 */
template <typename A, cpp_enable_if(all_dma<A>::value && all_single_precision<A>::value)>
float sum(const A& a) {
    decltype(auto) handle = start_cublas();

    auto ones = etl::impl::cuda::cuda_allocate_only<float>(etl::size(a));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

    // Set a vector of float one... :(
    float alpha = 1.0;
    int32_t alpha_bits = *reinterpret_cast<int32_t*>(&alpha);
    cuMemsetD32(CUdeviceptr(ones.get()), alpha_bits, etl::size(a));

#pragma GCC diagnostic pop

    auto a_gpu = a.direct();

    a_gpu.ensure_gpu_up_to_date();

    float prod = 0.0;
    cublas_check(cublasSdot(handle.get(), etl::size(a), a_gpu.gpu_memory(), 1, ones.get(), 1, &prod));
    return prod;
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_if(all_dma<A>::value && all_double_precision<A>::value)>
double sum(const A& a) {
    decltype(auto) handle = start_cublas();

    //TODO Need to do much better than that!

    etl::dyn_vector<value_t<A>> ones(etl::size(a), 1.0);

    auto a_gpu = a.direct();
    auto b_gpu = ones.direct();

    a_gpu.ensure_gpu_up_to_date();
    b_gpu.ensure_gpu_up_to_date();

    double prod = 0.0;
    cublas_check(cublasDdot(handle.get(), etl::size(a), a_gpu.gpu_memory(), 1, b_gpu.gpu_memory(), 1, &prod));
    return prod;
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_if(!all_dma<A>::value)>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

#else

/*!
 * \copydoc sum
 */
template <typename A>
void sum(const A& /*a*/) {
    cpp_unreachable("CUBLAS not enabled/available");
}

#endif

} //end of namespace cublas
} //end of namespace impl
} //end of namespace etl
