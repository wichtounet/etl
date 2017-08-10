//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief CUBLAS implementation of scalar operations
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
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_add(T&& lhs, float rhs) {
    lhs.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_SADD
    egblas_scalar_sadd(lhs.gpu_memory(), size(lhs), 1, rhs);
#else
    decltype(auto) handle = start_cublas();

    // Note: This is immensely retarded...

    auto ones = etl::impl::cuda::cuda_allocate_only<float>(etl::size(lhs));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

    // Set a vector of float one... :(
    float alpha = 1.0;
    int32_t alpha_bits = *reinterpret_cast<int32_t*>(&alpha);
    cuMemsetD32(CUdeviceptr(ones.get()), alpha_bits, etl::size(lhs));

#pragma GCC diagnostic pop

    cublasSaxpy(handle.get(), size(lhs), &rhs, ones.get(), 1, lhs.gpu_memory(), 1);
#endif

    lhs.invalidate_cpu();
}

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_add(T&& lhs, double rhs) {
    lhs.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_DADD
    egblas_scalar_dadd(lhs.gpu_memory(), size(lhs), 1, rhs);
#else
    decltype(auto) handle = start_cublas();

    // Note: This is immensely retarded...

    etl::dyn_vector<value_t<T>> ones(etl::size(lhs), 1.0);

    ones.ensure_gpu_up_to_date();

    cublasDaxpy(handle.get(), size(lhs), &rhs, ones.gpu_memory(), 1, lhs.gpu_memory(), 1);
#endif

    lhs.invalidate_cpu();
}

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_iff(!is_floating<T>)>
void scalar_add(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_sub(T&& lhs, float rhs) {
    lhs.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_SADD
    egblas_scalar_sadd(lhs.gpu_memory(), size(lhs), 1, -rhs);
#else
    decltype(auto) handle = start_cublas();

    // Note: This is immensely retarded...

    auto ones = etl::impl::cuda::cuda_allocate_only<float>(etl::size(lhs));

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

    // Set a vector of float one... :(
    float alpha = -1.0;
    int32_t alpha_bits = *reinterpret_cast<int32_t*>(&alpha);
    cuMemsetD32(CUdeviceptr(ones.get()), alpha_bits, etl::size(lhs));

#pragma GCC diagnostic pop

    cublasSaxpy(handle.get(), size(lhs), &rhs, ones.get(), 1, lhs.gpu_memory(), 1);
#endif

    lhs.invalidate_cpu();
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_sub(T&& lhs, double rhs) {
    lhs.ensure_gpu_up_to_date();

#ifdef EGBLAS_HAS_DADD
    egblas_scalar_dadd(lhs.gpu_memory(), size(lhs), 1, -rhs);
#else
    decltype(auto) handle = start_cublas();

    // Note: This is immensely retarded...

    etl::dyn_vector<value_t<T>> ones(etl::size(lhs), -1.0);

    ones.ensure_gpu_up_to_date();

    cublasDaxpy(handle.get(), size(lhs), &rhs, ones.gpu_memory(), 1, lhs.gpu_memory(), 1);
#endif

    lhs.invalidate_cpu();
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_iff(!is_floating<T>)>
void scalar_sub(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_mul(T&& lhs, float rhs) {
    decltype(auto) handle = start_cublas();

    lhs.ensure_gpu_up_to_date();

    cublasSscal(handle.get(), size(lhs), &rhs, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_mul(T&& lhs, double rhs) {
    decltype(auto) handle = start_cublas();

    lhs.ensure_gpu_up_to_date();

    cublasDscal(handle.get(), size(lhs), &rhs, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_iff(!is_floating<T>)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_div(T&& lhs, float rhs) {
    decltype(auto) handle = start_cublas();

    lhs.ensure_gpu_up_to_date();

    float rhs_down = 1.0f / rhs;
    cublasSscal(handle.get(), size(lhs), &rhs_down, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_div(T&& lhs, double rhs) {
    decltype(auto) handle = start_cublas();

    lhs.ensure_gpu_up_to_date();

    double rhs_down = 1.0 / rhs;
    cublasDscal(handle.get(), size(lhs), &rhs_down, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_iff(!is_floating<T>)>
void scalar_div(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_add(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_sub(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T>
void scalar_div(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cublas
} //end of namespace impl
} //end of namespace etl
