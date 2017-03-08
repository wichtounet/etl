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
 * \brief Add the rhs to lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_single_precision<L, R>::value)>
void direct_add(L&& lhs, R&& rhs) {
    decltype(auto) handle = start_cublas();

    float alpha = 1.0;
    cublasSaxpy(handle.get(), size(lhs), &alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Add the rhs to lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_double_precision<L, R>::value)>
void direct_add(L&& lhs, R&& rhs) {
    decltype(auto) handle = start_cublas();

    double alpha = 1.0;
    cublasDaxpy(handle.get(), size(lhs), &alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Add the rhs to lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(!all_floating<L, R>::value)>
void direct_add(L&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not possible on this type");
}

/*!
 * \brief Subtract the rhs from lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_single_precision<L, R>::value)>
void direct_sub(L&& lhs, R&& rhs) {
    decltype(auto) handle = start_cublas();

    float alpha = -1.0;
    cublasSaxpy(handle.get(), size(lhs), &alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Subtract the rhs from lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_double_precision<L, R>::value)>
void direct_sub(L&& lhs, R&& rhs) {
    decltype(auto) handle = start_cublas();

    double alpha = -1.0;
    cublasDaxpy(handle.get(), size(lhs), &alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();
}

/*!
 * \brief Subtract the rhs from lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(!all_floating<L, R>::value)>
void direct_sub(L&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not possible on this type");
}

/*!
 * \brief Multiply the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_single_precision<L, R>::value)>
bool direct_mul(L&& lhs, R&& rhs) {
#ifdef EGBLAS_HAS_SAXMY
    float alpha = 1.0;
    egblas_saxmy(size(lhs), alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();

    return true;
#else
    return false;
#endif
}

/*!
 * \brief Multiply the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_double_precision<L, R>::value)>
bool direct_mul(L&& lhs, R&& rhs) {
#ifdef EGBLAS_HAS_DAXMY
    float alpha = 1.0;
    egblas_daxmy(size(lhs), alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();

    return true;
#else
    return false;
#endif
}

/*!
 * \brief Multiply the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(!all_floating<L, R>::value)>
bool direct_mul(L&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not possible on this type");
    return false;
}

/*!
 * \brief Divide the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_single_precision<L, R>::value)>
bool direct_div(L&& lhs, R&& rhs) {
#ifdef EGBLAS_HAS_SAXDY
    float alpha = 1.0;
    egblas_saxdy(size(lhs), alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();

    return true;
#else
    return false;
#endif
}

/*!
 * \brief Divide the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(all_double_precision<L, R>::value)>
bool direct_div(L&& lhs, R&& rhs) {
#ifdef EGBLAS_HAS_DAXDY
    float alpha = 1.0;
    egblas_daxdy(size(lhs), alpha, rhs.gpu_memory(), 1, lhs.gpu_memory(), 1);

    lhs.invalidate_cpu();

    return true;
#else
    return false;
#endif
}

/*!
 * \brief Divide the lhs by rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R, cpp_enable_if(!all_floating<L, R>::value)>
bool direct_div(L&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not possible on this type");
    return false;
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Add the rhs to lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R>
void direct_add(T&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

/*!
 * \brief Subtract the rhs from lhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R>
void direct_sub(T&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
}

/*!
 * \brief Multiply the lhs by the rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R>
bool direct_mul(T&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
    return false;
}

/*!
 * \brief Divide the lhs by the rhs
 * \param lhs The lhs matrix
 * \param rhs The rhs matrix
 */
template <typename L, typename R>
bool direct_div(T&& lhs, R&& rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
    cpp_unreachable("CUBLAS not available/enabled");
    return false;
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cublas
} //end of namespace impl
} //end of namespace etl
