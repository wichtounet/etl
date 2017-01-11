//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of scalar operations
 */

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h" //For ddot/sdot
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_single_precision<T>::value)>
void scalar_add(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    float fake_array = 1.0;
    cblas_saxpy(size(lhs), rhs, &fake_array, 0, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_double_precision<T>::value)>
void scalar_add(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    double fake_array = 1.0;
    cblas_daxpy(size(lhs), rhs, &fake_array, 0, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(!all_floating<T>::value)>
void scalar_add(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_single_precision<T>::value)>
void scalar_sub(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    float fake_array = -1.0;
    cblas_saxpy(size(lhs), rhs, &fake_array, 0, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_double_precision<T>::value)>
void scalar_sub(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    double fake_array = -1.0;
    cblas_daxpy(size(lhs), rhs, &fake_array, 0, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(!all_floating<T>::value)>
void scalar_sub(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_single_precision<T>::value)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    cblas_sscal(size(lhs), rhs, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_double_precision<T>::value)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    cblas_dscal(size(lhs), rhs, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(!all_floating<T>::value)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    cpp_unused(lhs);
    cpp_unused(rhs);
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_single_precision<T>::value)>
void scalar_div(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    cblas_sscal(size(lhs), 1.0f / rhs, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(is_double_precision<T>::value)>
void scalar_div(T&& lhs, value_t<T> rhs) {
    lhs.ensure_cpu_up_to_date();

    cblas_dscal(size(lhs), 1.0 / rhs, lhs.memory_start(), 1);

    lhs.invalidate_gpu();
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename T, cpp_enable_if(!all_floating<T>::value)>
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
    cpp_unreachable("CBLAS not available/enabled");
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
    cpp_unreachable("CBLAS not available/enabled");
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
    cpp_unreachable("CBLAS not available/enabled");
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
    cpp_unreachable("CBLAS not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
