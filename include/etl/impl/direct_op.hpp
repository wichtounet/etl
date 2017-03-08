//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementations of direct compound operations
 *
 * This module is using GPU BLAS and BLAS operation when possible
 * for performance.
 */

#pragma once

//Include the implementations
#include "etl/impl/cublas/direct_op.hpp"

namespace etl {

namespace detail {

// TODO: It is probably interesting to use BLAS too here

/*!
 * \brief Add the RHS matrix to the LHS matrix
 * \param lhs The left-hand-side matrix
 * \param rhs The right-hand-side matrix
 */
template <typename L, typename R>
bool direct_add(L&& lhs, R&& rhs) {
    if(!all_floating<L, R>::value){
        return false;
    }

    if(cublas_enabled && lhs.is_gpu_up_to_date() && rhs.is_gpu_up_to_date()){
        impl::cublas::direct_add(lhs, rhs);
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief Subtract the RHS matrix from the LHS matrix
 * \param lhs The left-hand-side matrix
 * \param rhs The right-hand-side matrix
 */
template <typename L, typename R>
bool direct_sub(L&& lhs, R&& rhs) {
    if(!all_floating<L, R>::value){
        return false;
    }

    if(cublas_enabled && lhs.is_gpu_up_to_date() && rhs.is_gpu_up_to_date()){
        impl::cublas::direct_sub(lhs, rhs);
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief Multiply the LHS matrix by the RHS matrix
 * \param lhs The expression
 * \param rhs The scalar
 */
template <typename L, typename R>
bool direct_mul(L&& lhs, R&& rhs) {
    if(!all_floating<L, R>::value){
        return false;
    }

    if(cublas_enabled && egblas_enabled && lhs.is_gpu_up_to_date() && rhs.is_gpu_up_to_date()){
        return impl::cublas::direct_mul(lhs, rhs);
    } else {
        return false;
    }
}

/*!
 * \brief Divide the LHS matrix by the RHS matrix
 * \param lhs The expression
 * \param rhs The scalar
 */
template <typename L, typename R>
bool direct_div(L&& lhs, R&& rhs) {
    if(!all_floating<L, R>::value){
        return false;
    }

    if(cublas_enabled && egblas_enabled && lhs.is_gpu_up_to_date() && rhs.is_gpu_up_to_date()){
        return impl::cublas::direct_div(lhs, rhs);
    } else {
        return false;
    }
}

} //end of namespace detail

} //end of namespace etl
