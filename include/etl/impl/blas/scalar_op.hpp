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

template <typename T, cpp_enable_if(is_single_precision<T>::value)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    cblas_sscal(size(lhs), rhs, lhs.memory_start(), 1);
}

template <typename T, cpp_enable_if(is_double_precision<T>::value)>
void scalar_mul(T&& lhs, value_t<T> rhs) {
    cblas_dscal(size(lhs), rhs, lhs.memory_start(), 1);
}

template <typename T, cpp_enable_if(!all_floating<T>::value)>
void scalar_mul(T&& /*lhs*/, value_t<T> /*rhs*/) {}

#else

template <typename T>
void scalar_mul(T&& /*lhs*/, value_t<T> /*rhs*/) {
    cpp_unreachable("CBLAS not available/enabled");
}

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
