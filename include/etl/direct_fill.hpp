//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard memory utilities
 */

#pragma once

#include "etl/impl/egblas/scalar_set.hpp"

namespace etl {

/*!
 * \brief Fill the given ETL value class with the given value
 * \param mat The ETL value class
 * \param value The value to set to each element of the matrix
 */
template <typename E, typename V>
void direct_fill(E&& mat, V value) {
    if constexpr (is_single_precision<E> && egblas_enabled && impl::egblas::has_scalar_sset) {
        value_t<E> value_conv = value;

        if (mat.gpu_memory()) {
            impl::egblas::scalar_set(mat.gpu_memory(), etl::size(mat), 1, value_conv);

            mat.validate_gpu();
        }

        std::fill(mat.memory_start(), mat.memory_end(), value_conv);

        mat.validate_cpu();
    } else if constexpr (is_double_precision<E> && egblas_enabled && impl::egblas::has_scalar_dset) {
        value_t<E> value_conv = value;

        if (mat.gpu_memory()) {
            impl::egblas::scalar_set(mat.gpu_memory(), etl::size(mat), 1, value_conv);

            mat.validate_gpu();
        }

        std::fill(mat.memory_start(), mat.memory_end(), value_conv);

        mat.validate_cpu();
    } else {
        std::fill(mat.memory_start(), mat.memory_end(), value);

        mat.validate_cpu();
        mat.invalidate_gpu();
    }
}

} //end of namespace etl
