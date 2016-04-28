//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime matrix implementation
 */

#pragma once

#include "etl/dyn_base.hpp"    //The base class and utilities

namespace etl {

template <typename Matrix, typename Enable = void>
struct static_check_square {};

template <typename Matrix>
struct static_check_square<Matrix, std::enable_if_t<all_fast<Matrix>::value && is_2d<Matrix>::value>> {
    static_assert(etl_traits<Matrix>::template dim<0>() == etl_traits<Matrix>::template dim<1>(), "Static matrix must be square");
};

/*!
 * \brief Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename Matrix>
struct sym_matrix final {
    using matrix_t = Matrix; ///< The adapted matrix type

    static_assert(etl_traits<matrix_t>::dimensions() == 2, "Symmetric matrix must be two-dimensional");
    using scs = static_check_square<matrix_t>; ///< static_check trick

    static constexpr const std::size_t n_dimensions = matrix_t::D;                           ///< The number of dimensions
    static constexpr const order storage_order      = matrix_t::SO;                          ///< The storage order
    static constexpr const std::size_t alignment    = intrinsic_traits<matrix_t>::alignment; ///< The memory alignment

    using value_type        = value_t<matrix_t>; ///< The value type
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type
    using iterator          = memory_type;       ///< The type of iterator
    using const_iterator    = const_memory_type; ///< The type of const iterator

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_t>;

private:
    matrix_t matrix; ///< The adapted matrix

public:
    sym_matrix() noexcept {
        matrix = value_type(0);
    }

    sym_matrix(value_type value) noexcept : matrix(value) {
        //Nothing else to init
    }

    sym_matrix(std::size_t dim) noexcept : matrix(dim, dim) {
        matrix = value_type(0);
    }

    sym_matrix(std::size_t dim, value_type value) noexcept : matrix(dim, dim, value) {
        //Nothing else to init
    }

    sym_matrix(const sym_matrix& rhs) = default;
    sym_matrix& operator=(const sym_matrix& rhs) = default;

    sym_matrix(sym_matrix&& rhs) = default;
    sym_matrix& operator=(sym_matrix&& rhs) = default;

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return 2;
    }
};

} //end of namespace etl
