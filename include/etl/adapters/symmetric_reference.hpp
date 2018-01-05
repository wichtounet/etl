//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains symmetric matrix reference proxy implementation.
 */

#pragma once

namespace etl {

namespace sym_detail {

/*!
 * \brief A proxy representing a reference to a mutable element of a symmetric matrix
 * \tparam M The matrix type
 */
template <typename M>
struct symmetric_reference {
    using matrix_type              = M;                                ///< The matrix type
    using value_type               = typename matrix_type::value_type; ///< The value type
    using raw_pointer_type         = value_type*;                      ///< A raw pointer type
    using raw_reference_type       = value_type&;                      ///< A raw reference type
    using const_raw_reference_type = std::add_const_t<value_type>&;    ///< A raw reference type
    using expr_t                   = M;                                ///< The symmetric matrix

    matrix_type& matrix;   ///< Reference to the matrix
    size_t i;         ///< The first index
    size_t j;         ///< The second index
    value_type& value;     ///< Reference to the value
    value_type& sym_value; ///< Reference to the symmetric value

    /*!
     * \brief Constructs a new symmetric_reference
     * \param matrix The source matrix
     * \param i The index i of the first dimension
     * \param j The index j of the second dimension
     */
    symmetric_reference(matrix_type& matrix, size_t i, size_t j)
            : matrix(matrix), i(i), j(j), value(matrix(i, j)), sym_value(matrix(j, i)) {
        //Nothing else to init
    }

    /*!
     * \brief Sets a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator=(const value_type& rhs) {
        value = rhs;
        if(i != j){
            sym_value = rhs;
        }
        return *this;
    }

    /*!
     * \brief Adds a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator+=(value_type rhs) {
        value += rhs;
        if(i != j){
            sym_value += rhs;
        }
        return *this;
    }

    /*!
     * \brief Subtract a new value from the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator-=(value_type rhs) {
        value -= rhs;
        if(i != j){
            sym_value -= rhs;
        }
        return *this;
    }

    /*!
     * \brief Multiply by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator*=(value_type rhs) {
        value *= rhs;
        if(i != j){
            sym_value *= rhs;
        }
        return *this;
    }

    /*!
     * \brief Divide by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator/=(value_type rhs) {
        value /= rhs;
        if(i != j){
            sym_value /= rhs;
        }
        return *this;
    }

    /*!
     * \brief Modulo by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator%=(value_type rhs) {
        value %= rhs;
        if(i != j){
            sym_value %= rhs;
        }
        return *this;
    }

    /*!
     * \brief Casts the proxy reference to the raw reference type
     * \return a raw reference to the element
     */
    operator const_raw_reference_type&() const {
        return value;
    }
};

} //end of namespace sym_detail

} //end of namespace etl
