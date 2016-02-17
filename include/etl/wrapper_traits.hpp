//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains traits for wrapper expressions
*/

#pragma once

namespace etl {

/*!
 * \brief Traits for wrapper expressions
 */
template <typename T>
struct wrapper_traits {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename T::expr_t>;
    using sub_traits = etl_traits<sub_expr_t>;

    static constexpr const bool is_etl                  = sub_traits::is_etl;                  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = sub_traits::is_transformer;          ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = sub_traits::is_view;                 ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = sub_traits::is_magic_view;           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = sub_traits::is_fast;                 ///< Indicates if the expression is fast
    static constexpr const bool is_value                = sub_traits::is_value;                ///< Indicates if the expression is of value type
    static constexpr const bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_generator            = sub_traits::is_generator;            ///< Indicates if the expression is a generator expression
    static constexpr const bool needs_temporary_visitor = sub_traits::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = sub_traits::needs_evaluator_visitor; ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = sub_traits::storage_order;           ///< The expression storage order

    /*!
     * The vectorization type for V
     */
    template <vector_mode_t V>
    using vectorizable = typename sub_traits::template vectorizable<V>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return sub_traits::size(v.value());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return sub_traits::dim(v.value(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return sub_traits::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return sub_traits::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

} //end of namespace etl
