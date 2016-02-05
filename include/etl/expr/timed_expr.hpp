//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains a timed expression implementation.
*/

#pragma once

#include <iosfwd> //For stream support

namespace etl {

/*!
 * \brief A wrapper for expressions that need to be timed
 */
template <typename Expr>
struct timed_expr final {
private:
    Expr _value;

public:
    using value_type = value_t<Expr>; ///< The value type

    //Cannot be constructed with no args
    timed_expr() = delete;

    /*!
     * \brief Construt a new timed expression around the given ETL expression
     * \param l The ETL expression
     */
    timed_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    timed_expr(const timed_expr& e) = default;
    timed_expr(timed_expr&& e) noexcept = default;

    //Expressions are invariant
    timed_expr& operator=(const timed_expr& e) = delete;
    timed_expr& operator=(timed_expr&& e) = delete;

    /*!
     * \brief Return the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<Expr> value() {
        return _value;
    }

    /*!
     * \brief Return the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<Expr> value() const {
        return _value;
    }
};

/*!
 * \brief Specilization of the traits for timed_expr
 * timed expression simply use the same traits as its expression
 */
template <typename Expr>
struct etl_traits<etl::timed_expr<Expr>> : etl_traits<Expr> {};

/*!
 * \brief Prints the type of the timed expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename Expr>
std::ostream& operator<<(std::ostream& os, const timed_expr<Expr>& expr) {
    return os << "timed(" << expr.value() << ")";
}

} //end of namespace etl
