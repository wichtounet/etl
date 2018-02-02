//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

#include "etl/wrapper_traits.hpp"

namespace etl {

/*!
 * \brief A wrapper for expressions that need to be timed
 */
template <typename Expr, typename R>
struct timed_expr final {
    using clock_resolution = R;             ///< The clock resolution
    using expr_t           = Expr;          ///< The wrapped expression type
    using value_type       = value_t<Expr>; ///< The value type

private:
    Expr value;

    friend struct wrapper_traits<timed_expr>;

public:
    //Cannot be constructed with no args
    timed_expr() = delete;

    /*!
     * \brief Construt a new timed expression around the given ETL expression
     * \param l The ETL expression
     */
    explicit timed_expr(Expr l) : value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    timed_expr(const timed_expr& e)     = default;
    timed_expr(timed_expr&& e) noexcept = default;

    //Expressions are invariant
    timed_expr& operator=(const timed_expr& e) = delete;
    timed_expr& operator=(timed_expr&& e) = delete;

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param other The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& other) const noexcept {
        return value.alias(other);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_add_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(+=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_sub_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(-=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_mul_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(*=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_div_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(/=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        auto start_time = etl::timer_clock::now();

        value.assign_mod_to(lhs);

        auto end_time = etl::timer_clock::now();
        auto duration = std::chrono::duration_cast<clock_resolution>(end_time - start_time);

        std::cout << "timed(%=): " << value << " took " << duration.count() << resolution_to_string<clock_resolution>() << std::endl;
    }

    /*!
     * \brief Prints the type of the timed expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const timed_expr& expr) {
        return os << "timed(" << expr.value << ")";
    }
};

/*!
 * \brief Specilization of the traits for timed_expr
 * timed expression simply use the same traits as its expression
 */
template <typename Expr, typename R>
struct etl_traits<etl::timed_expr<Expr, R>> : wrapper_traits<etl::timed_expr<Expr, R>> {};

} //end of namespace etl
