//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file generator_expr.hpp
 * \brief Contains generator expressions.
 *
 * A generator expression is an expression that yields any number of values, for instance random values. The indexes
 * are not taken into account, but rather the sequence in which the functions are called. This is mostly useful for
 * initializing matrices / vectors.
*/

#pragma once

#include <iosfwd> //For stream support

namespace etl {

template <typename Generator>
class generator_expr final {
private:
    mutable Generator generator;

public:
    using value_type = typename Generator::value_type;

    template <typename... Args>
    explicit generator_expr(Args... args)
            : generator(std::forward<Args>(args)...) {}

    generator_expr(const generator_expr& e) = default;
    generator_expr(generator_expr&& e) noexcept = default;

    //Expression are invariant
    generator_expr& operator=(const generator_expr& e) = delete;
    generator_expr& operator=(generator_expr&& e) = delete;

    //Apply the expression

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        cpp_unused(i);
        return generator();
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        cpp_unused(i);
        return generator();
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    constexpr bool alias(const E& /*rhs*/) const noexcept {
        return false;
    }

    /*!
     * \brief Apply the functor
     * \return The new value of the generator
     */
    value_type operator()() const {
        return generator();
    }

    /*!
     * \brief Returns a reference to the generator op
     * \return a reference to the generator op.
    const Generator& get_generator() const {
        return generator;
    }
};

/*!
 * \brief Specialization generator_expr
 */
template <typename Generator>
struct etl_traits<etl::generator_expr<Generator>> {
    static constexpr const bool is_etl                  = true;
    static constexpr const bool is_transformer          = false;
    static constexpr const bool is_view                 = false;
    static constexpr const bool is_magic_view           = false;
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_linear               = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = true;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;
};


template <typename Generator>
std::ostream& operator<<(std::ostream& os, const generator_expr<Generator>& expr) {
    return os << expr.get_generator();
}

} //end of namespace etl
