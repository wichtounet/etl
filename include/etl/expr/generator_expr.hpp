//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

/*!
 * \brief A generator expression
 *
 * \tparam Generator The generator functor
 */
template <typename Generator>
class generator_expr final {
private:
    mutable Generator generator;

public:
    using value_type = typename Generator::value_type; ///< The type of value generated

    /*!
     * \brief Construct a generator expression and forward the arguments to the generator
     * \param args The input arguments of the generator
     */
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
    template <typename E>
    constexpr bool alias(const E& rhs) const noexcept {
        return (void)rhs, false;
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
     */
    const Generator& get_generator() const {
        return generator;
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor) const {
        cpp_unused(visitor);
    }
};

/*!
 * \brief Specialization generator_expr
 */
template <typename Generator>
struct etl_traits<etl::generator_expr<Generator>> {
    using value_type = typename Generator::value_type; ///< The value type

    static constexpr bool is_etl                  = true;            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;           ///< Indicates if the type is a magic view
    static constexpr bool is_linear               = true;            ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = false;           ///< Indicates if the expression is thread safe
    static constexpr bool is_fast                 = true;            ///< Indicates if the expression is fast
    static constexpr bool is_value                = false;           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = true;            ///< Indicates if the expression is a generator
    static constexpr bool needs_evaluator_visitor = false;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool is_padded               = false;           ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = false;           ///< Indicates if the expression is padded
    static constexpr order storage_order          = order::RowMajor; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Return the size of the expression
     */
    static constexpr std::size_t size() {
        return 0;
    }
};

/*!
 * \brief Outputs the expression to the given stream
 * \param os The output stream
 * \param expr The generator expr
 * \return The output stream
 */
template <typename Generator>
std::ostream& operator<<(std::ostream& os, const generator_expr<Generator>& expr) {
    return os << expr.get_generator();
}

} //end of namespace etl
