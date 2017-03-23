//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

/*!
 * \brief A derivative of the 2D max pooling (combine derivative and upsampling for performance)
 * \tparam A The input type
 * \tparam B The output type
 * \tparam C The errors type
 */
template <typename A, typename B, typename C, size_t C1, size_t C2>
struct max_pool_upsample_2d_expr : base_temporary_expr_tern<max_pool_upsample_2d_expr<A, B, C, C1, C2>, A, B, C> {
    using value_type = value_t<A>;                                     ///< The type of value of the expression
    using sub_traits = etl::decay_traits<A>;                           /// The traits of the first sub type
    using this_type  = max_pool_upsample_2d_expr<A, B, C, C1, C2>; ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>;   ///< The base type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    friend struct etl_traits<max_pool_upsample_2d_expr>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    max_pool_upsample_2d_expr(A a, B b, C c) : base_type(a, b, c) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R, cpp_enable_if(all_fast<A, B, C, R>::value)>
    static void check(const A& a, const B& b, const C& c, const R& result) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");

        static_assert(etl::decay_traits<R>::size() == etl::decay_traits<A>::size(), "max_pool_upsample_2d:A and R must have the same size");
        static_assert(etl::decay_traits<B>::size() == etl::decay_traits<C>::size(), "max_pool_upsample_2d:B and C must have the same size");
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R, cpp_disable_if(all_fast<A, B, C, R>::value)>
    static void check(const A& a, const B& b, const C& c, const R& result) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");

        cpp_assert(etl::size(result) == etl::size(a), "max_pool_upsample_2d:A and R must have the same size");
        cpp_assert(etl::size(b) == etl::size(c), "max_pool_upsample_2d:B and C must have the same size");
    }

    /*!
     * \brief Apply the expression
     * \param a The input
     * \param c The expression where to store the results
     */
    template <typename R>
    static void apply(A&& a, B&& b, C&& c, R&& result) {
        static_assert(all_etl_expr<A, B, C, R>::value, "Max Pool Derivative only supported for ETL expressions");

        check(a, b, c, result);

        impl::max_pool_upsample_2d::apply<C1, C2>(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            make_temporary(std::forward<C>(c)),
            make_temporary(std::forward<R>(result)));
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        this->apply_base(lhs);
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
};

/*!
 * \brief Traits for a pooling usample expression
 * \tparam A The pooling usample sub type
 */
template <typename A, typename B, typename C, size_t C1, size_t C2>
struct etl_traits<etl::max_pool_upsample_2d_expr<A, B, C, C1, C2>> {
    using expr_t     = etl::max_pool_upsample_2d_expr<A, B, C, C1, C2>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                 ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                          ///< The sub traits
    using value_type = value_t<A>;                                      ///< The value type of the expression

    static constexpr bool is_etl                  = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = sub_traits::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool is_linear               = true;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_gpu                  = false;                     ///< Indicates if the expression can be done on GPU
    static constexpr bool needs_evaluator_visitor = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <std::size_t DD>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static std::size_t dim(const expr_t& e, std::size_t d) {
        return etl::dim(e.a(), d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static std::size_t size(const expr_t& e) {
        return etl::size(e.a());
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr std::size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

/*!
 * \brief A derivative of the max pooling (combine derivative and upsampling for performance)
 * \tparam A The input type
 * \tparam B The output type
 * \tparam C The errors type
 */
template <typename A, typename B, typename C, size_t C1, size_t C2, size_t C3>
struct max_pool_upsample_3d_expr : base_temporary_expr_tern<max_pool_upsample_3d_expr<A, B, C, C1, C2, C3>, A, B, C> {
    using value_type = value_t<A>;                                     ///< The type of value of the expression
    using sub_traits = etl::decay_traits<A>;                           /// The traits of the first sub type
    using this_type  = max_pool_upsample_3d_expr<A, B, C, C1, C2, C3>; ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>;   ///< The base type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    friend struct etl_traits<max_pool_upsample_3d_expr>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    max_pool_upsample_3d_expr(A a, B b, C c) : base_type(a, b, c) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R, cpp_enable_if(all_fast<A, B, C, R>::value)>
    static void check(const A& a, const B& b, const C& c, const R& result) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");

        static_assert(etl::decay_traits<R>::size() == etl::decay_traits<A>::size(), "max_pool_upsample_3d:A and R must have the same size");
        static_assert(etl::decay_traits<B>::size() == etl::decay_traits<C>::size(), "max_pool_upsample_3d:B and C must have the same size");
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R, cpp_disable_if(all_fast<A, B, C, R>::value)>
    static void check(const A& a, const B& b, const C& c, const R& result) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");

        cpp_assert(etl::size(result) == etl::size(a), "max_pool_upsample_3d:A and R must have the same size");
        cpp_assert(etl::size(b) == etl::size(c), "max_pool_upsample_3d:B and C must have the same size");
    }

    /*!
     * \brief Apply the expression
     * \param a The input
     * \param c The expression where to store the results
     */
    template <typename R>
    static void apply(A&& a, B&& b, C&& c, R&& result) {
        static_assert(all_etl_expr<A, B, C, R>::value, "Max Pool Derivative only supported for ETL expressions");

        check(a, b, c, result);

        impl::max_pool_upsample_3d::apply<C1, C2, C3>(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            make_temporary(std::forward<C>(c)),
            make_temporary(std::forward<R>(result)));
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        this->apply_base(lhs);
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
};

/*!
 * \brief Traits for a pooling usample expression
 * \tparam A The pooling usample sub type
 */
template <typename A, typename B, typename C, size_t C1, size_t C2, size_t C3>
struct etl_traits<etl::max_pool_upsample_3d_expr<A, B, C, C1, C2, C3>> {
    using expr_t     = etl::max_pool_upsample_3d_expr<A, B, C, C1, C2, C3>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                     ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                              ///< The sub traits
    using value_type = value_t<A>;                                          ///< The value type of the expression

    static constexpr bool is_etl                  = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = sub_traits::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool is_linear               = true;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_gpu                  = false;                     ///< Indicates if the expression can be done on GPU
    static constexpr bool needs_evaluator_visitor = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <std::size_t DD>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static std::size_t dim(const expr_t& e, std::size_t d) {
        return etl::dim(e.a(), d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static std::size_t size(const expr_t& e) {
        return etl::size(e.a());
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr std::size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

/*!
 * \brief A derivative of the 2D max pooling (combine derivative and upsampling for performance)
 * \tparam A The input type
 * \tparam B The output type
 * \tparam C The errors type
 */
template <typename A, typename B, typename C>
struct dyn_max_pool_upsample_2d_expr : base_temporary_expr_tern<dyn_max_pool_upsample_2d_expr<A, B, C>, A, B, C> {
    using value_type = value_t<A>;                                     ///< The type of value of the expression
    using sub_traits = etl::decay_traits<A>;                           /// The traits of the first sub type
    using this_type  = dyn_max_pool_upsample_2d_expr<A, B, C>;         ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>;   ///< The base type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

private:

    const size_t c1;
    const size_t c2;

    friend struct etl_traits<dyn_max_pool_upsample_2d_expr>;

public:

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    dyn_max_pool_upsample_2d_expr(A a, B b, C c, size_t c1, size_t c2) : base_type(a, b, c), c1(c1), c2(c2) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R>
    void check(const A& a, const B& b, const C& c, const R& result) const {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_2d");

        cpp_assert(etl::size(result) == etl::size(a), "max_pool_upsample_2d:A and R must have the same size");
        cpp_assert(etl::size(b) == etl::size(c), "max_pool_upsample_2d:B and C must have the same size");
    }

    /*!
     * \brief Apply the expression
     * \param a The input
     * \param c The expression where to store the results
     */
    template <typename R>
    void apply(A&& a, B&& b, C&& c, R&& result) const {
        static_assert(all_etl_expr<A, B, C, R>::value, "Max Pool Derivative only supported for ETL expressions");

        check(a, b, c, result);

        impl::max_pool_upsample_2d::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            make_temporary(std::forward<C>(c)),
            make_temporary(std::forward<R>(result)),
            c1, c2);
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        this->apply_base(lhs);
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
};

/*!
 * \brief Traits for a pooling usample expression
 * \tparam A The pooling usample sub type
 */
template <typename A, typename B, typename C>
struct etl_traits<etl::dyn_max_pool_upsample_2d_expr<A, B, C>> {
    using expr_t     = etl::dyn_max_pool_upsample_2d_expr<A, B, C>;     ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                 ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                          ///< The sub traits
    using value_type = value_t<A>;                                      ///< The value type of the expression

    static constexpr bool is_etl                  = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                     ///< Indicates if the expression is fast
    static constexpr bool is_linear               = true;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_gpu                  = false;                     ///< Indicates if the expression can be done on GPU
    static constexpr bool needs_evaluator_visitor = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static std::size_t dim(const expr_t& e, std::size_t d) {
        return etl::dim(e.a(), d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static std::size_t size(const expr_t& e) {
        return etl::size(e.a());
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

/*!
 * \brief A derivative of the 3D max pooling (combine derivative and upsampling for performance)
 * \tparam A The input type
 * \tparam B The output type
 * \tparam C The errors type
 */
template <typename A, typename B, typename C>
struct dyn_max_pool_upsample_3d_expr : base_temporary_expr_tern<dyn_max_pool_upsample_3d_expr<A, B, C>, A, B, C> {
    using value_type = value_t<A>;                                     ///< The type of value of the expression
    using sub_traits = etl::decay_traits<A>;                           /// The traits of the first sub type
    using this_type  = dyn_max_pool_upsample_3d_expr<A, B, C>;         ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>;   ///< The base type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

private:

    const size_t c1;
    const size_t c2;
    const size_t c3;

    friend struct etl_traits<dyn_max_pool_upsample_3d_expr>;

public:

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    dyn_max_pool_upsample_3d_expr(A a, B b, C c, size_t c1, size_t c2, size_t c3) : base_type(a, b, c), c1(c1), c2(c2), c3(c3) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R>
    void check(const A& a, const B& b, const C& c, const R& result) const {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
        cpp_unused(result);

        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsampl_3d");

        cpp_assert(etl::size(result) == etl::size(a), "max_pool_upsample_3d:A and R must have the same size");
        cpp_assert(etl::size(b) == etl::size(c), "max_pool_upsample_3d:B and C must have the same size");
    }

    /*!
     * \brief Apply the expression
     * \param a The input
     * \param c The expression where to store the results
     */
    template <typename R>
    void apply(A&& a, B&& b, C&& c, R&& result) const {
        static_assert(all_etl_expr<A, B, C, R>::value, "Max Pool Derivative only supported for ETL expressions");

        check(a, b, c, result);

        impl::max_pool_upsample_3d::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            make_temporary(std::forward<C>(c)),
            make_temporary(std::forward<R>(result)),
            c1, c2, c3);
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        this->apply_base(lhs);
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
};

/*!
 * \brief Traits for a pooling usample expression
 * \tparam A The pooling usample sub type
 */
template <typename A, typename B, typename C>
struct etl_traits<etl::dyn_max_pool_upsample_3d_expr<A, B, C>> {
    using expr_t     = etl::dyn_max_pool_upsample_3d_expr<A, B, C>;     ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                 ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                          ///< The sub traits
    using value_type = value_t<A>;                                      ///< The value type of the expression

    static constexpr bool is_etl                  = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                     ///< Indicates if the expression is fast
    static constexpr bool is_linear               = true;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_gpu                  = false;                     ///< Indicates if the expression can be done on GPU
    static constexpr bool needs_evaluator_visitor = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static std::size_t dim(const expr_t& e, std::size_t d) {
        return etl::dim(e.a(), d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static std::size_t size(const expr_t& e) {
        return etl::size(e.a());
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

} //end of namespace etl
