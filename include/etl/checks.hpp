//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains utility checks
 *
 * The functions are using assertions to validate their conditions. When
 * possible, static assertions are used.
 */

#pragma once

namespace etl {

/*!
 * \brief Make sure the two expressions have the same size
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator)>
void validate_expression_impl([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) noexcept {
    //Nothing to test, generators are of infinite size
}

/*!
 * \brief Make sure the two expressions have the same size
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE,
          typename RE,
          cpp_enable_iff(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE> && !all_fast<LE, RE>)>
void validate_expression_impl([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    cpp_assert(etl::size(lhs) == etl::size(rhs), "Cannot perform element-wise operations on collections of different size");
}

/*!
 * \brief Make sure the two expressions have the same size
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE> && all_fast<LE, RE>)>
void validate_expression_impl([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

// In relaxed mode, the expressions are only validated when assigned
// This allows for batch_hint to works

#ifdef ETL_RELAXED
#define validate_expression(lhs, rhs) static_assert(all_etl_expr<decltype(lhs), decltype(rhs)>, "ETL functions are only made for ETL expressions ");
#else
#define validate_expression(lhs, rhs)                                                                              \
    static_assert(all_etl_expr<decltype(lhs), decltype(rhs)>, "ETL functions are only made for ETL expressions "); \
    validate_expression_impl(lhs, rhs);
#endif

/*!
 * \brief Make sure that rhs can assigned to lhs.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(etl_traits<RE>::is_generator)>
void validate_assign([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) noexcept {
    static_assert(is_etl_expr<LE>, "Assign can only work on ETL expressions");
    //Nothing to test, generators are of infinite size
}

/*!
 * \brief Make sure that rhs can assigned to lhs.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(!etl_traits<RE>::is_generator && is_etl_expr<RE> && !all_fast<LE, RE> && !is_wrapper_expr<RE>)>
void validate_assign([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    static_assert(is_etl_expr<LE>, "Assign can only work on ETL expressions");
    cpp_assert(etl::size(lhs) == etl::size(rhs), "Cannot perform element-wise operations on collections of different size");
}

/*!
 * \brief Make sure that rhs can assigned to lhs.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(!etl_traits<RE>::is_generator && is_etl_expr<RE> && all_fast<LE, RE> && !is_wrapper_expr<RE>)>
void validate_assign([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    static_assert(is_etl_expr<LE>, "Assign can only work on ETL expressions");
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

/*!
 * \brief Make sure that rhs can assigned to lhs.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(!is_etl_expr<RE> && !is_wrapper_expr<RE>)>
void validate_assign([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    static_assert(is_etl_expr<LE>, "Assign can only work on ETL expressions");
    cpp_assert(etl::size(lhs) == rhs.size(), "Cannot perform element-wise operations on collections of different size");
}

/*!
 * \brief Make sure that rhs can assigned to lhs.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 */
template <typename LE, typename RE, cpp_enable_iff(is_wrapper_expr<RE>)>
void validate_assign([[maybe_unused]] const LE& lhs, [[maybe_unused]] const RE& rhs) {
    static_assert(is_etl_expr<LE>, "Assign can only work on ETL expressions");
    cpp_assert(etl::size(lhs) == etl::size(rhs), "Cannot perform element-wise operations on collections of different size");
}

/*!
 * \brief Make sure that the expression is square
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param expr The expression to assert
 */
template <typename E>
void assert_square([[maybe_unused]] E&& expr) {
    static_assert(is_2d<E>, "Function undefined for non-square matrix");

    if constexpr (is_fast<E>) {
        static_assert(decay_traits<E>::template dim<0>() == decay_traits<E>::template dim<1>(), "Function undefined for non-square matrix");
    } else {
        cpp_assert(etl::dim<0>(expr) == etl::dim<1>(expr), "Function undefined for non-square matrix");
    }
}

namespace detail {

/*!
 * \brief Make sure that the pooling ratios are correct and that the expression can be pooled from.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \tparam C1 The pooling ratio of the first dimension
 * \tparam C2 The pooling ratio of the second dimension
 * \param e The expression to assert
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_2d<E> && !etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e) {
    cpp_assert(etl::template dim<0>(e) % C1 == 0 && etl::template dim<1>(e) % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_3d<E> && !etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e) {
    cpp_assert(etl::template dim<1>(e) % C1 == 0 && etl::template dim<2>(e) % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_4d<E> && !etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e) {
    cpp_assert(etl::template dim<2>(e) % C1 == 0 && etl::template dim<3>(e) % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_2d<E>&& etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& /*unused*/) {
    static_assert(etl_traits<E>::template dim<0>() % C1 == 0 && etl_traits<E>::template dim<1>() % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_3d<E>&& etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& /*unused*/) {
    static_assert(etl_traits<E>::template dim<1>() % C1 == 0 && etl_traits<E>::template dim<2>() % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <size_t C1, size_t C2, typename E, cpp_enable_iff(is_4d<E>&& etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& /*unused*/) {
    static_assert(etl_traits<E>::template dim<2>() % C1 == 0 && etl_traits<E>::template dim<3>() % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \brief Make sure that the pooling ratios are correct and that the expression can be pooled from.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param c1 The pooling ratio of the first dimension
 * \param c2 The pooling ratio of the second dimension
 * \param e The expression to assert
 */
template <typename E, cpp_enable_iff(is_2d<E>)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e, [[maybe_unused]] size_t c1, [[maybe_unused]] size_t c2) {
    cpp_assert(etl::template dim<0>(e) % c1 == 0 && etl::template dim<1>(e) % c2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <typename E, cpp_enable_iff(is_3d<E>)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e, [[maybe_unused]] size_t c1, [[maybe_unused]] size_t c2) {
    cpp_assert(etl::template dim<1>(e) % c1 == 0 && etl::template dim<2>(e) % c2 == 0, "Dimensions not divisible by the pooling ratio");
}

/*!
 * \copydoc validate_pmax_pooling_impl
 */
template <typename E, cpp_enable_iff(is_4d<E>)>
void validate_pmax_pooling_impl([[maybe_unused]] const E& e, [[maybe_unused]] size_t c1, [[maybe_unused]] size_t c2) {
    cpp_assert(etl::template dim<2>(e) % c1 == 0 && etl::template dim<3>(e) % c2 == 0, "Dimensions not divisible by the pooling ratio");
}

} //end of namespace detail

/*!
 * \brief Make sure that the pooling ratios are correct and that the expression can be pooled from.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \tparam C1 The pooling ratio of the first dimension
 * \tparam C2 The pooling ratio of the second dimension
 * \param expr The expression to assert
 */
template <size_t C1, size_t C2, typename E>
void validate_pmax_pooling(const E& expr) {
    static_assert(is_etl_expr<E>, "Prob. Max Pooling only defined for ETL expressions");
    static_assert(etl_traits<E>::dimensions() >= 2 && etl_traits<E>::dimensions() <= 4, "Prob. Max Pooling only defined for 2D and 3D");

    detail::validate_pmax_pooling_impl<C1, C2>(expr);
}

/*!
 * \brief Make sure that the pooling ratios are correct and that the expression can be pooled from.
 *
 * This function uses assertion to validate the condition. If possible, the
 * assertion is done at compile time.
 *
 * \param c1 The pooling ratio of the first dimension
 * \param c2 The pooling ratio of the second dimension
 * \param expr The expression to assert
 */
template <typename E>
void validate_pmax_pooling(const E& expr, size_t c1, size_t c2) {
    static_assert(is_etl_expr<E>, "Prob. Max Pooling only defined for ETL expressions");
    static_assert(etl_traits<E>::dimensions() >= 2 && etl_traits<E>::dimensions() <= 4, "Prob. Max Pooling only defined for 2D and 3D");

    detail::validate_pmax_pooling_impl(expr, c1, c2);
}

} //end of namespace etl
