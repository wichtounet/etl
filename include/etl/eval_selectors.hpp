//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file eval_selectors.hpp
 * \brief Contains TMP selectors to select evaluation methods based
 * on configuration.
 */

#pragma once

namespace etl {

namespace detail {

// Utilities

/*!
 * \brief Traits to test if the given assignment is vectorizable with the given vector mode
 * \tparam V The vector mode to test
 * \tparam E The Expression to assign to the result
 * \tparam R The result type
 */
template <vector_mode_t V, typename E, typename R>
using are_vectorizable_select = cpp::and_u<
                               vectorize_expr, // ETL must be allowed to vectorize expressions
                               decay_traits<R>::template vectorizable<V>::value, // The LHS expression must be vectorizable
                               decay_traits<E>::template vectorizable<V>::value, // The RHS expression must be vectorizable
                               decay_traits<E>::storage_order == decay_traits<R>::storage_order, // Both expressions must have the same order
                               get_intrinsic_traits<V>::template type<value_t<R>>::vectorizable, // The LHS type must be vectorizable
                               get_intrinsic_traits<V>::template type<value_t<E>>::vectorizable, // The RHS type must be vectorizable
                               std::is_same< /// Both vector types must be the same
                                    typename get_intrinsic_traits<V>::template type<value_t<R>>::intrinsic_type,
                                    typename get_intrinsic_traits<V>::template type<value_t<E>>::intrinsic_type
                                >::value>;

/*!
 * \brief Integral constant indicating if vectorization is possible
 */
template <typename E, typename R>
using are_vectorizable = cpp::or_u<
    avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>::value,
    avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>::value,
    sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>::value>;

/*!
 * \brief Select a vector mode for the given assignment type
 * \tparam E The Expression to assign to the result
 * \tparam R The result type
 */
template <typename E, typename R>
inline constexpr vector_mode_t select_vector_mode(){
    return
          (avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>::value) ? vector_mode_t::AVX512
        : (avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>::value) ? vector_mode_t::AVX
        : (sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>::value) ? vector_mode_t::SSE3
                                                                                : vector_mode_t::NONE;
}

//Selectors for assign

/*!
 * \brief Integral constant indicating if a fast assign is possible
 */
template <typename E, typename R>
using fast_assign = cpp::and_c<has_direct_access<E>, has_direct_access<R>>;

/*!
 * \brief Integral constant indicating if a vectorized assign is possible
 */
template <typename E, typename R>
using vectorized_assign = cpp::and_u<!fast_assign<E, R>::value, are_vectorizable<E, R>::value>;

/*!
 * \brief Integral constant indicating if a direct assign is possible
 */
template <typename E, typename R>
using direct_assign = cpp::and_u<!are_vectorizable<E, R>::value, !has_direct_access<E>::value, has_direct_access<R>::value>;

/*!
 * \brief Integral constant indicating if a standard assign is necessary
 */
template <typename E, typename R>
using standard_assign = cpp::not_c<has_direct_access<R>>;

//Selectors for compound operations

/*!
 * \brief Integral constant indicating if a vectorized compound assign is possible
 */
template <typename E, typename R>
using vectorized_compound = cpp::and_u<
                               are_vectorizable<E, R>::value>;

/*!
 * \brief Integral constant indicating if a direct compound assign is possible
 */
template <typename E, typename R>
using direct_compound = cpp::and_u<
                           !vectorized_compound<E, R>::value,
                           has_direct_access<R>::value>;

/*!
 * \brief Integral constant indicating if a standard compound assign is necessary
 */
template <typename E, typename R>
using standard_compound = cpp::and_u<
                             !vectorized_compound<E, R>::value,
                             !direct_compound<E, R>::value>;

//Selectors for compound div operation

/*!
 * \brief Integral constant indicating if a vectorized compound div assign is possible
 */
template <typename E, typename R>
using vectorized_compound_div = cpp::and_u<
    (is_floating_t<value_t<E>>::value || is_complex_t<value_t<E>>::value),
    are_vectorizable<E, R>::value>;

/*!
 * \brief Integral constant indicating if a direct compound div assign is possible
 */
template <typename E, typename R>
using direct_compound_div = cpp::and_u<
                           !vectorized_compound_div<E, R>::value,
                           has_direct_access<R>::value>;

/*!
 * \brief Integral constant indicating if a standard compound div assign is necessary
 */
template <typename E, typename R>
using standard_compound_div = cpp::and_u<
                             !vectorized_compound_div<E, R>::value,
                             !direct_compound_div<E, R>::value>;

// Selectors for optimized evaluation

namespace detail {

/*!
 * \brief Implementation of an integral constant indicating if a direct transpose evaluation is possible.
 */
template <typename E, typename R>
struct is_direct_transpose_impl : std::false_type {};

/*!
 * \copydoc is_direct_transpose_impl
 */
template <typename T, typename E, typename R>
struct is_direct_transpose_impl<unary_expr<T, transpose_transformer<E>, transform_op>, R>
    : cpp::and_u<
        has_direct_access<E>::value,
        decay_traits<unary_expr<T, transpose_transformer<E>, transform_op>>::storage_order == decay_traits<R>::storage_order> {};

} //end of namespace detail

/*!
 * \brief Integral constant indicating if a direct transpose evaluation is possible.
 */
template <typename E, typename R>
using is_direct_transpose = detail::is_direct_transpose_impl<std::decay_t<E>, std::decay_t<R>>;

/*!
 * \brief Integral constant indicating if an optimized evaluation is available
 */
template <typename E, typename R>
using has_optimized_evaluation = cpp::or_c<is_direct_transpose<E, R>>;

} //end of namespace detail

} //end of namespace etl
