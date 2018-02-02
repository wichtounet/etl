//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

namespace etl::detail {

// Utilities

/*!
 * \brief Traits to test if the given assignment is vectorizable with the given vector mode
 * \tparam V The vector mode to test
 * \tparam E The Expression to assign to the result
 * \tparam R The result type
 */
template <vector_mode_t V, typename E, typename R>
constexpr bool are_vectorizable_select = vectorize_expr                                       // ETL must be allowed to vectorize expressions
                                             && decay_traits<R>::template vectorizable<V>     // The LHS expression must be vectorizable
                                                 && decay_traits<E>::template vectorizable<V> // The RHS expression must be vectorizable
                                                     && decay_traits<E>::storage_order
                                         == decay_traits<R>::storage_order                                          // Both expressions must have the same order
                                                && get_intrinsic_traits<V>::template type<value_t<R>>::vectorizable // The LHS type must be vectorizable
                                                    && get_intrinsic_traits<V>::template type<value_t<E>>::vectorizable // The RHS type must be vectorizable
                                                        && std::is_same<                                                /// Both vector types must be the same
                                                            typename get_intrinsic_traits<V>::template type<value_t<R>>::intrinsic_type,
                                                            typename get_intrinsic_traits<V>::template type<value_t<E>>::intrinsic_type>::value;

/*!
 * \brief Integral constant indicating if vectorization is possible
 */
template <typename E, typename R>
constexpr bool are_vectorizable = (avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>)
                                  || (avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>)
                                  || (sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>);

/*!
 * \brief Select a vector mode for the given assignment type
 * \tparam E The Expression to assign to the result
 * \tparam R The result type
 */
template <typename E, typename R>
constexpr vector_mode_t select_vector_mode() {
    return (avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>)
               ? vector_mode_t::AVX512
               : (avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>)
                     ? vector_mode_t::AVX
                     : (sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>) ? vector_mode_t::SSE3 : vector_mode_t::NONE;
}

//Selectors for assign

/*!
 * \brief Integral constant indicating if a fast assign is possible.
 *
 * A fast assign is a simple memory copy from E into R.
 */
template <typename E, typename R>
constexpr bool fast_assign = all_dma<E, R>;

/*!
 * \brief Integral constant indicating if a GPU assign is possible
 */
template <typename E, typename R>
constexpr bool gpu_assign = all_homogeneous<E, R> && !fast_assign<E, R> && all_gpu_computable<E, R> && is_dma<R> && !is_scalar<E>;

/*!
 * \brief Integral constant indicating if a vectorized assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_assign = !fast_assign<E, R> && !gpu_assign<E, R> && are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct assign is possible
 */
template <typename E, typename R>
constexpr bool direct_assign = !gpu_assign<E, R> && !are_vectorizable<E, R> && !is_dma<E> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_assign = !is_dma<R>;

//Selectors for compound operations

/*!
 * \brief Integral constant indicating if a GPU compound assign is possible
 */
template <typename E, typename R>
constexpr bool gpu_compound = all_homogeneous<E, R>&& all_gpu_computable<E, R>&& is_dma<R>&& cublas_enabled&& egblas_enabled;

/*!
 * \brief Integral constant indicating if a vectorized compound assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_compound = !gpu_compound<E, R> && are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct compound assign is possible
 */
template <typename E, typename R>
constexpr bool direct_compound = !gpu_compound<E, R> && !vectorized_compound<E, R> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard compound assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_compound = !gpu_compound<E, R> && !vectorized_compound<E, R> && !direct_compound<E, R>;

//Selectors for compound div operation

/*!
 * \brief Integral constant indicating if a GPU compound assign is possible
 */
template <typename E, typename R>
constexpr bool gpu_compound_div = all_homogeneous<E, R>&& all_gpu_computable<E, R>&& is_dma<R>&& cublas_enabled&& egblas_enabled;

/*!
 * \brief Integral constant indicating if a vectorized compound div assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_compound_div = !gpu_compound_div<E, R> && (is_floating_t<value_t<E>> || is_complex_t<value_t<E>>)&&are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct compound div assign is possible
 */
template <typename E, typename R>
constexpr bool direct_compound_div = !gpu_compound_div<E, R> && !vectorized_compound_div<E, R> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard compound div assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_compound_div = !gpu_compound_div<E, R> && !vectorized_compound_div<E, R> && !direct_compound_div<E, R>;

//Selectors without GPU

/*!
 * \brief Integral constant indicating if a fast assign is possible.
 *
 * A fast assign is a simple memory copy from E into R.
 */
template <typename E, typename R>
constexpr bool fast_assign_no_gpu = all_dma<E, R>;

/*!
 * \brief Integral constant indicating if a vectorized assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_assign_no_gpu = !fast_assign_no_gpu<E, R> && are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct assign is possible
 */
template <typename E, typename R>
constexpr bool direct_assign_no_gpu = !are_vectorizable<E, R> && !is_dma<E> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_assign_no_gpu = !is_dma<R>;

//Selectors for compound operations

/*!
 * \brief Integral constant indicating if a vectorized compound assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_compound_no_gpu = are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct compound assign is possible
 */
template <typename E, typename R>
constexpr bool direct_compound_no_gpu = !vectorized_compound_no_gpu<E, R> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard compound assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_compound_no_gpu = !vectorized_compound_no_gpu<E, R> && !direct_compound_no_gpu<E, R>;

//Selectors for compound div operation

/*!
 * \brief Integral constant indicating if a vectorized compound div assign is possible
 */
template <typename E, typename R>
constexpr bool vectorized_compound_div_no_gpu = (is_floating_t<value_t<E>> || is_complex_t<value_t<E>>)&&are_vectorizable<E, R>;

/*!
 * \brief Integral constant indicating if a direct compound div assign is possible
 */
template <typename E, typename R>
constexpr bool direct_compound_div_no_gpu = !vectorized_compound_div_no_gpu<E, R> && is_dma<R>;

/*!
 * \brief Integral constant indicating if a standard compound div assign is necessary
 */
template <typename E, typename R>
constexpr bool standard_compound_div_no_gpu = !vectorized_compound_div_no_gpu<E, R> && !direct_compound_div_no_gpu<E, R>;

} //end of namespace etl::detail
