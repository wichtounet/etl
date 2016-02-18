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

template <vector_mode_t V, typename E, typename R>
using are_vectorizable_select = cpp::and_u<
                               vectorize_expr,
                               decay_traits<E>::template vectorizable<V>::value,
                               intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                               std::is_same<intrinsic_type<value_t<R>>, intrinsic_type<value_t<E>>>::value>;

template <typename E, typename R>
using are_vectorizable = cpp::or_u<
    avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>::value,
    avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>::value,
    sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>::value>;

template <typename E, typename R>
inline constexpr vector_mode_t select_vector_mode(){
    return
          (avx512_enabled && are_vectorizable_select<vector_mode_t::AVX512, E, R>::value) ? vector_mode_t::AVX512
        : (avx_enabled && are_vectorizable_select<vector_mode_t::AVX, E, R>::value) ? vector_mode_t::AVX
        : (sse3_enabled && are_vectorizable_select<vector_mode_t::SSE3, E, R>::value) ? vector_mode_t::SSE3
                                                                                : vector_mode_t::NONE;
}


//Selectors for assign

template <typename E, typename R>
using fast_assign = cpp::and_u<
                         has_direct_access<E>::value,
                         has_direct_access<R>::value
                         >;

template <typename E, typename R>
using parallel_vectorized_assign = cpp::and_u<
                               !fast_assign<E, R>::value,
                               parallel,
                               are_vectorizable<E, R>::value>;

template <typename E, typename R>
using vectorized_assign = cpp::and_u<
                               !fast_assign<E, R>::value,
                               !parallel_vectorized_assign<E, R>::value,
                               vectorize_expr,
                               are_vectorizable<E, R>::value>;

template <typename E, typename R>
using parallel_assign = cpp::and_u<
                               has_direct_access<R>::value,
                               !fast_assign<E, R>::value,
                               !parallel_vectorized_assign<E, R>::value,
                               parallel>;

template <typename E, typename R>
using direct_assign = cpp::and_u<
                           !fast_assign<E, R>::value,
                           !parallel_assign<E, R>::value,
                           !parallel_vectorized_assign<E, R>::value,
                           !vectorized_assign<E, R>::value,
                           has_direct_access<R>::value>;

template <typename E, typename R>
using standard_assign = cpp::and_u<
                             !fast_assign<E, R>::value,
                             !parallel_assign<E, R>::value,
                             !parallel_vectorized_assign<E, R>::value,
                             !vectorized_assign<E, R>::value,
                             !has_direct_access<R>::value>;

//Selectors for compound operations

template <typename E, typename R>
using parallel_vectorized_compound = cpp::and_u<
                               parallel,
                               are_vectorizable<E, R>::value>;

template <typename E, typename R>
using vectorized_compound = cpp::and_u<
                               !parallel_vectorized_compound<E, R>::value,
                               are_vectorizable<E, R>::value>;

template <typename E, typename R>
using parallel_compound = cpp::and_u<
                               has_direct_access<R>::value,
                               !parallel_vectorized_compound<E, R>::value,
                               parallel>;

template <typename E, typename R>
using direct_compound = cpp::and_u<
                           !parallel_compound<E, R>::value,
                           !parallel_vectorized_compound<E, R>::value,
                           !vectorized_compound<E, R>::value,
                           has_direct_access<R>::value>;

template <typename E, typename R>
using standard_compound = cpp::and_u<
                             !parallel_compound<E, R>::value,
                             !parallel_vectorized_compound<E, R>::value,
                             !vectorized_compound<E, R>::value,
                             !direct_compound<E, R>::value>;

// Selectors for optimized evaluation

template <typename E, typename R>
struct is_direct_transpose : std::false_type {};

template <typename T, typename E, typename R>
struct is_direct_transpose<unary_expr<T, transpose_transformer<E>, transform_op>, R>
    : cpp::and_u<
        has_direct_access<E>::value,
        decay_traits<unary_expr<T, transpose_transformer<E>, transform_op>>::storage_order == decay_traits<R>::storage_order> {};

template <typename E, typename R, typename DE = std::decay_t<E>, typename DR = std::decay_t<R>>
using has_optimized_evaluation = cpp::or_c<is_direct_transpose<DE, DR>>;

} //end of namespace detail

} //end of namespace etl
