//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

#include "etl/traits_lite.hpp" //forward declaration of the traits

namespace etl {

namespace detail {

//Selectors for assign

template <typename E, typename R>
struct fast_assign : cpp::and_u<
                         has_direct_access<E>::value,
                         has_direct_access<R>::value,
                         !is_temporary_expr<E>::value> {};

template <typename E, typename R>
struct parallel_vectorized_assign : cpp::and_u<
                               !fast_assign<E, R>::value,
                               vectorize_expr,
                               parallel,
                               decay_traits<E>::vectorizable,
                               intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                               !is_temporary_expr<E>::value,
                               std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

template <typename E, typename R>
struct vectorized_assign : cpp::and_u<
                               !fast_assign<E, R>::value,
                               !parallel_vectorized_assign<E, R>::value,
                               vectorize_expr,
                               decay_traits<E>::vectorizable,
                               intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                               !is_temporary_expr<E>::value,
                               std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

template <typename E, typename R>
struct parallel_assign : cpp::and_u<
                               has_direct_access<R>::value,
                               !fast_assign<E, R>::value,
                               !parallel_vectorized_assign<E, R>::value,
                               parallel,
                               !is_temporary_expr<E>::value> {};

template <typename E, typename R>
struct direct_assign : cpp::and_u<
                           !fast_assign<E, R>::value,
                           !parallel_assign<E, R>::value,
                           !parallel_vectorized_assign<E, R>::value,
                           !vectorized_assign<E, R>::value,
                           has_direct_access<R>::value,
                           !is_temporary_expr<E>::value> {};

template <typename E, typename R>
struct standard_assign : cpp::and_u<
                             !fast_assign<E, R>::value,
                             !parallel_assign<E, R>::value,
                             !parallel_vectorized_assign<E, R>::value,
                             !vectorized_assign<E, R>::value,
                             !has_direct_access<R>::value,
                             !is_temporary_expr<E>::value> {};

//Selectors for compound operations

template <typename E, typename R>
struct parallel_vectorized_compound : cpp::and_u<
                               vectorize_expr,
                               parallel,
                               decay_traits<E>::vectorizable,
                               intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                               std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

template <typename E, typename R>
struct vectorized_compound : cpp::and_u<
                               !parallel_vectorized_compound<E, R>::value,
                               vectorize_expr,
                               decay_traits<E>::vectorizable,
                               intrinsic_traits<value_t<R>>::vectorizable, intrinsic_traits<value_t<E>>::vectorizable,
                               std::is_same<typename intrinsic_traits<value_t<R>>::intrinsic_type, typename intrinsic_traits<value_t<E>>::intrinsic_type>::value> {};

template <typename E, typename R>
struct parallel_compound : cpp::and_u<
                               has_direct_access<R>::value,
                               !parallel_vectorized_compound<E, R>::value,
                               parallel> {};

template <typename E, typename R>
struct direct_compound : cpp::and_u<
                           !parallel_compound<E, R>::value,
                           !parallel_vectorized_compound<E, R>::value,
                           !vectorized_compound<E, R>::value,
                           has_direct_access<R>::value> {};

template <typename E, typename R>
struct standard_compound : cpp::and_u<
                             !parallel_compound<E, R>::value,
                             !parallel_vectorized_compound<E, R>::value,
                             !vectorized_compound<E, R>::value,
                             !direct_compound<E, R>::value> {};

} //end of namespace detail

} //end of namespace etl
