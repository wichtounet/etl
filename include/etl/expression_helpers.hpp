//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file expression_helpers.hpp
 * \brief Contains internal helpers to build expressions.
*/

#pragma once

namespace etl {

namespace detail {

template<typename T>
using build_type = std::conditional_t<
    is_etl_value<T>::value,
    const std::decay_t<T>&,
    std::decay_t<T>>;

template<typename T>
using build_identity_type = std::conditional_t<
    is_etl_value<T>::value,
    std::conditional_t<
        std::is_const<std::remove_reference_t<T>>::value,
        const std::decay_t<T>&,
        std::decay_t<T>&>,
    std::decay_t<T>>;

template<typename LE, typename RE, template<typename> class OP>
using left_binary_helper = binary_expr<value_t<LE>, build_type<LE>, OP<value_t<LE>>, build_type<RE>>;

template<typename LE, typename RE, typename OP>
using left_binary_helper_op = binary_expr<value_t<LE>, build_type<LE>, OP, build_type<RE>>;

template<typename LE, typename RE, template<typename> class OP>
using right_binary_helper = binary_expr<value_t<RE>, build_type<LE>, OP<value_t<RE>>, build_type<RE>>;

template<typename E, template<typename> class OP>
using unary_helper = unary_expr<value_t<E>, build_type<E>, OP<value_t<E>>>;

template<typename E, typename OP>
using identity_helper = unary_expr<value_t<E>, OP, identity_op>;

template<typename E, typename OP>
using virtual_helper = unary_expr<E, OP, stateful_op>;

template<typename E, template<typename> class OP>
using stable_transform_helper = unary_expr<value_t<E>, OP<build_type<E>>, stateful_op>;

template<typename LE, typename RE, template<typename,typename> class OP>
using stable_transform_binary_helper = unary_expr<value_t<LE>, OP<build_type<LE>, build_type<RE>>, stateful_op>;

template<typename A, typename B, template<typename> class OP>
using temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>>, void>;

template<typename A, template<typename> class OP>
using temporary_unary_helper = temporary_unary_expr<value_t<A>, build_type<A>, OP<value_t<A>>, void>;

template<typename T, typename A, template<typename> class OP>
using temporary_unary_helper_type = temporary_unary_expr<T, build_type<A>, OP<T>, void>;

template<typename A, typename B, typename C, template<typename> class OP>
using forced_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>>, build_identity_type<C>>;

template<typename A, typename C, template<typename> class OP>
using forced_temporary_unary_helper = temporary_unary_expr<value_t<A>, build_type<A>, OP<value_t<A>>, build_identity_type<C>>;

template<typename T, typename A, typename C, template<typename> class OP>
using forced_temporary_unary_helper_type = temporary_unary_expr<T, build_type<A>, OP<T>, build_identity_type<C>>;

template<typename A, typename B, template<typename, std::size_t> class OP, std::size_t D>
using dim_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>, D>, void>;

template<typename A, typename B, typename C, template<typename, std::size_t> class OP, std::size_t D>
using dim_forced_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>, D>, build_identity_type<C>>;

} //end of namespace detail

} //end of namespace etl
