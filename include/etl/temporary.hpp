//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/stop.hpp"

namespace etl {

namespace detail {

template <typename E, typename Sequence>
struct build_fast_dyn_matrix_type;

template <typename E, std::size_t... I>
struct build_fast_dyn_matrix_type<E, std::index_sequence<I...>> {
    using type = fast_matrix_impl<value_t<E>, std::vector<value_t<E>>, decay_traits<E>::storage_order, decay_traits<E>::template dim<I>()...>;
};

} // end of namespace detail

/*!
 * \brief Force a temporary out of the expression
 *
 * In case of a fast matrix, a fast matrix with vector storage is created  even
 * if the input has array storage.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression
 */
template <typename E, cpp_enable_if(decay_traits<E>::is_fast)>
decltype(auto) force_temporary(E&& expr) {
    return typename detail::build_fast_dyn_matrix_type<E, std::make_index_sequence<decay_traits<E>::dimensions()>>::type{std::forward<E>(expr)};
}

/*!
 * \brief Force a temporary out of the expression
 *
 * In case of a fast matrix, a fast matrix with vector storage is created  even
 * if the input has array storage.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression
 */
template <typename E, cpp_enable_if(!decay_traits<E>::is_fast && !is_sparse_matrix<E>::value)>
decltype(auto) force_temporary(E&& expr) {
    //Sizes will be directly propagated
    return dyn_matrix_impl<value_t<E>, decay_traits<E>::storage_order, decay_traits<E>::dimensions()>{std::forward<E>(expr)};
}

//TODO the traits should include and is_sparse value that should be
//handled here

/*!
 * \brief Force a temporary out of the expression
 *
 * In case of a fast matrix, a fast matrix with vector storage is created  even
 * if the input has array storage.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression
 */
template <typename E, cpp_enable_if(is_sparse_matrix<E>::value)>
decltype(auto) force_temporary(E&& expr) {
    //Sizes will be directly propagated
    return std::decay_t<E>{std::forward<E>(expr)};
}

/*!
 * \brief Force a dynamic temporary out of the expression
 *
 * This function will always return a dyn_matrix.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression
 */
template <typename E>
decltype(auto) force_temporary_dyn(E&& expr) {
    //Sizes will be directly propagated
    return dyn_matrix_impl<value_t<E>, decay_traits<E>::storage_order, decay_traits<E>::dimensions()>{std::forward<E>(expr)};
}

/*!
 * \brief Make a temporary out of the expression if necessary
 *
 * A temporary is necessary when the expression has no direct access and when
 * creation of temporaries is not disabled.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression if necessary, otherwise the expression itself
 */
template <typename E, cpp_enable_if(has_direct_access<E>::value || !create_temporary)>
decltype(auto) make_temporary(E&& expr) {
    return std::forward<E>(expr);
}

/*!
 * \brief Make a temporary out of the expression if necessary
 *
 * A temporary is necessary when the expression has no direct access and when
 * creation of temporaries is not disabled.
 *
 * \param expr The expression to make a temporary from
 * \return a temporary of the expression if necessary, otherwise the expression itself
 */
template <typename E, cpp_enable_if(!has_direct_access<E>::value && create_temporary)>
decltype(auto) make_temporary(E&& expr) {
    return force_temporary(std::forward<E>(expr));
}

} //end of namespace etl
