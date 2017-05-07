//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some TMP utilities for expressions
*/

#pragma once

#include "etl/temporary.hpp"

namespace etl {

namespace detail {

//TODO Once all temporary expr are transformed to the new format,
//remove the typename T for all and use the traits

/*!
 * \brief Traits to build a fast dyn matrix type for the result
 * type of an temporary expression with 1 or more sub types
 * \tparam E The temporary expression (impl_expr)
 * \tparam I The indices to use
 * \tparam Subs The sub expressions
 */
template <typename E, typename T, typename I, typename... Subs>
struct fast_result_type_builder;

/*!
 * \copydoc fast_result_type_builder
 */
template <typename E, typename T, size_t... I, typename... Subs>
struct fast_result_type_builder<E, T, std::index_sequence<I...>, Subs...> {
    using value_type = T; ///< The value type

    /*!
     * \brief The built type for the given Subs
     */
    using type = fast_matrix_impl<value_type, cpp::aligned_vector<value_type, default_intrinsic_traits<value_type>::alignment>, E::template order<Subs...>(), E::template dim<Subs..., I>()...>;
};

/*!
 * \brief Traits to build the result type of a temporary expression
 * \tparam E The temporary expression type
 * \tparam Fast Indicates if the result is fast or dynamic
 * \tparam Subs The sub expressions
 */
template <typename E, typename T, bool Fast, typename... Subs>
struct expr_result;

/*!
 * \copydoc expr_result
 */
template <typename E, typename T, typename... Subs>
struct expr_result<E, T, false, Subs...> {
    /*!
     * \brief The built type for the given Subs
     */
    using type = dyn_matrix_impl<T, E::template order<Subs...>(), E::dimensions()>;
};

/*!
 * \copydoc expr_result
 */
template <typename E, typename T, typename... Subs>
struct expr_result<E, T, true, Subs...> {
    /*!
     * \brief The built type for the given Subs
     */
    using type = typename fast_result_type_builder<E, T, std::make_index_sequence<E::dimensions()>, Subs...>::type;
};

/*!
 * \brief Helper traits to directly get the result type for an impl_expr
 * \tparam E The temporary expression type
 * \tparam Subs The sub expressions
 */
template <typename E, typename T, typename... Subs>
using expr_result_t = typename expr_result<E, T, all_fast<Subs...>::value, Subs...>::type;

/*!
 * \brief Helper traits to directly get the result type for an impl_expr. The result is forced to be dynamic.
 * \tparam E The temporary expression type
 * \tparam Subs The sub expressions
 */
template <typename E, typename T, typename... Subs>
using dyn_expr_result_t = typename expr_result<E, T, false, Subs...>::type;

} // end of namespace detail

/*!
 * \brief Base class for temporary impl expression
 * \tparam D The derived type
 */
template <typename D, typename T>
struct impl_expr {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Helper traits to get the result type of this expression
     */
    template <typename... Subs>
    using result_type = detail::expr_result_t<derived_t, T, Subs...>;

    /*!
     * \brief Allocate the temporary for the expression
     * \param args The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, cpp_enable_if(all_fast<Subs...>::value)>
    static result_type<Subs...>* allocate(__attribute__((unused)) Subs&&... args) {
        return new result_type<Subs...>();
    }

    /*!
     * \brief Allocate the dynamic temporary for the expression
     * \param subs The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, size_t... I>
    static result_type<Subs...>* dyn_allocate(std::index_sequence<I...> /*seq*/, Subs&&... subs) {
        return new result_type<Subs...>(derived_t::dim(subs..., I)...);
    }

    /*!
     * \brief Allocate the temporary for the expression
     * \param args The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, cpp_disable_if(all_fast<Subs...>::value)>
    static result_type<Subs...>* allocate(Subs&&... args) {
        return dyn_allocate(std::make_index_sequence<derived_t::dimensions()>(), std::forward<Subs>(args)...);
    }
};

/*!
 * \brief Base class for dynamic temporary impl expression
 * \tparam D The derived type
 */
template <typename D, typename T>
struct dyn_impl_expr {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Helper traits to get the result type of this expression
     */
    template <typename... Subs>
    using result_type = detail::dyn_expr_result_t<derived_t, T, Subs...>;

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a const reference to the derived object, i.e. the object using the CRTP injector.
     * \return a const reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    /*!
     * \brief Allocate the dynamic temporary for the expression
     * \param subs The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, size_t... I>
    result_type<Subs...>* dyn_allocate(std::index_sequence<I...> /*seq*/, Subs&&... subs) const {
        return new result_type<Subs...>(as_derived().dim(subs..., I)...);
    }

    /*!
     * \brief Allocate the temporary for the expression
     * \param args The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs>
    result_type<Subs...>* allocate(Subs&&... args) const  {
        return dyn_allocate(std::make_index_sequence<derived_t::dimensions()>(), std::forward<Subs>(args)...);
    }
};

} //end of namespace etl
