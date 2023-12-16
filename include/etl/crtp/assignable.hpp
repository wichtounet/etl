//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <concepts>
#include "etl/impl/transpose.hpp"
#include "etl/impl/fft.hpp"

/*!
 * \file
 * \brief Use CRTP technique to inject assign operations into expressions and value classes.
 */

namespace etl {

/*!
 * \brief CRTP class to inject assign operations to matrix and vector structures.
 */
template <typename D, typename V>
struct assignable {
    using derived_t  = D; ///< The derived type
    using value_type = V; ///< The value type

    /*!
     * \brief Assign the given expression to the unary expression
     * \param e The expression to get the values from
     * \return the unary expression
     */
    template <etl_expr E>
    derived_t& operator=(E&& e) {
        validate_assign(as_derived(), e);

        if constexpr (!decay_traits<E>::is_linear) {
            if (e.alias(as_derived())) {
                auto tmp = etl::force_temporary_dim_only(as_derived());

                e.assign_to(tmp);

                as_derived() = tmp;
            } else {
                e.assign_to(as_derived());
            }
        } else {
            e.assign_to(as_derived());
        }

        return as_derived();
    }

    /*!
     * \brief Assign the given expression to the unary expression
     * \param v The expression to get the values from
     * \return the unary expression
     */
    derived_t& operator=(const value_type& v) {
        if constexpr (decay_traits<derived_t>::is_direct) {
            direct_fill(as_derived(), v);
        } else {
            std::fill(as_derived().begin(), as_derived().end(), v);
        }

        return as_derived();
    }

    /*!
     * \brief Assign the given container to the unary expression
     * \param vec The container to get the values from
     * \return the unary expression
     */
    template <std_container Container>
    derived_t& operator=(const Container& vec) requires(std::convertible_to<typename Container::value_type, value_type>) {
        validate_assign(as_derived(), vec);

        std::copy(vec.begin(), vec.end(), as_derived().begin());

        return as_derived();
    }

private:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }
};

} //end of namespace etl
