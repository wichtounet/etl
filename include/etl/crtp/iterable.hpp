//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file iterable.hpp
 * \brief Use CRTP technique to inject functions for iterators.
 */

#pragma once

namespace etl {

/*!
 * \brief CRTP class to inject iterators functions.
 *
 * This CRTP class injects iterators functions.
 */
template <typename D, bool DMA = false>
struct iterable {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto begin() noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_start();
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto end() noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_end();
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto cbegin() const noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_start();
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto cend() const noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_end();
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto begin() const noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_start();
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_enable_iff(B)>
    auto end() const noexcept {
        as_derived().ensure_cpu_up_to_date();
        return as_derived().memory_end();
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto begin() noexcept {
        return typename derived_t::iterator{as_derived(), 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto end() noexcept {
        return typename derived_t::iterator{as_derived(), etl::size(as_derived())};
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto cbegin() const noexcept {
        return typename derived_t::const_iterator{as_derived(), 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto cend() const noexcept {
        return typename derived_t::const_iterator{as_derived(), etl::size(as_derived())};
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto begin() const noexcept {
        return typename derived_t::const_iterator{as_derived(), 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    template<bool B = DMA, cpp_disable_iff(B)>
    auto end() const noexcept {
        return typename derived_t::const_iterator{as_derived(), etl::size(as_derived())};
    }
};

} //end of namespace etl
