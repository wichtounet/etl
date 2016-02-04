//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/cublas/cuda.hpp"

/*!
 * \file
 * \brief Use CRTP technique to inject GPU-related functions to
 * ETL expression classes.
 */

namespace etl {

/*!
 * \brief CRTP class to inject GPU-related functions into
 * expressions.
 *
 * This is done for expressions with temporaries that'll delegate
 * the GPU work to a temporary matrix expression.
 */
template <typename T, typename D>
struct gpu_delegate {
    using value_type = T;
    using derived_t = D;

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

    decltype(auto) delegate() const noexcept {
        return as_derived().gpu_delegate();
    }

    decltype(auto) delegate() noexcept {
        return as_derived().gpu_delegate();
    }

    bool is_valid() const noexcept {
        return as_derived().gpu_delegate_valid();
    }

    /*!
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return delegate().is_gpu_allocated();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if (is_valid()) {
            delegate().gpu_evict();
        }
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return delegate().gpu_memory();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate() const {
        delegate().gpu_allocate();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression, only if not already allocated
     */
    void gpu_allocate_if_necessary() const {
        delegate().gpu_allocate_if_necessary();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void gpu_allocate_copy() const {
        delegate().gpu_allocate_copy_if_necessary();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU, only if not already allocated.
     */
    void gpu_allocate_copy_if_necessary() const {
        delegate().gpu_allocate_copy_if_necessary();
    }

    /*!
     * \brief Reallocate the GPU memory.
     * \param memory The new GPU memory (will be moved)
     */
    void gpu_reallocate(impl::cuda::cuda_memory<T>&& memory){
        delegate().gpu_reallocate(std::move(memory));
    }

    /*!
     * \brief Release the GPU memory for another expression to use
     * \return A rvalue reference to the gpu_memory_handler.
     */
    impl::cuda::cuda_memory<T>&& gpu_release() const {
        return delegate().gpu_release();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void gpu_copy_from_if_necessary() const {
        delegate().gpu_copy_from_if_necessary();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_copy_from() const {
        return delegate().gpu_copy_from();
    }
};

} //end of namespace etl
