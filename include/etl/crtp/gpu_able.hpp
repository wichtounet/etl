//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDA
#include "etl/impl/cublas/cuda.hpp"
#endif

/*!
 * \file
 * \brief Use CRTP technique to inject GPU-related functions to
 * ETL value classes.
 */

/*
 * Notes
 *  - The bad constness is mostly here to allow compilation without
 *  too many tricks, it should be improved
 */

namespace etl {

#ifdef ETL_CUDA

/*!
 * \brief CRTP class to inject GPU-related functions
 *
 * Allocations and copy to GPU are considered const-functions,
 * because the state of the expression itself is not modified. On
 * the other hand, copy back from the GPU to the expression is
 * non-const.
 */
template <typename T, typename D>
struct gpu_able {
    using value_type = T;
    using derived_t = D;

    mutable impl::cuda::cuda_memory<value_type> gpu_memory_handler;

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
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return gpu_memory_handler.is_set();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if(is_gpu_allocated()){
            gpu_memory_handler.reset();
        }
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return gpu_memory_handler.get();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate() const {
        gpu_memory_handler = impl::cuda::cuda_allocate(as_derived());
    }

    /*!
     * \brief Allocate memory on the GPU for the expression, only if not already allocated
     */
    void gpu_allocate_if_necessary() const {
        if(!is_gpu_allocated()){
            gpu_allocate();
        }
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void gpu_allocate_copy() const {
        gpu_memory_handler = impl::cuda::cuda_allocate_copy(as_derived());
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU, only if not already allocated.
     */
    void gpu_allocate_copy_if_necessary() const {
        if(!is_gpu_allocated()){
            gpu_allocate_copy();
        }
    }

    /*!
     * \brief Reallocate the GPU memory.
     * \param memory The new GPU memory (will be moved)
     */
    void gpu_reallocate(impl::cuda::cuda_memory<T>&& memory){
        gpu_memory_handler = std::move(memory);
    }

    /*!
     * \brief Release the GPU memory for another expression to use
     * \return A rvalue reference to the gpu_memory_handler.
     */
    impl::cuda::cuda_memory<T>&& gpu_release() const {
        return std::move(gpu_memory_handler);
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void gpu_copy_from_if_necessary() const {
        if(is_gpu_allocated()){
            gpu_copy_from();
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_copy_from() const {
        cpp_assert(is_gpu_allocated(), "Cannot copy from unallocated GPU memory()");

        auto* gpu_ptr = gpu_memory();
        auto* cpu_ptr = as_derived().memory_start();

        cpp_assert(!std::is_const<std::remove_pointer_t<decltype(gpu_ptr)>>::value, "copy_from should not be used on const memory");
        cpp_assert(!std::is_const<value_type>::value, "copy_from should not be used on const memory");

        cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(cpu_ptr),
            const_cast<std::remove_const_t<value_type>*>(gpu_ptr),
            etl::size(as_derived()) * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

#else

template <typename T, typename D>
struct gpu_able {};

#endif //ETL_CUBLAS_MODE

} //end of namespace etl
