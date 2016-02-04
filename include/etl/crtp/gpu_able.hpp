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
            std::cout << "gpu:: evict " << this << std::endl;
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
        std::cout << "gpu:: allocate " << this << std::endl;
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
        std::cout << "gpu:: allocate_copy " << this << std::endl;
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
        std::cout << "gpu:: reallocate" << this << std::endl;
        gpu_memory_handler = std::move(memory);
    }

    /*!
     * \brief Release the GPU memory for another expression to use
     * \return A rvalue reference to the gpu_memory_handler.
     */
    impl::cuda::cuda_memory<T>&& gpu_release() const {
        std::cout << "gpu:: release" << this << std::endl;
        return std::move(gpu_memory_handler);
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_copy_from(){
        std::cout << "gpu:: copy_back" << this << std::endl;
        cpp_assert(is_gpu_allocated(), "Cannot copy from unallocated GPU memory()");
        cudaMemcpy(as_derived().memory_start(), gpu_memory(), etl::size(as_derived()) * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

#else

template <typename T, typename D>
struct gpu_able {};

#endif //ETL_CUBLAS_MODE

} //end of namespace etl
