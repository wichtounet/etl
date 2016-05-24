//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/cublas/cuda.hpp"

namespace etl {

#ifdef ETL_CUDA
template<typename T>
using gpu_handler = impl::cuda::cuda_memory<T>;
#else
template<typename>
using gpu_handler = int;
#endif

//TODO Remove the duplication of fields between both implementations

template <typename T, std::size_t D>
struct opaque_memory {
    static constexpr const std::size_t n_dimensions = D;                      ///< The number of dimensions

    using value_type        = T;
    using memory_type       = T*;
    using const_memory_type = std::add_const_t<T>*;

    T* memory;
    const std::size_t etl_size;
    const std::array<std::size_t, D> dims;
    gpu_handler<T>& _gpu_memory_handler;
    const order storage_order; ///< The storage order

    opaque_memory(const T* memory, std::size_t size, const std::array<std::size_t, D>& dims, const gpu_handler<T>& handler, order storage_order)
            : memory(const_cast<T*>(memory)),
              etl_size(size),
              dims(dims),
              _gpu_memory_handler(const_cast<gpu_handler<T>&>(handler)),
              storage_order(storage_order) {
        //Nothing else to init
    }

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
    }

    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    std::size_t size() const noexcept {
        return etl_size;
    }

    /*!
     * \brief Returns the Dth dimension of the matrix
     * \return The Dth dimension of the matrix
     */
    template <std::size_t DD>
    std::size_t dim() const noexcept {
        return dims[DD];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return &memory[0];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return &memory[0];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return &memory[size()];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return &memory[size()];
    }

#ifdef ETL_CUDA
    /*!
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return _gpu_memory_handler.is_set();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if (is_gpu_allocated()) {
            _gpu_memory_handler.reset();
        }
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return _gpu_memory_handler.get();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate() const {
        _gpu_memory_handler = impl::cuda::cuda_allocate_only<T>(etl_size);
    }

    /*!
     * \brief Allocate memory on the GPU for the expression, only if not already allocated
     */
    void gpu_allocate_if_necessary() const {
        if (!is_gpu_allocated()) {
            gpu_allocate();
        }
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void gpu_allocate_copy() const {
        _gpu_memory_handler = impl::cuda::cuda_allocate_copy(memory, etl_size);
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU, only if not already allocated.
     */
    void gpu_allocate_copy_if_necessary() const {
        if (!is_gpu_allocated()) {
            gpu_allocate_copy();
        }
    }

    /*!
     * \brief Reallocate the GPU memory.
     * \param memory The new GPU memory (will be moved)
     */
    void gpu_reallocate(impl::cuda::cuda_memory<T>&& memory) {
        _gpu_memory_handler = std::move(memory);
    }

    /*!
     * \brief Release the GPU memory for another expression to use
     * \return A rvalue reference to the _gpu_memory_handler.
     */
    impl::cuda::cuda_memory<T>&& gpu_release() const {
        return std::move(_gpu_memory_handler);
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void gpu_copy_from_if_necessary() const {
        if (is_gpu_allocated()) {
            gpu_copy_from();
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_copy_from() const {
        cpp_assert(is_gpu_allocated(), "Cannot copy from unallocated GPU memory()");

        auto* gpu_ptr = gpu_memory();
        auto* cpu_ptr = memory;

        cpp_assert(!std::is_const<std::remove_pointer_t<decltype(gpu_ptr)>>::value, "copy_from should not be used on const memory");
        cpp_assert(!std::is_const<value_type>::value, "copy_from should not be used on const memory");

        cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(cpu_ptr),
            const_cast<std::remove_const_t<value_type>*>(gpu_ptr),
            etl_size * sizeof(T), cudaMemcpyDeviceToHost);
    }
#else
    /*!
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return false;
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {}

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return nullptr;
    }

    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate() const {}

    /*!
     * \brief Allocate memory on the GPU for the expression, only if not already allocated
     */
    void gpu_allocate_if_necessary() const {}

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void gpu_allocate_copy() const {}

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU, only if not already allocated.
     */
    void gpu_allocate_copy_if_necessary() const {}

    /*!
     * \brief Reallocate the GPU memory.
     * \param memory The new GPU memory (will be moved)
     */
    void gpu_reallocate(impl::cuda::cuda_memory<T>&& memory) {
        cpp_unused(memory);
    }

    /*!
     * \brief Release the GPU memory for another expression to use
     * \return A rvalue reference to the gpu_memory_handler.
     */
    impl::cuda::cuda_memory<T>&& gpu_release() const {}

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void gpu_copy_from_if_necessary() const {}

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_copy_from() const {}
#endif
};

} //end of namespace etl
