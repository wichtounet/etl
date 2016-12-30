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

/*!
 * \brief Wrapper for opaque memory.
 *
 * This is used to give direct memory access to any expressions
 * without showing the complete type, acting as a form of type
 * erasure
 */
template <typename T, std::size_t D>
struct opaque_memory {
    static constexpr std::size_t n_dimensions = D; ///< The number of dimensions

    using value_type        = T;                    ///< The type of value
    using memory_type       = T*;                   ///< The type of memory
    using const_memory_type = std::add_const_t<T>*; ///< The type of const memory

private:
    T* memory;                             ///< The memory pointer
    const std::size_t etl_size;            ///< The full size of the matrix
    const std::array<std::size_t, D> dims; ///< The dimensions of the matrix
    gpu_handler<T>& _gpu_memory_handler;   ///< The GPU memory handler
    const order storage_order;             ///< The storage order

public:
    /*!
     * \brief Create a new opaque memory
     * \param memory The pointer to memory
     * \param size The size of the memory
     * \param dims The dimensions
     * \param handler The GPU memory handler
     * \param storage_order The Storage order
     */
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
     * \brief Returns the DDth dimension of the matrix
     * \return The DDth dimension of the matrix
     */
    std::size_t dim(std::size_t DD) const noexcept {
        return dims[DD];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() const noexcept {
        return &memory[0];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() const noexcept {
        return &memory[etl_size];
    }

#ifdef ETL_CUDA
private:
    /*!
     * \brief Allocate memory on the GPU for the expression
     */
    void gpu_allocate_impl() const {
        cpp_assert(!is_gpu_allocated(), "Trying to allocate already allocated GPU memory");

        _gpu_memory_handler = impl::cuda::cuda_allocate_only<T>(etl_size);

        inc_counter("gpu:allocate");
    }

    /*!
     * \brief Copy back from the CPU to the GPU
     */
    void cpu_to_gpu() const {
        cpp_assert(is_gpu_allocated(), "Cannot copy to unallocated GPU memory");
        cpp_assert(!_gpu_memory_handler.gpu_up_to_date, "Copy must only be done if necessary");
        cpp_assert(_gpu_memory_handler.cpu_up_to_date, "Copy from invalid memory!");

        auto* gpu_ptr = gpu_memory();
        auto* cpu_ptr = memory;

        cuda_check(cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(gpu_ptr),
            const_cast<std::remove_const_t<value_type>*>(cpu_ptr),
            etl_size * sizeof(T), cudaMemcpyHostToDevice));

        _gpu_memory_handler.gpu_up_to_date = true;

        inc_counter("gpu:cpu_to_gpu");
    }

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void gpu_to_cpu() const {
        cpp_assert(is_gpu_allocated(), "Cannot copy from unallocated GPU memory()");
        cpp_assert(_gpu_memory_handler.gpu_up_to_date, "Cannot copy from invalid memory");
        cpp_assert(!_gpu_memory_handler.cpu_up_to_date, "Copy done without reason");

        auto* gpu_ptr = gpu_memory();
        auto* cpu_ptr = memory;

        cpp_assert(!std::is_const<std::remove_pointer_t<decltype(gpu_ptr)>>::value, "copy_from should not be used on const memory");
        cpp_assert(!std::is_const<value_type>::value, "copy_from should not be used on const memory");

        cuda_check(cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(cpu_ptr),
            const_cast<std::remove_const_t<value_type>*>(gpu_ptr),
            etl_size * sizeof(T), cudaMemcpyDeviceToHost));

        _gpu_memory_handler.cpu_up_to_date = true;

        inc_counter("gpu:gpu_to_cpu");
    }

public:

    /*!
     * \brief Indicates if the expression is allocated in GPU.
     * \return true if the expression is allocated in GPU, false otherwise
     */
    bool is_gpu_allocated() const noexcept {
        return _gpu_memory_handler.is_set();
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return _gpu_memory_handler.get();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if (is_gpu_allocated()) {
            _gpu_memory_handler.reset();
        }

        invalidate_gpu();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        _gpu_memory_handler.cpu_up_to_date = false;
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        _gpu_memory_handler.gpu_up_to_date = false;
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        if (!is_gpu_allocated()) {
            gpu_allocate_impl();
        }

        _gpu_memory_handler.gpu_up_to_date = true;
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        // Make sure there is some memory allocate
        if (!is_gpu_allocated()) {
            gpu_allocate_impl();
        }

        if(!_gpu_memory_handler.gpu_up_to_date){
            cpu_to_gpu();
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        if(!_gpu_memory_handler.cpu_up_to_date){
            gpu_to_cpu();
        }
    }

    void transfer_to(opaque_memory& rhs){
        rhs._gpu_memory_handler = std::move(_gpu_memory_handler);

        // The memory was transferred, not up to date anymore
        _gpu_memory_handler.gpu_up_to_date = false;

        // The target is up to date on GPU but CPU is not to date anymore
        rhs._gpu_memory_handler.gpu_up_to_date = true;
        rhs._gpu_memory_handler.cpu_up_to_date = false;
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
     * \brief Allocate memory on the GPU for the expression, only if not already allocated
     */
    void ensure_gpu_allocated() const {}

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {}

    /*!
     * \brief Copy back from the GPU to the expression memory.
     */
    void ensure_cpu_up_to_date() const {}

    void invalidate_gpu() const {}
    void invalidate_cpu() const {}

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
#endif
};

} //end of namespace etl
