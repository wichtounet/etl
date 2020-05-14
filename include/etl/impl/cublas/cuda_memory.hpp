//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define cuda_check(call)                                                                                \
    {                                                                                                   \
        auto status = call;                                                                             \
        if (status != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                           \
        }                                                                                               \
    }

#define cuda_check_assert(call)                                                                         \
    {                                                                                                   \
        auto status = call;                                                                             \
        if (status != cudaSuccess) {                                                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                           \
            std::abort();                                                                               \
        }                                                                                               \
    }

#endif

namespace etl::impl::cuda {

#ifdef ETL_CUDA

/*!
 * \brief Wrapper for CUDA memory
 */
template <typename T>
struct cuda_memory {
    T* memory;   ///< Pointer the allocated GPU memory
    size_t size; ///< Size of the GPU memory

    /*!
     * \brief Create a new empty cuda_memory
     */
    cuda_memory() noexcept : memory(nullptr), size(0) {}

    /*!
     * \brief Create a new cuda_memory over existing memory
     */
    cuda_memory(T* memory, size_t size) noexcept : memory(memory), size(size) {}

    /*!
     * \brief Copy construct a new cuda_memory
     */
    cuda_memory(const cuda_memory& rhs) noexcept : memory(nullptr), size(0) {
        cpp_unused(rhs);
        cpp_assert(!rhs.is_set(), "copy of cuda_memory is only possible when not allocated");
    }

    /*!
     * \brief Copy assign from another cuda_memory
     */
    cuda_memory& operator=(const cuda_memory& rhs) noexcept {
        if (this != &rhs) {
            cpp_assert(!is_set(), "copy of cuda_memory is only possible when not allocated");
            cpp_assert(!rhs.is_set(), "copy of cuda_memory is only possible when not allocated");
        }

        return *this;
    }

    /*!
     * \brief Move construct from another cuda_memory
     * \param rhs The cuda_memory from which to move
     */
    cuda_memory(cuda_memory&& rhs) noexcept : memory(rhs.memory), size(rhs.size) {
        rhs.memory = nullptr;
        rhs.size   = 0;
    }

    /*!
     * \brief Move assign from another cuda_memory
     * \param rhs The cuda_memory from which to move
     */
    cuda_memory& operator=(cuda_memory&& rhs) noexcept {
        if (this != &rhs) {
            free_memory();

            memory = rhs.memory;
            size   = rhs.size;

            rhs.memory = nullptr;
            rhs.size   = 0;
        }

        return *this;
    }

    /*!
     * \brief Destruct the cuda_memory.
     *
     * This will free any existing memory
     */
    ~cuda_memory() {
        free_memory();
    }

    /*!
     * \brief Assign a new GPU pointer.
     *
     * This will free any existing GPU memory.
     */
    cuda_memory& operator=(T* new_memory) {
        free_memory();

        memory = new_memory;

        return *this;
    }

    /*!
     * \brief Returns a pointer to the allocated GPU memory
     */
    T* get() const {
        return memory;
    }

    /*!
     * \brief Indicates if the memory is set.
     * \return true if the memory is set, false otherwise
     */
    bool is_set() const {
        return memory;
    }

    /*!
     * \brief Reset the cuda_memory.
     *
     * This will free any existing GPU memory.
     */
    void reset() {
        free_memory();
        memory = nullptr;
    }

private:
    /*!
     * \brief Release any existing allocated memory
     */
    void free_memory() {
        if (memory) {
            gpu_memory_allocator::release(memory, size);
        }
    }
};

/*!
 * \brief Allocate some GPU memory of the given size and type
 * \param size The number of elements to allocate
 * \tparam E The type of the elements
 */
template <typename E>
auto cuda_allocate_only(size_t size) -> cuda_memory<E> {
    auto* memory = gpu_memory_allocator::allocate<E>(size);
    return cuda_memory<E>{memory, size};
}

/*!
 * \brief Allocate some GPU memory of the same size and type as the given ETL expression
 * \param expr The expression from which to allocate
 * \param copy Boolean indicating if a copy of the CPU memory is performed.
 */
template <typename E>
auto cuda_allocate(const E& expr, bool copy = false) -> cuda_memory<value_t<E>> {
    auto* memory = gpu_memory_allocator::allocate<value_t<E>>(etl::size(expr));

    if (copy) {
        cuda_check(cudaMemcpy(memory, expr.memory_start(), etl::size(expr) * sizeof(value_t<E>), cudaMemcpyHostToDevice));
    }

    return cuda_memory<value_t<E>>{memory, etl::size(expr)};
}

/*!
 * \brief Allocate some GPU memory of the same size and type as the given ETL expression and copy from the expression
 * \param expr The expression from which to allocate and copy
 */
template <typename E>
auto cuda_allocate_copy(const E& expr) -> cuda_memory<value_t<E>> {
    return cuda_allocate(expr, true);
}

/*!
 * \brief Allocate some GPU memory of the given size and type, and optionally copy from CPU into the newly allocated GPU memory
 * \param ptr Pointer to the corresponding CPU memory
 * \param n The number of elements to allocate
 * \param copy Boolean indicating if a copy of the CPU memory is performed.
 * \tparam E The type of the elements
 */
template <typename E>
auto cuda_allocate(E* ptr, size_t n, bool copy = false) -> cuda_memory<E> {
    auto* memory = gpu_memory_allocator::allocate<E>(n);

    if (copy) {
        cuda_check(cudaMemcpy(memory, ptr, n * sizeof(E), cudaMemcpyHostToDevice));
    }

    return cuda_memory<E>{memory};
}

/*!
 * \brief Allocate some GPU memory of the given size and type, and copy from CPU into the newly allocated GPU memory
 * \param ptr Pointer to the corresponding CPU memory
 * \param n The number of elements to allocate
 * \tparam E The type of the elements
 */
template <typename T>
auto cuda_allocate_copy(T* ptr, size_t n) -> cuda_memory<T> {
    return cuda_allocate(ptr, n, true);
}

#else

/*!
 * \brief Wrapper for CUDA memory (when disabled CUDA support)
 */
template <typename T>
struct cuda_memory {
    //Nothing is enought
};

#endif

} //end of namespace etl::impl::cuda
