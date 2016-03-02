//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDA
#include "cuda_runtime.h"
#endif

namespace etl {

namespace impl {

namespace cuda {

#ifdef ETL_CUDA

/*!
 * \brief Wrapper for CUDA memory
 */
template <typename T>
struct cuda_memory {
    T* memory;

    cuda_memory() noexcept : memory(nullptr) {}
    cuda_memory(T* memory) noexcept : memory(memory) {}

    //Delete copy operations
    cuda_memory(const cuda_memory& rhs) noexcept : memory(nullptr) {
        cpp_assert(!rhs.is_set(), "copy of cuda_memory is only possible when not allocated");
    }

    cuda_memory& operator=(const cuda_memory& rhs) noexcept {
        if(this != &rhs){
            cpp_assert(!is_set(), "copy of cuda_memory is only possible when not allocated");
            cpp_assert(!rhs.is_set(), "copy of cuda_memory is only possible when not allocated");
        }

        return *this;
    }

    cuda_memory(cuda_memory&& rhs) noexcept : memory(rhs.memory){
        rhs.memory = nullptr;
    }

    cuda_memory& operator=(cuda_memory&& rhs) noexcept {
        if(this != &rhs){
            free_memory();

            memory = rhs.memory;
            rhs.memory = nullptr;
        }

        return *this;
    }

    cuda_memory& operator=(T* new_memory){
        free_memory();

        memory = new_memory;

        return *this;
    }

    T* get() const {
        return memory;
    }

    bool is_set() const {
        return memory;
    }

    void reset(){
        free_memory();
        memory = nullptr;
    }

    ~cuda_memory() {
        free_memory();
    }

private:
    void free_memory(){
        if(memory){
            //Note: the const_cast is only here to allow compilation
            cudaFree((reinterpret_cast<void*>(const_cast<std::remove_const_t<T>*>(memory))));
        }
    }
};

template <typename E>
auto cuda_allocate(const E& expr, bool copy = false) -> cuda_memory<value_t<E>> {
    value_t<E>* memory;

    auto cuda_status = cudaMalloc(&memory, etl::size(expr) * sizeof(value_t<E>));

    if (cuda_status != cudaSuccess) {
        std::cout << "cuda: Failed to allocate GPU memory: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cout << "      Tried to allocate " << etl::size(expr) * sizeof(value_t<E>) << "B" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (copy) {
        cudaMemcpy(memory, expr.memory_start(), etl::size(expr) * sizeof(value_t<E>), cudaMemcpyHostToDevice);
    }

    return {memory};
}

template <typename E>
auto cuda_allocate_copy(const E& expr) -> cuda_memory<value_t<E>> {
    return cuda_allocate(expr, true);
}

template <typename E>
auto cuda_allocate(E* ptr, std::size_t n, bool copy = false) -> cuda_memory<E> {
    E* memory;

    auto cuda_status = cudaMalloc(&memory, n * sizeof(E));

    if (cuda_status != cudaSuccess) {
        std::cout << "cuda: Failed to allocate GPU memory: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cout << "      Tried to allocate " << n * sizeof(E) << "B" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (copy) {
        cudaMemcpy(memory, ptr, n * sizeof(E), cudaMemcpyHostToDevice);
    }

    return {memory};
}

template <typename E>
auto cuda_allocate_copy(E* ptr, std::size_t n) -> cuda_memory<E> {
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

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
