//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cuda_runtime.h"

namespace etl {

namespace impl {

namespace cublas {

template<typename T>
struct cuda_memory {
    T* memory;
    cuda_memory(T* memory) : memory(memory) {}
    T* get(){
        return memory;
    }
    ~cuda_memory(){
        cudaFree(memory);
    }
};

template<typename E>
auto cuda_allocate(const E& expr, bool copy = false) -> cuda_memory<value_t<E>> {
    value_t<E>* memory;

    auto cuda_status = cudaMalloc(&memory, etl::size(expr) * sizeof(value_t<E>));

    if (cuda_status != cudaSuccess) {
        std::cout << "Failed to allocate GPU memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(copy){
        cudaMemcpy(memory, expr.memory_start(), etl::size(expr) * sizeof(value_t<E>), cudaMemcpyHostToDevice);
    }

    return {memory};
}

template<typename E>
auto cuda_allocate_copy(const E& expr) -> cuda_memory<value_t<E>> {
    return cuda_allocate(expr, true);
}

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
