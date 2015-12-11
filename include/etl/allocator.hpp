//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory>

namespace etl {

template <typename Expr, std::size_t A>
struct aligned_allocator {
    template <typename T>
    static T* allocate(std::size_t size) {
        auto required_bytes = sizeof(T) * size;
        auto offset         = (A - 1) + sizeof(uintptr_t);
        auto orig           = malloc(required_bytes + offset);

        if (!orig) {
            return nullptr;
        }

        auto aligned = reinterpret_cast<void**>((reinterpret_cast<size_t>(orig) + offset) & ~(A - 1));
        aligned[-1] = orig;
        return reinterpret_cast<T*>(aligned);
    }

    template <typename T>
    static void release(T* ptr) {
        free((reinterpret_cast<void**>(ptr))[-1]);
    }
};

template <typename T>
auto allocate(std::size_t size) {
    return std::make_unique<T[]>(size);
}

template <typename T>
T* aligned_allocate(std::size_t size) {
    return aligned_allocator<void, 32>::allocate<T>(size);
}

template <typename T>
void aligned_release(T* ptr) {
    return aligned_allocator<void, 32>::release<T>(ptr);
}

} //end of namespace etl
