//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory>

namespace etl {

/*!
 * \brief Allocated for aligned memory
 * \tparam Expr The expression typoe
 * \tparam A The alignment
 */
template <typename Expr, std::size_t A>
struct aligned_allocator {
    /*!
     * \brief Allocate a block of memory of *size* elements
     * \param size The number of elements
     * \return A pointer to the allocated memory
     */
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

    /*!
     * \brief Release the memory
     * \param ptr The pointer to the memory to be released
     */
    template <typename T>
    static void release(T* ptr) {
        //Note the const_cast is only to allow compilation
        free((reinterpret_cast<void**>(const_cast<std::remove_const_t<T>*>(ptr)))[-1]);
    }
};


/*!
 * \brief Allocate an array of the given size for the given type
 * \param size The number of elements
 * \return An unique pointer to the memory
 */
template <typename T>
auto allocate(std::size_t size) {
    return std::make_unique<T[]>(size);
}

/*!
 * \brief Allocate an aligned rray of the given size for the given type
 * \param size The number of elements
 * \return A pointer to the aligned memory
 */
template <typename T>
T* aligned_allocate(std::size_t size) {
    return aligned_allocator<void, 32>::allocate<T>(size);
}

/*!
 * \brief Release some aligned memory
 * \param ptr The ptr to the aligned memory
 */
template <typename T>
void aligned_release(T* ptr) {
    return aligned_allocator<void, 32>::release<T>(ptr);
}

} //end of namespace etl
