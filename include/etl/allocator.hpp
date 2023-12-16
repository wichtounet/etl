//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains allocation utilities
 */

#pragma once

namespace etl {
/*
 * GCC mangling of vector types (__m128, __m256, ...) is terribly
 * broken. To avoid this, the chosen solution is to use special
 * functions for allocations of these types
 */

/*!
 * \brief Use of this type in the parameter with the size of
 * a vector type fakes mangling
 */
template <size_t T>
struct mangling_faker {};

/*!
 * \brief Test if the given type can be mangled correctly
 *
 */
template <typename T>
concept is_mangle_able = std::same_as<std::decay_t<T>, float> || std::same_as<std::decay_t<T>, double>
                                || cpp::specialization_of<std::complex, T> || cpp::specialization_of<etl::complex, T>;

/*!
 * \brief Allocated for aligned memory
 * \tparam A The alignment
 */
template <size_t A>
struct aligned_allocator {
    /*!
     * \brief Allocate a block of memory of *size* elements
     * \param size The number of elements
     * \return A pointer to the allocated memory
     */
    template <typename T, size_t S = sizeof(T)>
    static T* allocate(size_t size, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
        auto required_bytes = sizeof(T) * size;
        auto offset         = (A - 1) + sizeof(uintptr_t);
        auto orig           = malloc(required_bytes + offset);

        if (!orig) {
            return nullptr;
        }

        auto aligned = reinterpret_cast<void**>((reinterpret_cast<size_t>(orig) + offset) & ~(A - 1));
        aligned[-1]  = orig;
        return reinterpret_cast<T*>(aligned);
    }

    /*!
     * \brief Release the memory
     * \param ptr The pointer to the memory to be released
     */
    template <typename T, size_t S = sizeof(T)>
    static void release(T* ptr, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
        //Note the const_cast is only to allow compilation
        free((reinterpret_cast<void**>(const_cast<std::remove_const_t<T>*>(ptr)))[-1]);
    }
};

/*!
 * \brief Allocate an array of the given size for the given type
 * \param size The number of elements
 * \return An unique pointer to the memory
 */
template <typename T, size_t S = sizeof(T)>
auto allocate(size_t size, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
    static_assert(is_mangle_able<T>, "allocate does not work with vector types");
    return std::make_unique<T[]>(size);
}

/*!
 * \brief Allocate an aligned rray of the given size for the given type
 * \param size The number of elements
 * \return A pointer to the aligned memory
 */
template <typename T, size_t S = sizeof(T)>
T* aligned_allocate(size_t size, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
    return aligned_allocator<32>::allocate<T>(size);
}

/*!
 * \brief Release some aligned memory
 * \param ptr The ptr to the aligned memory
 */
template <typename T, size_t S = sizeof(T)>
void aligned_release(T* ptr, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
    return aligned_allocator<32>::release<T>(ptr);
}

/*!
 * \brief RAII wrapper for allocated aligned memory
 */
template <typename T, size_t S = sizeof(T)>
struct aligned_ptr {
    T* ptr; ///< The raw pointer

    /*!
     * \brief Build an aligned_ptr managing the given pointer
     */
    explicit aligned_ptr(T* ptr) : ptr(ptr) {}

    aligned_ptr(const aligned_ptr& rhs) = delete;
    aligned_ptr& operator=(const aligned_ptr& rhs) = delete;

    /*!
     * \brief Move construct an aligned_ptr
     * \param rhs The pointer to move
     */
    aligned_ptr(aligned_ptr&& rhs) noexcept : ptr(rhs.ptr) {
        rhs.ptr = nullptr;
    }

    /*!
     * \brief Move assign an aligned_ptr
     * \param rhs The pointer to move
     * \return the aligned_ptr
     */
    aligned_ptr& operator=(aligned_ptr&& rhs) noexcept {
        if (this != &rhs) {
            ptr     = rhs.ptr;
            rhs.ptr = nullptr;
        }

        return *this;
    }

    /*!
     * \brief Returns a reference to the element at psition i
     */
    inline T& operator[](size_t i) {
        return ptr[i];
    }

    /*!
     * \brief Returns a reference to the element at psition i
     */
    inline const T& operator[](size_t i) const {
        return ptr[i];
    }

    /*!
     * \brief Destruct the aligned_ptr and release the aligned memory
     */
    ~aligned_ptr() {
        if (ptr) {
            aligned_release(ptr);
        }
    }

    /*!
     * \brief Returns the raw underlying pointer
     */
    T* get() {
        return ptr;
    }
};

/*!
 * \brief Allocate an aligned rray of the given size for the given type
 * \param size The number of elements
 * \return A pointer to the aligned memory
 */
template <typename T, size_t S = sizeof(T)>
aligned_ptr<T> aligned_allocate_auto(size_t size, mangling_faker<S> /*unused*/ = mangling_faker<S>()) {
    return aligned_ptr<T>{aligned_allocate<T>(size)};
}

} //end of namespace etl
