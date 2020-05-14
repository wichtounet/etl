//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brie Contains aligned_vector utility and handling of bool vector
 */

#pragma once

#include "cpp_utils/aligned_vector.hpp"

namespace etl {

/*!
 * \brief A simple std::vector to work with bool.
 *
 * This vector does not have the same behaviour as std::vector<bool>, it's
 * a real vector without proxy reference.
 */
template <typename T, typename Alloc>
struct simple_vector {
    using value_type = T;

    simple_vector() = default;

    simple_vector(const simple_vector& rhs) : size(rhs.size) {
        resize_impl(rhs.size, false);

        for (size_t i = 0; i < size; ++i) {
            _data[i] = rhs._data[i];
        }
    }

    simple_vector(simple_vector&& rhs) : size(rhs.size) {
        _data     = rhs._data;
        rhs._data = nullptr;
    }

    simple_vector& operator=(const simple_vector& rhs) {
        if (this != &rhs) {
            resize_impl(rhs.size, false);

            size = rhs.size;

            for (size_t i = 0; i < size; ++i) {
                _data[i] = rhs._data[i];
            }
        }

        return *this;
    }

    simple_vector& operator=(simple_vector&& rhs) {
        if (this != &rhs) {
            release();

            size = rhs.size;

            _data     = rhs._data;
            rhs._data = nullptr;
        }

        return *this;
    }

    ~simple_vector() {
        release();
    }

    void resize(size_t n) {
        resize_impl(n, true);
    }

    T& operator[](size_t i) {
        return _data[i];
    }

    const T& operator[](size_t i) const {
        return _data[i];
    }

private:
    void release() {
        if (_data) {
            //In case of non-trivial type, we need to call the destructors
            if constexpr (!std::is_trivial<T>::value) {
                for (size_t i = 0; i < size; ++i) {
                    _data[i].~T();
                }
            }

            allocator.deallocate(_data, size);

            _data = nullptr;
        }
    }

    void resize_impl(size_t n, bool copy = true) {
        auto* new_data = allocator.allocate(n);

        // Call all the constructors if necessary
        if constexpr (!std::is_trivial<T>::value) {
            new (new_data) T[n]();
        }

        // Initialize to the default values
        if constexpr (std::is_trivial<T>::value) {
            std::fill_n(new_data, n, T());
        }

        if (copy && _data) {
            for (size_t i = 0; i < size && i < n; ++i) {
                new_data[i] = _data[i];
            }
        }

        release();

        _data = new_data;
        size  = n;
    }

    T* _data    = nullptr;
    size_t size = 0;
    Alloc allocator;
};

template <typename T, std::size_t A>
struct aligned_vector_impl {
    using type = std::vector<T, cpp::aligned_allocator<T, A>>;
};

template <std::size_t A>
struct aligned_vector_impl<bool, A> {
    using type = simple_vector<bool, cpp::aligned_allocator<bool, A>>;
};

template <typename T, std::size_t A>
using aligned_vector = typename aligned_vector_impl<T, A>::type;

} //end of namespace etl
