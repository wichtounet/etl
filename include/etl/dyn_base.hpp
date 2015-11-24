//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <array>     //To store the dimensions
#include <tuple>     //For TMP stuff
#include <algorithm> //For std::find_if
#include <iosfwd>    //For stream support

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/traits_lite.hpp" //forward declaration of the traits
#include "etl/compat.hpp"      //To make it work with g++

namespace etl {

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

template <typename... V>
struct values_t {
    const std::tuple<V...> values;
    explicit values_t(V... v)
            : values(v...){};

    template <typename T, std::size_t... I>
    std::vector<T> list_sub(const std::index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }

    template <typename T>
    std::vector<T> list() const {
        return list_sub<T>(std::make_index_sequence<sizeof...(V)>());
    }
};

template <typename... V>
values_t<V...> values(V... v) {
    return values_t<V...>{v...};
}

namespace dyn_detail {

template <typename... S>
struct is_init_constructor : std::false_type {};

template <typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> : std::is_same<init_flag_t, typename cpp::nth_type<1 + sizeof...(S), S1, S2, S3, S...>::type> {};

template <typename... S>
struct is_initializer_list_constructor : std::false_type {};

template <typename S1, typename S2, typename... S>
struct is_initializer_list_constructor<S1, S2, S...> : cpp::is_specialization_of<std::initializer_list, typename cpp::last_type<S1, S2, S...>::type> {};

inline std::size_t size(std::size_t first) {
    return first;
}

template <typename... T>
inline std::size_t size(std::size_t first, T... args) {
    return first * size(args...);
}

template <std::size_t... I, typename... T>
inline std::size_t size(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return size((cpp::nth_value<I>(args...))...);
}

template <std::size_t... I, typename... T>
inline std::array<std::size_t, sizeof...(I)> sizes(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return {{static_cast<std::size_t>(cpp::nth_value<I>(args...))...}};
}

} // end of namespace dyn_detail

/*!
 * \brief Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, std::size_t D>
struct dyn_base {
    static_assert(D > 0, "A matrix must have a least 1 dimension");

protected:
    static constexpr const std::size_t n_dimensions = D;
    static constexpr const std::size_t alignment    = intrinsic_traits<T>::alignment;

    using value_type             = T;
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;
    using memory_type            = value_type*;
    using const_memory_type      = const value_type*;
    using vec_type               = intrinsic_type<T>;

    std::size_t _size;
    dimension_storage_impl _dimensions;

    void check_invariants() {
        cpp_assert(_dimensions.size() == D, "Invalid dimensions");

#ifndef NDEBUG
        auto computed = std::accumulate(_dimensions.begin(), _dimensions.end(), std::size_t(1), std::multiplies<std::size_t>());
        cpp_assert(computed == _size, "Incoherency in dimensions");
#endif
    }

    static memory_type allocate(std::size_t n) {
        auto* memory = aligned_allocator<void, alignment>::template allocate<T>(n);
        cpp_assert(memory, "Impossible to allocate memory for dyn_matrix");
        cpp_assert(reinterpret_cast<uintptr_t>(memory) % alignment == 0, "Failed to align memory of matrix");

        //In case of non-trivial type, we need to call the constructors
        if(!std::is_trivial<value_type>::value){
            new (memory) value_type[n]();
        }

        return memory;
    }

    static void release(memory_type ptr, std::size_t n) {
        //In case of non-trivial type, we need to call the destructors
        if(!std::is_trivial<value_type>::value){
            for(std::size_t i = 0; i < n; ++i){
                ptr[i].~value_type();
            }
        }

        aligned_allocator<void, alignment>::template release<T>(ptr);
    }

    dyn_base() noexcept : _size(0) {
        std::fill(_dimensions.begin(), _dimensions.end(), 0);

        check_invariants();
    }

    dyn_base(const dyn_base& rhs) = default;
    dyn_base(dyn_base&& rhs) = default;

    dyn_base(std::size_t size, dimension_storage_impl dimensions) noexcept : _size(size), _dimensions(dimensions) {
        check_invariants();
    }

    template <typename E>
    dyn_base(E&& rhs) : _size(etl::size(rhs)) {
        for (std::size_t d = 0; d < etl::dimensions(rhs); ++d) {
            _dimensions[d] = etl::dim(rhs, d);
        }

        check_invariants();
    }

public:

    // Accessors

    std::size_t size() const noexcept {
        return _size;
    }

    std::size_t rows() const noexcept {
        return _dimensions[0];
    }

    std::size_t columns() const noexcept {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }
};

} //end of namespace etl
