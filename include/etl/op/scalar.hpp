//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"

namespace etl {

template <typename T>
struct scalar {
    using value_type = T;

    template<typename V = default_vec>
    using vec_type   = typename V::template vec_type<T>;

    const T value;

    explicit constexpr scalar(T v)
            : value(v) {}

    constexpr const T operator[](std::size_t /*d*/) const noexcept {
        return value;
    }

    constexpr const T read_flat(std::size_t /*d*/) const noexcept {
        return value;
    }

    template<typename V = default_vec>
    constexpr const vec_type<V> load(std::size_t /*d*/) const noexcept {
        return V::set(value);
    }

    template <typename... S>
    T operator()(S... /*args*/) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value;
    }
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>, void> {
    static constexpr const bool is_etl                 = true;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view = false;
    static constexpr const bool is_magic_view = false;
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = true;
    static constexpr const bool vectorizable            = true;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const scalar<T>& s) {
    return os << s.value;
}

} //end of namespace etl
