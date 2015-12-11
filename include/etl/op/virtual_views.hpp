//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd> //For stream support

#include "cpp_utils/tmp.hpp"

#include "etl/tmp.hpp"

namespace etl {

namespace detail {

template <typename V>
V compute(std::size_t n, std::size_t i, std::size_t j) {
    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return i == 0 && j == 0 ? 1 : i == 0 && j == 1 ? 3 : i == 1 && j == 0 ? 4
                                                                              : 2;
    } else {
        //Siamese method
        return n * (((i + 1) + (j + 1) - 1 + n / 2) % n) + (((i + 1) + 2 * (j + 1) - 2) % n) + 1;
    }
}

} //end of namespace detail

//Note: Matrix of even order > 2 are only pseudo-magic
//TODO Add algorithm for even order
template <typename V>
struct magic_view {
    using value_type = V;

    const std::size_t n;

    explicit magic_view(std::size_t n)
            : n(n) {}

    value_type operator[](std::size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    value_type read_flat(std::size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    value_type operator[](std::size_t i) {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    value_type operator()(std::size_t i, std::size_t j) {
        return detail::compute<value_type>(n, i, j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return detail::compute<value_type>(n, i, j);
    }
};

template <typename V, std::size_t N>
struct fast_magic_view {
    using value_type = V;

    value_type operator[](std::size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    value_type read_flat(std::size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    value_type operator[](std::size_t i) {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    value_type operator()(std::size_t i, std::size_t j) {
        return detail::compute<value_type>(N, i, j);
    }

    value_type operator()(std::size_t i, std::size_t j) const {
        return detail::compute<value_type>(N, i, j);
    }
};

template <typename V>
struct etl_traits<etl::magic_view<V>> {
    using expr_t = etl::magic_view<V>;

    static constexpr const bool is_etl                 = true;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view = false;
    static constexpr const bool is_magic_view = true;
    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;

    static std::size_t size(const expr_t& v) {
        return v.n * v.n;
    }

    static std::size_t dim(const expr_t& v, std::size_t /*unused*/) {
        return v.n;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

template <std::size_t N, typename V>
struct etl_traits<etl::fast_magic_view<V, N>> {
    using expr_t = etl::fast_magic_view<V, N>;

    static constexpr const bool is_etl                 = true;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view = false;
    static constexpr const bool is_magic_view = true;
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;

    static constexpr std::size_t size() {
        return N * N;
    }

    static std::size_t size(const expr_t& /*unused*/) {
        return N * N;
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return N;
    }

    static std::size_t dim(const expr_t& /*e*/, std::size_t /*unused*/) {
        return N;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

} //end of namespace etl
