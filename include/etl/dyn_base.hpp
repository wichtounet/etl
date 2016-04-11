//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Base class and utilities for dyn matrix implementations
 */

#pragma once

namespace etl {

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

/*!
 * \brief Simple collection of values to initialize a dyn matrix
 */
template <typename... V>
struct values_t {
    const std::tuple<V...> values; ///< The contained values

    /*!
     * \brief Construct a new sequence of values.
     */
    explicit values_t(V... v)
            : values(v...){};

    /*!
     * \brief Returns the sequence of values as a std::vector
     * \return a std::vector containing all the values
     */
    template <typename T>
    std::vector<T> list() const {
        return list_sub<T>(std::make_index_sequence<sizeof...(V)>());
    }

private:
    /*!
     * \brief Returns the sequence of values as a std::vector
     */
    template <typename T, std::size_t... I>
    std::vector<T> list_sub(const std::index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }
};

/*!
 * \brief Create a list of values for initializing a dyn_matrix
 */
template <typename... V>
values_t<V...> values(V... v) {
    return values_t<V...>{v...};
}

namespace dyn_detail {

template <typename... S>
struct is_init_constructor : std::false_type {};

template <typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> : std::is_same<init_flag_t, typename cpp::nth_type<sizeof...(S), S2, S3, S...>::type> {};

template <typename... S>
struct is_initializer_list_constructor : std::false_type {};

template <typename S1, typename S2, typename... S>
struct is_initializer_list_constructor<S1, S2, S...> : cpp::is_specialization_of<std::initializer_list, typename cpp::last_type<S2, S...>::type> {};

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
    static constexpr const std::size_t n_dimensions = D;                              ///< The number of dimensions
    static constexpr const std::size_t alignment    = intrinsic_traits<T>::alignment; ///< The memory alignment

    using value_type             = T;                                     ///< The value type
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>; ///< The type used to store the dimensions
    using memory_type            = value_type*;                           ///< The memory type
    using const_memory_type      = const value_type*;                     ///< The const memory type

    std::size_t _size;                  ///< The size of the matrix
    dimension_storage_impl _dimensions; ///< The dimensions of the matrix

    /*!
     * \brief Verify some invariants with assertions
     *
     * This function should only be used internally to ensure that
     * no breaking changes are made.
     */
    void check_invariants() {
        cpp_assert(_dimensions.size() == D, "Invalid dimensions");

#ifndef NDEBUG
        auto computed = std::accumulate(_dimensions.begin(), _dimensions.end(), std::size_t(1), std::multiplies<std::size_t>());
        cpp_assert(computed == _size, "Incoherency in dimensions");
#endif
    }

    /*!
     * \brief Allocate aligned memory for n elements of the given type
     * \tparam M the type of objects to allocate
     * \param n The number of elements to allocate
     * \return The allocated memory
     */
    template <typename M = value_type>
    static M* allocate(std::size_t n) {
        M* memory = aligned_allocator<void, alignment>::template allocate<M>(n);
        cpp_assert(memory, "Impossible to allocate memory for dyn_matrix");
        cpp_assert(reinterpret_cast<uintptr_t>(memory) % alignment == 0, "Failed to align memory of matrix");

        //In case of non-trivial type, we need to call the constructors
        if (!std::is_trivial<M>::value) {
            new (memory) M[n]();
        }

        return memory;
    }

    /*!
     * \brief Release aligned memory for n elements of the given type
     * \param ptr Pointer to the memory to release
     * \param n The number of elements to release
     */
    template <typename M>
    static void release(M* ptr, std::size_t n) {
        //In case of non-trivial type, we need to call the destructors
        if (!std::is_trivial<M>::value) {
            for (std::size_t i = 0; i < n; ++i) {
                ptr[i].~M();
            }
        }

        aligned_allocator<void, alignment>::template release<M>(ptr);
    }

    /*!
     * \brief Initialize the dyn_base with a size of 0
     */
    dyn_base() noexcept : _size(0) {
        std::fill(_dimensions.begin(), _dimensions.end(), 0);

        check_invariants();
    }

    /*!
     * \brief Copy construct a dyn_base
     * \param rhs The dyn_base to copy from
     */
    dyn_base(const dyn_base& rhs) noexcept = default;

    /*!
     * \brief Move construct a dyn_base
     * \param rhs The dyn_base to move from
     */
    dyn_base(dyn_base&& rhs) noexcept = default;

    /*!
     * \brief Construct a dyn_base if the given size and dimensions
     * \param size The size of the matrix
     * \param dimensions The dimensions of the matrix
     */
    dyn_base(std::size_t size, dimension_storage_impl dimensions) noexcept : _size(size), _dimensions(dimensions) {
        check_invariants();
    }

    /*!
     * \brief Move construct a dyn_base
     * \param rhs The dyn_base to move from
     */
    template <typename E>
    dyn_base(E&& rhs)
            : _size(etl::size(rhs)) {
        for (std::size_t d = 0; d < etl::dimensions(rhs); ++d) {
            _dimensions[d] = etl::dim(rhs, d);
        }

        check_invariants();
    }

public:
    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    std::size_t size() const noexcept {
        return _size;
    }

    /*!
     * \brief Returns the number of rows of the matrix (the first dimension)
     * \return The number of rows of the matrix
     */
    std::size_t rows() const noexcept {
        return _dimensions[0];
    }

    /*!
     * \brief Returns the number of columns of the matrix (the first dimension)
     * \return The number of columns of the matrix
     */
    std::size_t columns() const noexcept {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }

    /*!
     * \brief Returns the dth dimension of the matrix
     * \param d The dimension to get
     * \return The Dth dimension of the matrix
     */
    std::size_t dim(std::size_t d) const noexcept {
        cpp_assert(d < n_dimensions, "Invalid dimension");

        return _dimensions[d];
    }

    /*!
     * \brief Returns the D2th dimension of the matrix
     * \return The D2th dimension of the matrix
     */
    template <std::size_t D2>
    std::size_t dim() const noexcept {
        cpp_assert(D2 < n_dimensions, "Invalid dimension");

        return _dimensions[D2];
    }
};

} //end of namespace etl
