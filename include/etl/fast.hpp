//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains static matrix implementation
 */

#pragma once

#include "cpp_utils/array_wrapper.hpp"

namespace etl {

namespace matrix_detail {

template <typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

template <typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template <typename M, std::size_t I, typename Enable = void>
struct matrix_leadingsize : std::integral_constant<std::size_t, M::template dim<I - 1>() * matrix_leadingsize<M, I - 1>::value> {};

template <typename M>
struct matrix_leadingsize<M, 0> : std::integral_constant<std::size_t, 1> {};

template <typename M, std::size_t I>
inline cpp14_constexpr std::size_t rm_compute_index(std::size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return first;
}

template <typename M, std::size_t I, typename... S>
inline cpp14_constexpr std::size_t rm_compute_index(std::size_t first, std::size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_subsize<M, I>::value * first + rm_compute_index<M, I + 1>(second, args...);
}

template <typename M, std::size_t I>
inline cpp14_constexpr std::size_t cm_compute_index(std::size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize<M, I>::value * first;
}

template <typename M, std::size_t I, typename... S>
inline cpp14_constexpr std::size_t cm_compute_index(std::size_t first, std::size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize<M, I>::value * first + cm_compute_index<M, I + 1>(second, args...);
}

template <typename M, std::size_t I, typename... S>
inline constexpr std::size_t compute_index(S... args) noexcept(assert_nothrow) {
    return M::storage_order == order::ColumnMajor
        ? cm_compute_index<M, I>(args...)
        : rm_compute_index<M, I>(args...);
}

template <typename N>
struct is_vector : std::false_type {};

template <typename N, typename A>
struct is_vector<std::vector<N, A>> : std::true_type {};

template <typename N>
struct is_vector<std::vector<N>> : std::true_type {};

template <typename T>
struct iterator_type {
    using iterator       = typename T::iterator;
    using const_iterator = typename T::const_iterator;
};

template <typename T>
struct iterator_type <T*> {
    using iterator       = T*;
    using const_iterator = const T*;
};

template <typename T>
struct iterator_type <const T*> {
    using iterator       = const T*;
    using const_iterator = const T*;
};

template <typename T>
using iterator_t = typename iterator_type<T>::iterator;

template <typename T>
using const_iterator_t = typename iterator_type<T>::const_iterator;

} //end of namespace matrix_detail

/*!
 * \brief Matrix with compile-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, typename ST, order SO, std::size_t... Dims>
struct fast_matrix_impl final : inplace_assignable<fast_matrix_impl<T, ST, SO, Dims...>>, comparable<fast_matrix_impl<T, ST, SO, Dims...>>, expression_able<fast_matrix_impl<T, ST, SO, Dims...>>, value_testable<fast_matrix_impl<T, ST, SO, Dims...>>, dim_testable<fast_matrix_impl<T, ST, SO, Dims...>>, gpu_able<T, fast_matrix_impl<T, ST, SO, Dims...>> {
    static_assert(sizeof...(Dims) > 0, "At least one dimension must be specified");

public:
    static constexpr const std::size_t n_dimensions = sizeof...(Dims);                      ///< The number of dimensions
    static constexpr const std::size_t etl_size     = mul_all<Dims...>::value;              ///< The size of the matrix
    static constexpr const order storage_order      = SO;                                   ///< The storage order
    static constexpr const bool array_impl          = !matrix_detail::is_vector<ST>::value; ///< true if the storage is an std::arraw, false otherwise

    using value_type        = T;                                             ///< The value type
    using storage_impl      = ST;                                            ///< The storage implementation
    using iterator          = matrix_detail::iterator_t<storage_impl>;       ///< The iterator type
    using const_iterator    = matrix_detail::const_iterator_t<storage_impl>; ///< The const iterator type
    using this_type         = fast_matrix_impl<T, ST, SO, Dims...>;          ///< this type
    using memory_type       = value_type*;                                   ///< The memory type
    using const_memory_type = const value_type*;                             ///< The const memory type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

private:
    storage_impl _data;

    template <typename... S>
    static constexpr std::size_t index(S... args) {
        return matrix_detail::compute_index<this_type, 0>(args...);
    }

    template <typename... S>
    value_type& access(S... args) {
        return _data[index(args...)];
    }

    template <typename... S>
    const value_type& access(S... args) const {
        return _data[index(args...)];
    }

    template <typename S = ST, cpp_enable_if(matrix_detail::is_vector<S>::value)>
    void init() {
        _data.resize(etl_size);
    }

    template <typename S = ST, cpp_disable_if(matrix_detail::is_vector<S>::value)>
    void init() noexcept {
        //Nothing to init
    }

public:
    /// Construction

    /*!
     * \brief Construct an empty fast matrix
     */
    fast_matrix_impl() noexcept {
        init();
    }

    template <typename VT, cpp_enable_if_or(std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value)>
    explicit fast_matrix_impl(const VT& value) noexcept {
        init();
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix_impl(std::initializer_list<value_type> l) {
        init();

        cpp_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    fast_matrix_impl(storage_impl data) : _data(data) {
        //Nothing else to init
    }

    fast_matrix_impl(const fast_matrix_impl& rhs) noexcept {
        init();
        standard_evaluator::direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    fast_matrix_impl(fast_matrix_impl&& rhs) noexcept : _data(std::move(rhs._data)) {
        //Nothing else to init
    }

    template <typename T2, typename ST2, order SO2, std::size_t... Dims2, cpp_enable_if(SO == SO2)>
    fast_matrix_impl(const fast_matrix_impl<T2, ST2, SO2, Dims2...>& rhs) noexcept {
        init();
        validate_assign(*this, rhs);
        standard_evaluator::direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    template <typename T2, typename ST2, order SO2, std::size_t... Dims2, cpp_disable_if(SO == SO2)>
    fast_matrix_impl(const fast_matrix_impl<T2, ST2, SO2, Dims2...>& rhs) noexcept {
        init();
        validate_assign(*this, rhs);
        assign_evaluate(rhs, *this);
    }

    template <typename E, cpp_enable_if(!is_fast_matrix<E>::value, std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    explicit fast_matrix_impl(E&& e) {
        init();
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
    }

    template <typename Container, cpp_enable_if(
                                      std::is_convertible<typename Container::value_type, value_type>::value,
                                      cpp::not_c<is_etl_expr<Container>>::value)>
    explicit fast_matrix_impl(const Container& vec) {
        init();
        validate_assign(*this, vec);
        std::copy(vec.begin(), vec.end(), begin());
    }

    // Assignment

    // Copy assignment operator

    fast_matrix_impl& operator=(const fast_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            standard_evaluator::direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
        }
        return *this;
    }

    template <std::size_t... SDims>
    fast_matrix_impl& operator=(const fast_matrix_impl<T, ST, SO, SDims...>& rhs) noexcept {
        validate_assign(*this, rhs);
        assign_evaluate(rhs, *this);
        return *this;
    }

    //Allow copy from other containers

    template <typename Container, cpp_enable_if(!std::is_same<Container, value_type>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    fast_matrix_impl& operator=(const Container& vec) noexcept {
        validate_assign(*this, vec);
        std::copy(vec.begin(), vec.end(), begin());
        return *this;
    }

    //Construct from expression

    template <typename E, cpp_enable_if(std::is_convertible<typename E::value_type, value_type>::value, is_etl_expr<E>::value)>
    fast_matrix_impl& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    //Set the same value to each element of the matrix
    template <typename VT, cpp_enable_if_or(std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value)>
    fast_matrix_impl& operator=(const VT& value) noexcept {
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    fast_matrix_impl& operator=(fast_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            _data = std::move(rhs._data);
        }

        return *this;
    }

    // Swap operations

    /*!
     * \brief Swap the contents of the matrix with another matrix
     * \param other The other matrix
     */
    void swap(fast_matrix_impl& other) {
        using std::swap;
        swap(_data, other._data);
    }

    // Accessors

    /*!
     * \brief Returns the size of the matrix, in O(1)
     * \return The size of the matrix
     */
    static constexpr std::size_t size() noexcept {
        return etl_size;
    }

    /*!
     * \brief Returns the number of rows of the matrix (the first dimension), in O(1)
     * \return The number of rows of the matrix
     */
    static constexpr std::size_t rows() noexcept {
        return dim<0>();
    }

    /*!
     * \brief Returns the number of columns of the matrix (the second dimension), in O(1)
     * \return The number of columns of the matrix
     */
    static constexpr std::size_t columns() noexcept {
        static_assert(n_dimensions > 1, "columns() can only be used on 2D+ matrices");

        return dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
    }

    /*!
     * \brief Returns the Dth dimension of the matrix
     * \return The Dth dimension of the matrix
     */
    template <std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return nth_size<D, 0, Dims...>::value;
    }

    /*!
     * \brief Returns the dth dimension of the matrix
     * \param d The dimension to get
     * \return The Dth dimension of the matrix
     */
    std::size_t dim(std::size_t d) const noexcept {
        return dyn_nth_size<Dims...>(d);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) noexcept {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) const noexcept {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    CPP_DEBUG_AUTO_TRICK auto slice(std::size_t first, std::size_t last) noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    CPP_DEBUG_AUTO_TRICK auto slice(std::size_t first, std::size_t last) const noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Returns the value of the element at the position (args...)
     * \param args The position indices
     * \return The value of the element at (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), value_type&> operator()(S... args) noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Returns the value of the element at the position (args...)
     * \param args The position indices
     * \return The value of the element at (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), const value_type&> operator()(S... args) const noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](std::size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](std::size_t i) noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept(assert_nothrow) {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_enable_if(all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_disable_if(all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return rhs.alias(*this);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const noexcept {
        return V::loadu(memory_start() + i);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an iterator pointing to the first element of the matrix
     */
    iterator begin() noexcept(noexcept(_data.begin())) {
        return _data.begin();
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    iterator end() noexcept(noexcept(_data.end())) {
        return _data.end();
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a const iterator pointing to the first element of the matrix
     */
    const_iterator begin() const noexcept(noexcept(_data.cbegin())) {
        return _data.cbegin();
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    const_iterator end() const noexcept(noexcept(_data.end())) {
        return _data.cend();
    }

    /*!
     * \brief Return a const iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    const_iterator cbegin() const noexcept(noexcept(_data.cbegin())) {
        return _data.cbegin();
    }

    /*!
     * \brief Return a const iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    const_iterator cend() const noexcept(noexcept(_data.end())) {
        return _data.cend();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return &_data[0];
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return &_data[0];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return &_data[size()];
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return &_data[size()];
    }
};

static_assert(std::is_nothrow_default_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable<fast_vector<double, 2>>::value, "fast_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable<fast_vector<double, 2>>::value, "fast_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow destructible");

template <std::size_t... Dims, typename T>
fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Dims...> fast_matrix_over(T* memory){
    return fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Dims...>(cpp::array_wrapper<T>(memory, mul_all<Dims...>::value));
}

/*!
 * \brief Swaps the given two matrices
 * \param lhs The first matrix to swap
 * \param rhs The second matrix to swap
 */
template <typename T, typename ST, order SO, std::size_t... Dims>
void swap(fast_matrix_impl<T, ST, SO, Dims...>& lhs, fast_matrix_impl<T, ST, SO, Dims...>& rhs) {
    lhs.swap(rhs);
}

/*!
 * \brief Prints a fast matrix type (not the contents) to the given stream
 * \param os The output stream
 * \param matrix The fast matrix to print
 * \return the output stream
 */
template <typename T, typename ST, order SO, std::size_t... Dims>
std::ostream& operator<<(std::ostream& os, const fast_matrix_impl<T, ST, SO, Dims...>& matrix) {
    cpp_unused(matrix);

    if (sizeof...(Dims) == 1) {
        return os << "V[" << concat_sizes(Dims...) << "]";
    }

    return os << "M[" << concat_sizes(Dims...) << "]";
}

} //end of namespace etl
