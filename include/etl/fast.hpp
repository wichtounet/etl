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

#include "etl/fast_base.hpp"

namespace etl {

/*!
 * \brief Matrix with compile-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, typename ST, order SO, std::size_t... Dims>
struct fast_matrix_impl final :
        fast_matrix_base<fast_matrix_impl<T, ST, SO, Dims...>, T, ST, SO, Dims...>,
        inplace_assignable<fast_matrix_impl<T, ST, SO, Dims...>>,
        comparable<fast_matrix_impl<T, ST, SO, Dims...>>,
        expression_able<fast_matrix_impl<T, ST, SO, Dims...>>,
        value_testable<fast_matrix_impl<T, ST, SO, Dims...>>,
        iterable<fast_matrix_impl<T, ST, SO, Dims...>, SO == order::RowMajor>,
        dim_testable<fast_matrix_impl<T, ST, SO, Dims...>> {
    static_assert(sizeof...(Dims) > 0, "At least one dimension must be specified");

public:
    static constexpr std::size_t n_dimensions = sizeof...(Dims);                      ///< The number of dimensions
    static constexpr std::size_t etl_size     = mul_all<Dims...>::value;              ///< The size of the matrix
    static constexpr order storage_order      = SO;                                   ///< The storage order
    static constexpr bool array_impl          = !matrix_detail::is_vector<ST>::value; ///< true if the storage is an std::arraw, false otherwise

    using this_type          = fast_matrix_impl<T, ST, SO, Dims...>; ///< this type
    using base_type          = fast_matrix_base<this_type, T, ST, SO, Dims...>;
    using iterable_base_type = iterable<this_type, SO == order::RowMajor>;
    using value_type         = T;                 ///< The value type
    using storage_impl       = ST;                ///< The storage implementation
    using memory_type        = value_type*;       ///< The memory type
    using const_memory_type  = const value_type*; ///< The const memory type

    using base_type::dim;
    using base_type::size;
    using iterable_base_type::begin;
    using iterable_base_type::end;
    using base_type::memory_start;
    using base_type::memory_end;

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

private:
    using base_type::_data;

    mutable gpu_handler<T> _gpu_memory_handler; ///< The GPU memory handler

    /*!
     * \brief Init the container if necessary
     */
    template <typename S = ST, cpp_enable_if(matrix_detail::is_vector<S>::value)>
    void init() {
        _data.resize(alloc_size_mat<value_type>(size(), this_type::template dim<n_dimensions - 1>()));
    }

    /*!
     * \copydoc init
     */
    template <typename S = ST, cpp_disable_if(matrix_detail::is_vector<S>::value)>
    void init() noexcept {
        //Nothing else to init
    }

public:
    /// Construction

    /*!
     * \brief Construct an empty fast matrix
     */
    fast_matrix_impl() noexcept: base_type()  {
        init();
    }

    /*!
     * \brief Construct a fast matrix filled with the same value
     * \param value the value to fill the matrix with
     */
    template <typename VT, cpp_enable_if_or(std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value)>
    explicit fast_matrix_impl(const VT& value) noexcept: base_type()  {
        init();
        std::fill(begin(), end(), value);
    }

    /*!
     * \brief Construct a fast matrix filled with the given values
     * \param l the list of values to fill the matrix with
     */
    fast_matrix_impl(std::initializer_list<value_type> l): base_type()  {
        init();

        cpp_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    /*!
     * \brief Construct a fast matrix directly from storage
     * \param data The storage container to copy
     */
    fast_matrix_impl(const storage_impl& data) : base_type(data) {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a fast matrix
     * \param rhs The fast matrix to copy from
     */
    fast_matrix_impl(const fast_matrix_impl& rhs) noexcept : base_type() {
        init();
        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Move construct a fast matrix
     * \param rhs The fast matrix to move from
     */
    fast_matrix_impl(fast_matrix_impl&& rhs) noexcept : base_type(), _gpu_memory_handler(std::move(rhs._gpu_memory_handler)) {
        _data = std::move(rhs._data);
    }

    /*!
     * \brief Copy construct a fast matrix from a different matrix fast matrix type
     * \param rhs The fast matrix to copy from
     */
    template <typename T2, typename ST2, order SO2, std::size_t... Dims2, cpp_enable_if(SO == SO2)>
    fast_matrix_impl(const fast_matrix_impl<T2, ST2, SO2, Dims2...>& rhs) noexcept: base_type()  {
        init();
        validate_assign(*this, rhs);
        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Copy construct a fast matrix from a different matrix fast matrix type
     * \param rhs The fast matrix to copy from
     */
    template <typename T2, typename ST2, order SO2, std::size_t... Dims2, cpp_disable_if(SO == SO2)>
    fast_matrix_impl(const fast_matrix_impl<T2, ST2, SO2, Dims2...>& rhs) noexcept: base_type()  {
        init();
        validate_assign(*this, rhs);
        assign_evaluate(rhs, *this);
    }

    /*!
     * \brief Construct a fast matrix from the given ETL expression
     * \param e The ETL expression
     */
    template <typename E, cpp_enable_if(!is_fast_matrix<E>::value, std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    explicit fast_matrix_impl(E&& e) : base_type() {
        init();
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
    }

    /*!
     * \brief Construct a fast matrix from the given STL container
     * \param container The container to get values from
     */
    template <typename C, cpp_enable_if(std::is_convertible<value_t<C>, value_type>::value, !is_etl_expr<C>::value)>
    explicit fast_matrix_impl(const C& container): base_type()  {
        init();
        validate_assign(*this, container);
        std::copy(container.begin(), container.end(), begin());
    }

    // Assignment

    /*!
     * \brief Copy assign a fast matrix
     * \param rhs The fast matrix to copy from
     * \return a reference to the fast matrix
     */
    fast_matrix_impl& operator=(const fast_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
        }

        return *this;
    }

    /*!
     * \brief Copy assign a fast matrix from a different matrix fast matrix type
     * \param rhs The fast matrix to copy from
     * \return a reference to the fast matrix
     */
    template <std::size_t... SDims>
    fast_matrix_impl& operator=(const fast_matrix_impl<T, ST, SO, SDims...>& rhs) noexcept {
        validate_assign(*this, rhs);
        assign_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Assign the values of the STL container to the fast matrix
     * \param container The STL container to get the values from
     * \return a reference to the fast matrix
     */
    template <typename C, cpp_enable_if(!std::is_same<C, value_type>::value, std::is_convertible<value_t<C>, value_type>::value)>
    fast_matrix_impl& operator=(const C& container) noexcept {
        validate_assign(*this, container);
        std::copy(container.begin(), container.end(), begin());
        return *this;
    }

    /*!
     * \brief Assign the values of the ETL expression to the fast matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <typename E, cpp_enable_if(std::is_convertible<typename E::value_type, value_type>::value, is_etl_expr<E>::value)>
    fast_matrix_impl& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    /*!
     * \brief Assign the value to each element
     * \param value The value to assign to each element
     * \return a reference to the fast matrix
     */
    template <typename VT, cpp_enable_if_or(std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value)>
    fast_matrix_impl& operator=(const VT& value) noexcept {
        std::fill(begin(), end(), value);

        return *this;
    }

    /*!
     * \brief Move assign a fast matrix
     * \param rhs The fast matrix to move from
     * \return a reference to the fast matrix
     */
    fast_matrix_impl& operator=(fast_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            _data               = std::move(rhs._data);
            _gpu_memory_handler = std::move(rhs._gpu_memory_handler);
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

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        V::store(memory_start() + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        V::storeu(memory_start() + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        V::stream(memory_start() + i, in);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(std::size_t i) const noexcept {
        return V::load(memory_start() + i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(std::size_t i) const noexcept {
        return V::loadu(memory_start() + i);
    }

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<T, n_dimensions> direct() const {
        return opaque_memory<T, n_dimensions>(memory_start(), etl_size, {{Dims...}}, _gpu_memory_handler, SO);
    }
};

static_assert(std::is_nothrow_default_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable<fast_vector<double, 2>>::value, "fast_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable<fast_vector<double, 2>>::value, "fast_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<fast_vector<double, 2>>::value, "fast_vector should be nothrow destructible");

/*!
 * \brief Create a fast_matrix of the given dimensions over the given memory
 * \param memory The memory
 * \return A fast_matrix using the given memory
 *
 * The memory must be large enough to hold the matrix
 */
template <std::size_t... Dims, typename T>
fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Dims...> fast_matrix_over(T* memory) {
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

/*!
 * \brief Serialize the given matrix using the given serializer
 * \param os The serializer
 * \param matrix The matrix to serialize
 */
template <typename Stream, typename T, typename ST, order SO, std::size_t... Dims>
void serialize(serializer<Stream>& os, const fast_matrix_impl<T, ST, SO, Dims...>& matrix) {
    for (const auto& value : matrix) {
        os << value;
    }
}

/*!
 * \brief Deserialize the given matrix using the given serializer
 * \param os The deserializer
 * \param matrix The matrix to deserialize
 */
template <typename Stream, typename T, typename ST, order SO, std::size_t... Dims>
void deserialize(deserializer<Stream>& os, fast_matrix_impl<T, ST, SO, Dims...>& matrix) {
    for (auto& value : matrix) {
        os >> value;
    }
}

} //end of namespace etl
