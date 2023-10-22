//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/dyn_base.hpp" //The base class and utilities

namespace etl {

namespace sparse_detail {

/*!
 * \brief A proxy representing a reference to an element of a sparse matrix
 * \tparam M The matrix type
 */
template <typename M>
struct sparse_reference {
    using matrix_type              = M;                                ///< The matrix type
    using value_type               = typename matrix_type::value_type; ///< The value type
    using raw_pointer_type         = value_type*;                      ///< A raw pointer type
    using raw_reference_type       = value_type&;                      ///< A raw reference type
    using const_raw_reference_type = std::add_const_t<value_type>&;    ///< A raw const reference type

    matrix_type& matrix;  ///< Reference to the matrix
    size_t i;             ///< The first index
    size_t j;             ///< The second index
    size_t n;             ///< hint
    raw_pointer_type ptr; ///< Pointer to the element

    /*!
     * \brief Constructs a new sparse_reference
     * \param matrix The source matrix
     * \param i The index i of the first dimension
     * \param j The index j of the second dimension
     */
    sparse_reference(matrix_type& matrix, size_t i, size_t j) : matrix(matrix), i(i), j(j) {
        n = matrix.find_n(i, j);
        matrix.unsafe_set_hint(i, j, n, matrix.get_hint(i, j, n));
        ptr = &matrix.unsafe_ref_hint(n);
    }

    /*!
     * \brief Destruct the proxy reference and updates the matrix to the correct value
     */
    ~sparse_reference() {
        //Update the value, possibly erasing it
        matrix.set_hint(i, j, n, *ptr);
    }

    sparse_reference(sparse_reference&) = delete;
    sparse_reference& operator=(sparse_reference&) = delete;

    sparse_reference(sparse_reference&&) = delete;
    sparse_reference& operator=(sparse_reference&&) = delete;

    /*!
     * \brief Sets a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator=(value_type rhs) {
        get() = rhs;
        return *this;
    }

    /*!
     * \brief Adds a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator+=(value_type rhs) {
        get() += rhs;
        return *this;
    }

    /*!
     * \brief Subtract a new value from the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator-=(value_type rhs) {
        get() -= rhs;
        return *this;
    }

    /*!
     * \brief Multiply by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator*=(value_type rhs) {
        get() *= rhs;
        return *this;
    }

    /*!
     * \brief Divide by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator/=(value_type rhs) {
        get() /= rhs;
        return *this;
    }

    /*!
     * \brief Modulo by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    sparse_reference& operator%=(value_type rhs) {
        get() %= rhs;
        return *this;
    }

    /*!
     * \brief Casts the proxy reference to the raw reference type
     * \return a raw reference to the element
     */
    operator raw_reference_type() {
        return get();
    }

    /*!
     * \brief Casts the proxy reference to the raw reference type
     * \return a raw reference to the element
     */
    operator const_raw_reference_type() const {
        return get();
    }

private:
    raw_reference_type get() {
        return *ptr;
    }

    const_raw_reference_type get() const {
        return *ptr;
    }
};

/*!
 * \brief Utility function to test if a number is zero, overloaded for many
 * types.
 * \param a The number to test
 * \return true if the number is zero, false otherwise.
 */
inline bool is_zero(double a) {
    return a == 0.0;
}

/*!
 * \copydoc is_zero
 */
inline bool is_zero(float a) {
    return a == 0.0f;
}

/*!
 * \copydoc is_zero
 */
inline bool is_zero(std::complex<float> a) {
    return a.real() == 0.0f && a.imag() == 0.0f;
}

/*!
 * \copydoc is_zero
 */
inline bool is_zero(std::complex<double> a) {
    return a.real() == 0.0 && a.imag() == 0.0;
}

/*!
 * \copydoc is_zero
 */
inline bool is_zero(etl::complex<float> a) {
    return a.real == 0.0f && a.imag == 0.0f;
}

/*!
 * \copydoc is_zero
 */
inline bool is_zero(etl::complex<double> a) {
    return a.real == 0.0 && a.imag == 0.0;
}

/*!
 * \brief Utility function to test if a number is not a zero, overloaded for many
 * types.
 * \param value The number to test
 * \return true if the number is not a zero, false otherwise.
 */
template <typename T>
bool is_non_zero(T value) {
    return !is_zero(value);
}

} //end of namespace sparse_detail

/*!
 * \brief Sparse matrix implementation
 * \tparam T The type of value
 * \tparam SS The storage type
 * \tparam D The number of dimensions
 */
template <typename T, sparse_storage SS, size_t D>
struct sparse_matrix_impl;

/*!
 * \brief Sparse matrix implementation with COO storage type
 * \tparam T The type of value
 * \tparam D The number of dimensions
 */
template <typename T, size_t D>
struct sparse_matrix_impl<T, sparse_storage::COO, D> final : dyn_base<sparse_matrix_impl<T, sparse_storage::COO, D>, T, D> {
    static constexpr size_t n_dimensions           = D;                                      ///< The number of dimensions
    static constexpr sparse_storage storage_format = sparse_storage::COO;                    ///< The sparse storage scheme
    static constexpr order storage_order           = order::RowMajor;                        ///< The storage order
    static constexpr size_t alignment              = default_intrinsic_traits<T>::alignment; ///< The alignment

    using this_type              = sparse_matrix_impl<T, sparse_storage::COO, D>;    ///< this type
    using base_type              = dyn_base<this_type, T, D>;                        ///< The base type
    using reference_type         = sparse_detail::sparse_reference<this_type>;       ///< The type of reference returned by the functions
    using const_reference_type   = sparse_detail::sparse_reference<const this_type>; ///< The type of const reference returned by the functions
    using value_type             = T;                                                ///< The type of value returned by the function
    using dimension_storage_impl = std::array<size_t, n_dimensions>;                 ///< The type used to store the dimensions
    using memory_type            = value_type*;                                      ///< The memory type
    using const_memory_type      = const value_type*;                                ///< The const memory type
    using index_type             = size_t;                                           ///< The type used to store the COO index
    using index_memory_type      = index_type*;                                      ///< The memory type to the COO index

    friend struct sparse_detail::sparse_reference<this_type>;
    friend struct sparse_detail::sparse_reference<const this_type>;

    static_assert(n_dimensions == 2, "Only 2D sparse matrix are supported");

private:
    using base_type::_dimensions;
    using base_type::_size;
    memory_type _memory;          ///< The memory
    index_memory_type _row_index; ///< The row index
    index_memory_type _col_index; ///< The column index
    size_t nnz;                   ///< The number of nonzeros in the matrix

    using base_type::allocate;
    using base_type::check_invariants;
    using base_type::release;

    /*!
     * \brief Build the content of the sparse matrix from an
     * iterable collection
     */
    template <typename It>
    void build_from_iterable(const It& iterable) {
        nnz = 0;
        for (auto v : iterable) {
            if (sparse_detail::is_non_zero(v)) {
                ++nnz;
            }
        }

        if (nnz > 0) {
            //Allocate space for the three arrays
            _memory    = allocate(nnz);
            _row_index = base_type::template allocate<index_type>(nnz);
            _col_index = base_type::template allocate<index_type>(nnz);

            auto it  = iterable.begin();
            size_t n = 0;

            for (size_t i = 0; i < rows(); ++i) {
                for (size_t j = 0; j < columns(); ++j) {
                    if (sparse_detail::is_non_zero(*it)) {
                        _memory[n]    = *it;
                        _row_index[n] = i;
                        _col_index[n] = j;
                        ++n;
                    }

                    ++it;
                }
            }
        }
    }

    /*!
     * \brief Reserve enough space to put a value in position hint
     */
    void reserve_hint(size_t hint) {
        cpp_assert(hint < nnz + 1, "Invalid hint for reserve_hint");

        if (_memory) {
            auto new_memory    = allocate(nnz + 1);
            auto new_row_index = base_type::template allocate<index_type>(nnz + 1);
            auto new_col_index = base_type::template allocate<index_type>(nnz + 1);

            if (hint == nnz) {
                //Copy the elements
                std::copy_n(_memory, nnz, new_memory);
                std::copy_n(_row_index, nnz, new_row_index);
                std::copy_n(_col_index, nnz, new_col_index);
            } else {
                //Copy the elements before hint
                std::copy(_memory, _memory + hint, new_memory);
                std::copy(_row_index, _row_index + hint, new_row_index);
                std::copy(_col_index, _col_index + hint, new_col_index);

                //Copy the elements after hint
                std::copy(_memory + hint, _memory + nnz, new_memory + hint + 1);
                std::copy(_row_index + hint, _row_index + nnz, new_row_index + hint + 1);
                std::copy(_col_index + hint, _col_index + nnz, new_col_index + hint + 1);
            }

            release(_memory, nnz);
            release(_row_index, nnz);
            release(_col_index, nnz);

            _memory    = new_memory;
            _col_index = new_col_index;
            _row_index = new_row_index;
        } else {
            cpp_assert(hint == 0, "Invalid hint for reserve_hint");

            _memory    = allocate(nnz + 1);
            _row_index = base_type::template allocate<index_type>(nnz + 1);
            _col_index = base_type::template allocate<index_type>(nnz + 1);
        }

        ++nnz;
    }

    /*!
     * \brief Erase the value in position n
     */
    void erase_hint(size_t n) {
        cpp_assert(nnz > 0, "Invalid erase_hint call (no non-zero elements");

        if (nnz == 1) {
            release(_memory, nnz);
            release(_row_index, nnz);
            release(_col_index, nnz);

            _memory    = nullptr;
            _row_index = nullptr;
            _col_index = nullptr;
        } else {
            auto new_memory    = allocate(nnz - 1);
            auto new_row_index = base_type::template allocate<index_type>(nnz - 1);
            auto new_col_index = base_type::template allocate<index_type>(nnz - 1);

            if (n == nnz - 1) {
                std::copy_n(_memory, nnz - 1, new_memory);
                std::copy_n(_row_index, nnz - 1, new_row_index);
                std::copy_n(_col_index, nnz - 1, new_col_index);
            } else {
                std::copy(_memory, _memory + n, new_memory);
                std::copy(_row_index, _row_index + n, new_row_index);
                std::copy(_col_index, _col_index + n, new_col_index);

                std::copy(_memory + n + 1, _memory + nnz, new_memory + n);
                std::copy(_row_index + n + 1, _row_index + nnz, new_row_index + n);
                std::copy(_col_index + n + 1, _col_index + nnz, new_col_index + n);
            }

            release(_memory, nnz);
            release(_row_index, nnz);
            release(_col_index, nnz);

            _memory    = new_memory;
            _row_index = new_row_index;
            _col_index = new_col_index;
        }

        --nnz;
    }

    /*!
     * \brief Find the position of the value at (i,j). Returns nnz
     * if the position is not found. Can also return a position
     * already taken if its place of insertion is already taken.
     */
    size_t find_n(size_t i, size_t j) const noexcept {
        for (size_t n = 0; n < nnz; ++n) {
            //The value exists, modify it
            if (_row_index[n] == i && _col_index[n] == j) {
                return n;
            }

            //The insertion point has been found
            if ((_row_index[n] == i && _col_index[n] > j) || _row_index[n] > i) {
                return n;
            }
        }

        return nnz;
    }

    /*!
     * \brief Set the value at index (i,j) and position n
     * \param value The new value to set
     */
    void unsafe_set_hint(size_t i, size_t j, size_t n, value_type value) {
        //The value exists, modify it
        if (n < nnz && _row_index[n] == i && _col_index[n] == j) {
            _memory[n] = value;
            return;
        }

        reserve_hint(n);

        _memory[n]    = value;
        _row_index[n] = i;
        _col_index[n] = j;
    }

    /*!
     * \brief Get the value at index (i,j) and position n
     */
    template <bool B = n_dimensions == 2, cpp_enable_iff(B)>
    value_type get_hint(size_t i, size_t j, size_t n) const noexcept {
        if (n < nnz && _row_index[n] == i && _col_index[n] == j) {
            return _memory[n];
        }

        return 0.0;
    }

    /*!
     * \brief Set the value at index (i,j) and position n.
     */
    void set_hint(size_t i, size_t j, size_t n, value_type value) {
        if (n < nnz) {
            if (_row_index[n] == i && _col_index[n] == j) {
                //At this point, there is already a value for (i,j)
                //If zero, we remove it, otherwise edit it
                if (sparse_detail::is_non_zero(value)) {
                    unsafe_set_hint(i, j, n, value);
                } else {
                    erase_hint(n);
                }
            } else {
                //At this point, the value does not exist
                //We insert it if not zero
                if (sparse_detail::is_non_zero(value)) {
                    unsafe_set_hint(i, j, n, value);
                }
            }
        } else {
            //At this point, the value does not exist
            //We insert it if not zero
            if (sparse_detail::is_non_zero(value)) {
                unsafe_set_hint(i, j, n, value);
            }
        }
    }

    /*!
     * \brief Get a direct reference to the element at position n
     */
    value_type& unsafe_ref_hint(size_t n) {
        return _memory[n];
    }

    /*!
     * \brief Get a direct const reference to the element at position n
     */
    const value_type& unsafe_ref_hint(size_t n) const {
        return _memory[n];
    }

    /*!
     * \brief Inherit the dimensions of an ETL expressions.
     * This must only be called when the matrix has no dimensions
     * \param e The expression to get the dimensions from.
     */
    template <not_generator E>
    void inherit(const E& e) {
        cpp_assert(n_dimensions == etl::dimensions(e), "Invalid number of dimensions");

        // Compute the size and new dimensions
        _size = 1;
        for (size_t d = 0; d < n_dimensions; ++d) {
            _dimensions[d] = etl::dim(e, d);
            _size *= _dimensions[d];
        }
    }

public:
    using base_type::columns;
    using base_type::dim;
    using base_type::rows;
    using base_type::size;

    // Construction

    /*!
     * \brief Constructs a new empty sparse matrix
     */
    sparse_matrix_impl() noexcept : base_type(), _memory(nullptr), _row_index(nullptr), _col_index(nullptr), nnz(0) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new sparse matrix of the given dimensions,
     * filled with zeroes
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == D && cpp::all_convertible_to_v<size_t, S...>)>
    explicit sparse_matrix_impl(S... sizes) noexcept
            : base_type(util::size(sizes...), {{static_cast<size_t>(sizes)...}}), _memory(nullptr), _row_index(nullptr), _col_index(nullptr), nnz(0) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new sparse matrix of the given dimensions
     * and use the initializer list to fill the matrix
     */
    template <typename... S, cpp_enable_iff(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit sparse_matrix_impl(S... sizes) noexcept
            : base_type(util::size(std::make_index_sequence<(sizeof...(S) - 1)>(), sizes...),
                        dyn_detail::sizes(std::make_index_sequence<(sizeof...(S) - 1)>(), sizes...)) {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        build_from_iterable(list);
    }

    /*!
     * \brief Construct a new sparse matrix of the given dimensions
     * and use the list of values list to fill the matrix
     */
    template <typename S1, typename... S, cpp_enable_iff((sizeof...(S) == D) && cpp::is_specialization_of_v<values_t, typename cpp::last_type<S1, S...>::type>)>
    explicit sparse_matrix_impl(S1 s1, S... sizes) noexcept
            : base_type(util::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                        dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        auto list = cpp::last_value(sizes...).template list<value_type>();
        build_from_iterable(list);
    }

    /*!
     * \brief Copy assign from another matrix
     *
     * This operator can change the dimensions of the matrix
     *
     * \param rhs The matrix to copy from
     * \return A reference to the matrix
     */
    sparse_matrix_impl& operator=(const sparse_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            if (!_size) {
                inherit(rhs);
            } else {
                validate_assign(*this, rhs);
            }

            // Note: The const_cast is necessary in order to call assign_to
            // and the parameter cannot be made non-const
            const_cast<sparse_matrix_impl&>(rhs).assign_to(*this);
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Assign an ETL expression to the sparse matrix
     */
    template <typename E,
              cpp_enable_iff(!std::is_same_v<std::decay_t<E>, sparse_matrix_impl<T, storage_format, D>>
                             && std::is_convertible_v<value_t<E>, value_type> && is_etl_expr<E>)>
    sparse_matrix_impl& operator=(E&& e) noexcept {
        // It is possible that the matrix was not initialized before
        // In the case, get the the dimensions from the expression and
        // initialize the matrix
        if (!_size) {
            inherit(e);
        } else {
            validate_assign(*this, e);
        }

        // Avoid aliasing issues
        if constexpr (!decay_traits<E>::is_linear) {
            if (e.alias(*this)) {
                // Create a temporary to hold the result
                this_type tmp(*this);

                // Assign the expression to the temporary
                tmp = e;

                // Assign the temporary to this matrix
                *this = tmp;
            } else {
                e.assign_to(*this);
            }
        } else {
            // Direct assignment of the expression into this matrix
            e.assign_to(*this);
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Returns the value at the given (i,j) position in the matrix.
     *
     * This function will never insert a new element in the matrix. It is
     * suited when only reading the matrix and not neeeding references.
     *
     * \param i The row
     * \param j The column
     *
     * \return The value at the (i,j) position.
     */
    value_type get(size_t i, size_t j) const noexcept(assert_nothrow) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        auto n = find_n(i, j);
        return get_hint(i, j, n);
    }

    /*!
     * \brief Returns a reference to the element at the position (i,j)
     * \param i The first index
     * \param j The second index
     * \return a sparse reference (proxy reference) to the element at position (i,j)
     */
    reference_type operator()(size_t i, size_t j) noexcept(assert_nothrow) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        return {*this, i, j};
    }

    /*!
     * \brief Returns a reference to the element at the position (i,j)
     * \param i The first index
     * \param j The second index
     * \return a sparse reference (proxy reference) to the element at position (i,j)
     */
    const_reference_type operator()(size_t i, size_t j) const noexcept(assert_nothrow) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        return {*this, i, j};
    }

    /*!
     * \brief Returns the element at the given index
     * This function may result in insertion of deletion of elements
     * in the matrix and therefore invalidation of some references.
     * \param n The index
     * \return a reference to the element at the given index.
     */
    reference_type operator[](size_t n) noexcept(assert_nothrow) {
        cpp_assert(n < size(), "Out of bounds");

        return {*this, n / columns(), n % columns()};
    }

    /*!
     * \brief Returns the element at the given index
     * This function may result in insertion of deletion of elements
     * in the matrix and therefore invalidation of some references.
     * \param n The index
     * \return a reference to the element at the given index.
     */
    const_reference_type operator[](size_t n) const noexcept(assert_nothrow) {
        cpp_assert(n < size(), "Out of bounds");

        return {*this, n / columns(), n % columns()};
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param n The index
     * \return the value at the given index.
     */
    template <bool B = n_dimensions == 2, cpp_enable_iff(B)>
    value_type read_flat(size_t n) const noexcept {
        return get(n / columns(), n % columns());
    }

    /*!
     * \brief Returns Returns the number of non zeros entries in the sparse matrix.
     *
     * This is a constant time O(1) operation.
     *
     * \return The number of non zeros entries in the sparse matrix.
     */
    size_t non_zeros() const noexcept {
        return nnz;
    }

    /*!
     * \brief Sets the element at the given position (i, j) to the given value
     * \param i The first index
     * \param j The second index
     * \param value The new value
     */
    void set(size_t i, size_t j, value_type value) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        auto n = find_n(i, j);
        set_hint(i, j, n, value);
    }

    /*!
     * \brief Sets the element at the given position (i, j) to the given value
     *
     * This function will always set the element to the given value, even if it
     * is zero (the normal behaviour would have been to erase it). This must be
     * used when we need a pointer to the element in memory.
     *
     * \param i The first index
     * \param j The second index
     * \param value The new value
     */
    void unsafe_set(size_t i, size_t j, value_type value) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        auto n = find_n(i, j);

        unsafe_set_hint(i, j, n, value);
    }

    /*!
     * \brief Erases (sets to zero) the element at the given position (i, j)
     * \param i The first index
     * \param j The second index
     */
    void erase(size_t i, size_t j) {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        auto n = find_n(i, j);

        if (n < nnz && _row_index[n] == i && _col_index[n] == j) {
            erase_hint(n);
        }
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        if constexpr (is_sparse_matrix<E>) {
            return this == &rhs;
        } else {
            return rhs.alias(*this);
        }
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename V>
    void visit([[maybe_unused]] V&& visitor) const {}

    /*!
     * \brief Destructs the matrix and releases all its memory
     */
    ~sparse_matrix_impl() noexcept {
        if (_memory) {
            release(_memory, nnz);
            release(_row_index, nnz);
            release(_col_index, nnz);
        }
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // No GPU support for sparse matrix so far
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // No GPU support for sparse matrix so far
    }

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief sub to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief mul to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Div to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Mod to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Prints a fast matrix type (not the contents) to the given stream
     * \param os The output stream
     * \param matrix The fast matrix to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const sparse_matrix_impl& matrix) {
        os << "SM[" << matrix.dim(0);

        for (size_t i = 1; i < D; ++i) {
            os << "," << matrix.dim(i);
        }

        return os << "]";
    }
};

} //end of namespace etl
