//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime matrix implementation
 */

#pragma once

#include "etl/dyn_base.hpp"    //The base class and utilities

namespace etl {

/*!
 * \brief Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, order SO, std::size_t D>
struct dyn_matrix_impl final : dyn_base<T, D>, inplace_assignable<dyn_matrix_impl<T, SO, D>>, comparable<dyn_matrix_impl<T, SO, D>>, expression_able<dyn_matrix_impl<T, SO, D>>, value_testable<dyn_matrix_impl<T, SO, D>>, dim_testable<dyn_matrix_impl<T, SO, D>> {
    static constexpr const std::size_t n_dimensions = D;                              ///< The number of dimensions
    static constexpr const order storage_order      = SO;                             ///< The storage order
    static constexpr const std::size_t alignment    = intrinsic_traits<T>::alignment; ///< The memory alignment

    using base_type              = dyn_base<T, D>;                        ///< The base type
    using value_type             = T;                                     ///< The value type
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>; ///< The type used to store the dimensions
    using memory_type            = value_type*;                           ///< The memory type
    using const_memory_type      = const value_type*;                     ///< The const memory type
    using iterator               = memory_type;                           ///< The type of iterator
    using const_iterator         = const_memory_type;                     ///< The type of const iterator

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<T>;

private:
    using base_type::_size;
    using base_type::_dimensions;
    bool managed = true; ///< Tag indicating if we manage the memory
    memory_type _memory; ///< Pointer to the allocated memory

    mutable gpu_handler<T> _gpu_memory_handler;

    using base_type::release;
    using base_type::allocate;
    using base_type::check_invariants;

public:
    using base_type::dim;

    // Construction

    /*!
     * \brief Construct an empty matrix
     *
     * This matrix don't have any memory nor dimensionsand most
     * operations will likely fail on it
     */
    dyn_matrix_impl() noexcept : base_type(), _memory(nullptr) {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    dyn_matrix_impl(const dyn_matrix_impl& rhs) noexcept : base_type(rhs), _memory(allocate(_size)) {
        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Move construct a matrix
     * \param rhs The matrix to move
     */
    dyn_matrix_impl(dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)), managed(rhs.managed), _memory(rhs._memory) {
        rhs._memory = nullptr;
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    template <typename T2, order SO2, std::size_t D2, cpp_enable_if(SO2 == SO)>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO2, D2>& rhs) noexcept : base_type(rhs), _memory(allocate(_size)) {
        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    template <typename T2, order SO2, std::size_t D2, cpp_disable_if(SO2 == SO)>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO2, D2>& rhs) noexcept : base_type(rhs), _memory(allocate(_size)) {
        //The type is different, so we must use assign
        assign_evaluate(rhs, *this);
    }

    /*!
     * \brief Construct a matrix from an expression
     * \param e The expression to initialize the matrix with
     */
    template <typename E, cpp_enable_if(
                              std::is_convertible<value_t<E>, value_type>::value,
                              is_etl_expr<E>::value,
                              !is_dyn_matrix<E>::value)>
    explicit dyn_matrix_impl(E&& e) noexcept
            : base_type(e), _memory(allocate(_size)) {
        assign_evaluate(e, *this);
    }

    /*!
     * \brief Construct a vector with the given values
     * \param list Initializer list containing all the values of the vector
     */
    dyn_matrix_impl(std::initializer_list<value_type> list) noexcept : base_type(list.size(), {{list.size()}}),
                                                                       _memory(allocate(_size)) {
        static_assert(n_dimensions == 1, "This constructor can only be used for 1D matrix");

        std::copy(list.begin(), list.end(), begin());
    }

    /*!
     * \brief Construct a matrix with the given dimensions
     * \param sizes The dimensions of the matrix
     *
     * The number of dimesnions must be the same as the D template
     * parameter of the matrix.
     */
    template <typename... S, cpp_enable_if(
                                 (sizeof...(S) == D),
                                 cpp::all_convertible_to<std::size_t, S...>::value,
                                 cpp::is_homogeneous<typename cpp::first_type<S...>::type, S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(sizes...), {{static_cast<std::size_t>(sizes)...}}),
                                                    _memory(allocate(_size)) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a matrix over existing memory
     * \param memory Pointer to the memory
     * \param sizes The dimensions of the matrix
     *
     * The number of dimesnions must be the same as the D template
     * parameter of the matrix.
     *
     * The memory won't be managed, meaning that it won't be
     * released once the matrix is destructed.
     */
    template <typename... S, cpp_enable_if(
                                 (sizeof...(S) == D),
                                 cpp::all_convertible_to<std::size_t, S...>::value,
                                 cpp::is_homogeneous<typename cpp::first_type<S...>::type, S...>::value)>
    explicit dyn_matrix_impl(value_type* memory, S... sizes) noexcept : base_type(dyn_detail::size(sizes...), {{static_cast<std::size_t>(sizes)...}}),
                                                    managed(false), _memory(memory) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a matrix with the given dimensions and values
     * \param sizes The dimensions of the matrix followed by an initializer_list
     */
    template <typename... S, cpp_enable_if(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
                                                    _memory(allocate(_size)) {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        std::copy(list.begin(), list.end(), begin());
    }

    /*!
     * \brief Construct a matrix with the given dimensions and values
     * \param sizes The dimensions of the matrix followed by a values_t
     */
    template <typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              cpp::is_specialization_of<values_t, typename cpp::last_type<std::size_t, S...>::type>::value)>
    explicit dyn_matrix_impl(std::size_t s1, S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                                     dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _memory(allocate(_size)) {
        auto list = cpp::last_value(sizes...).template list<value_type>();
        std::copy(list.begin(), list.end(), begin());
    }

    /*!
     * \brief Construct a matrix with the given dimensions and a value
     * \param sizes The dimensions of the matrix followed by a values
     *
     * Every element of the matrix will be set to this value.
     */
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              std::is_convertible<std::size_t, S1>::value, //The first type must be convertible to size_t
                                              cpp::is_sub_homogeneous<S1, S...>::value,                                          //The first N-1 types must homegeneous
                                              (std::is_arithmetic<typename cpp::last_type<S1, S...>::type>::value
                                                   ? std::is_convertible<value_type, typename cpp::last_type<S1, S...>::type>::value //The last type must be convertible to value_type
                                                   : std::is_same<value_type, typename cpp::last_type<S1, S...>::type>::value        //The last type must be exactly value_type
                                               ))>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : base_type(
                                                               dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                               dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)
                                                            ),
                                                           _memory(allocate(_size)) {
        intel_decltype_auto value = cpp::last_value(s1, sizes...);
        std::fill(begin(), end(), value);
    }

    /*!
     * \brief Construct a matrix with the given dimensions and a generator expression
     * \param sizes The dimensions of the matrix followed by a values
     *
     * The generator expression will be used to initialize the
     * elements of the matrix, in order.
     */
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              std::is_convertible<std::size_t, S1>::value,        //The first type must be convertible to size_t
                                              cpp::is_sub_homogeneous<S1, S...>::value,                                                 //The first N-1 types must homegeneous
                                              cpp::is_specialization_of<generator_expr, typename cpp::last_type<S1, S...>::type>::value //The last type must be a generator expr
                                              )>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                                     dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _memory(allocate(_size)) {
        intel_decltype_auto e = cpp::last_value(sizes...);

        assign_evaluate(e, *this);
    }

    /*!
     * \brief Construct a matrix with the given dimensions and a generator expression
     * \param sizes The dimensions of the matrix followed by an init_flag and a value
     *
     * Every element of the matrix will be set to this value.
     *
     * This constructor is necessary when the type of the matrix is
     * std::size_t
     */
    template <typename... S, cpp_enable_if(dyn_detail::is_init_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
                                                    _memory(allocate(_size)) {
        static_assert(sizeof...(S) == D + 2, "Invalid number of dimensions");

        std::fill(begin(), end(), cpp::last_value(sizes...));
    }

    /*!
     * \brief Construct a vector from a Container
     * \param container A STL container
     *
     * Only possible for 1D matrices
     */
    template <typename Container, cpp_enable_if(
                                      cpp::not_c<is_etl_expr<Container>>::value,
                                      std::is_convertible<typename Container::value_type, value_type>::value)>
    explicit dyn_matrix_impl(const Container& container)
            : base_type(container.size(), {{container.size()}}), _memory(allocate(container.size())) {
        static_assert(D == 1, "Only 1D matrix can be constructed from containers");

        // Copy the container directly inside the allocated memory
        std::copy_n(container.begin(), _size, _memory);
    }

    /*!
     * \brief Copy assign from another matrix
     *
     * This operator can change the dimensions of the matrix
     *
     * \param rhs The matrix to copy from
     * \return A reference to the matrix
     */
    dyn_matrix_impl& operator=(const dyn_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            if (!_size) {
                _size       = rhs._size;
                _dimensions = rhs._dimensions;
                _memory     = allocate(_size);
            } else {
                validate_assign(*this, rhs);
            }

            direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Move assign from another matrix
     *
     * The other matrix won't be usable after the move operation
     *
     * \param rhs The matrix to move from
     * \return A reference to the matrix
     */
    dyn_matrix_impl& operator=(dyn_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            if(_memory){
                release(_memory, _size);
            }

            _size       = rhs._size;
            _dimensions = std::move(rhs._dimensions);
            _memory = rhs._memory;

            rhs._size = 0;
            rhs._memory = nullptr;
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Resize with the new dimensions in the given array
     * \param dimensions The new dimensions
     */
    void resize_arr(const dimension_storage_impl& dimensions){
        auto new_size = std::accumulate(dimensions.begin(), dimensions.end(), std::size_t(1), std::multiplies<std::size_t>());

        if(_memory){
            auto new_memory = allocate(new_size);

            for (std::size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, _size);

            _memory = new_memory;
        } else {
            _memory     = allocate(new_size);
        }

        _size       = new_size;
        _dimensions = dimensions;
    }

    /*!
     * \brief Resize with the new given dimensions
     * \param sizes The new dimensions
     */
    template<typename... Sizes>
    void resize(Sizes... sizes){
        static_assert(sizeof...(Sizes), "Cannot change number of dimensions");

        auto new_size = dyn_detail::size(sizes...);

        if(_memory){
            auto new_memory = allocate(new_size);

            for (std::size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, _size);

            _memory = new_memory;
        } else {
            _memory     = allocate(new_size);
        }

        _size       = new_size;
        _dimensions = dyn_detail::sizes(std::make_index_sequence<D>(), sizes...);
    }

    /*!
     * \brief Assign from an ETL expression.
     * \param e The expression containing the values to assign to the matrix
     * \return A reference to the matrix
     */
    template <typename E, cpp_enable_if(!std::is_same<std::decay_t<E>, dyn_matrix_impl<T, SO, D>>::value, std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    dyn_matrix_impl& operator=(E&& e) noexcept {
        validate_assign(*this, e);

        assign_evaluate(e, *this);

        check_invariants();

        return *this;
    }

    /*!
     * \brief Assign from an STL container.
     * \param vec The container containing the values to assign to the matrix
     * \return A reference to the matrix
     */
    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    dyn_matrix_impl& operator=(const Container& vec) {
        validate_assign(*this, vec);

        std::copy(vec.begin(), vec.end(), begin());

        check_invariants();

        return *this;
    }

    /*!
     * \brief Assign the same value to each element of the matrix
     * \param value The value to assign to each element of the matrix
     * \return A reference to the matrix
     */
    dyn_matrix_impl& operator=(const value_type& value) noexcept {
        std::fill(begin(), end(), value);

        check_invariants();

        return *this;
    }

    /*!
     * \brief Destruct the matrix and release all its memory
     */
    ~dyn_matrix_impl() noexcept {
        if(managed && _memory){
            release(_memory, _size);
        }
    }

    /*!
     * \brief Swap the content of the matrix with the content of the given matrix
     * \param other The other matrix to swap content with
     */
    void swap(dyn_matrix_impl& other) {
        using std::swap;
        swap(_size, other._size);
        swap(_dimensions, other._dimensions);
        swap(_memory, other._memory);
        swap(managed, other.managed);

        //TODO swap is likely screwing up GPU memory!

        check_invariants();
    }

    // Accessors

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
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
    auto slice(std::size_t first, std::size_t last) noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(std::size_t first, std::size_t last) const noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Access the ith element of the matrix
     * \param i The index of the element to search
     * \return a reference to the ith element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 1, cpp_enable_if(B)>
    value_type& operator()(std::size_t i) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

    /*!
     * \brief Access the ith element of the matrix
     * \param i The index of the element to search
     * \return a reference to the ith element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 1, cpp_enable_if(B)>
    const value_type& operator()(std::size_t i) const noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 2, cpp_enable_if(B)>
    value_type& operator()(std::size_t i, std::size_t j) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        if (storage_order == order::RowMajor) {
            return _memory[i * dim(1) + j];
        } else {
            return _memory[j * dim(0) + i];
        }
    }

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    template <bool B = n_dimensions == 2, cpp_enable_if(B)>
    const value_type& operator()(std::size_t i, std::size_t j) const noexcept {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        if (storage_order == order::RowMajor) {
            return _memory[i * dim(1) + j];
        } else {
            return _memory[j * dim(0) + i];
        }
    }

    /*!
     * \brief Return the flat index for the element at the given position
     * \param sizes The indices
     * \return The flat index
     */
    template <typename... S>
    std::size_t index(S... sizes) const noexcept(assert_nothrow) {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        std::size_t index = 0;

        if (storage_order == order::RowMajor) {
            std::size_t subsize = _size;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    cpp_assert(s < dim(i), "Out of bounds");
                    subsize /= dim(i++);
                    index += subsize * s;
                },
                sizes...);
        } else {
            std::size_t subsize = 1;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    cpp_assert(s < dim(i), "Out of bounds");
                    index += subsize * s;
                    subsize *= dim(i++);
                },
                sizes...);
        }

        return index;
    }

    /*!
     * \brief Returns the value at the position (sizes...)
     * \param sizes The indices
     * \return The value at the position (sizes...)
     */
    template <typename... S, cpp_enable_if(
                                 (n_dimensions > 2),
                                 (sizeof...(S) == n_dimensions),
                                 cpp::all_convertible_to<std::size_t, S...>::value)>
    const value_type& operator()(S... sizes) const noexcept(assert_nothrow) {
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _memory[index(sizes...)];
    }

    /*!
     * \brief Returns the value at the position (sizes...)
     * \param sizes The indices
     * \return The value at the position (sizes...)
     */
    template <typename... S, cpp_enable_if(
                                 (n_dimensions > 2),
                                 (sizeof...(S) == n_dimensions),
                                 cpp::all_convertible_to<std::size_t, S...>::value)>
    value_type& operator()(S... sizes) noexcept(assert_nothrow) {
        static_assert(sizeof...(S) == n_dimensions, "Invalid number of parameters");

        return _memory[index(sizes...)];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](std::size_t i) const noexcept {
        cpp_assert(i < _size, "Out of bounds");

        return _memory[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](std::size_t i) noexcept {
        cpp_assert(i < _size, "Out of bounds");

        return _memory[i];
    }

    /*!
     * \returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        cpp_assert(i < _size, "Out of bounds");

        return _memory[i];
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const noexcept {
        return V::loadu(_memory + i);
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
     * \brief Return an iterator to the first element of the matrix
     * \return an iterator pointing to the first element of the matrix
     */
    iterator begin() noexcept {
        return _memory;
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    iterator end() noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an iterator pointing to the first element of the matrix
     */
    const_iterator begin() const noexcept {
        return _memory;
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    const_iterator end() const noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Return a const iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    const_iterator cbegin() const noexcept {
        return _memory;
    }

    /*!
     * \brief Return a const iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    const_iterator cend() const noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    inline memory_type memory_start() noexcept {
        return _memory;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    inline const_memory_type memory_start() const noexcept {
        return _memory;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer to the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return _memory + _size;
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    std::size_t& unsafe_dimension_access(std::size_t i) {
        cpp_assert(i < n_dimensions, "Out of bounds");
        return _dimensions[i];
    }

    opaque_memory<T, n_dimensions> direct() const {
        return opaque_memory<T, n_dimensions>(memory_start(), _size, _dimensions, _gpu_memory_handler, SO);
    }
};

static_assert(std::is_nothrow_default_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<dyn_vector<double>>::value, "dyn_vector should be nothrow destructible");

/*!
 * \brief Create a dyn_matrix of the given dimensions over the given memory
 * \param memory The memory
 * \param sizes The dimensions of the matrix
 * \return A dyn_matrix using the given memory
 *
 * The memory must be large enough to hold the matrix
 */
template<typename T, typename... Sizes>
dyn_matrix_impl<T, order::RowMajor, sizeof...(Sizes)> dyn_matrix_over(T* memory, Sizes... sizes){
    return dyn_matrix_impl<T, order::RowMajor, sizeof...(Sizes)>(memory, sizes...);
}

/*!
 * \brief Swap two dyn matrix
 * \param lhs The first matrix
 * \param rhs The second matrix
 */
template <typename T, order SO, std::size_t D>
void swap(dyn_matrix_impl<T, SO, D>& lhs, dyn_matrix_impl<T, SO, D>& rhs) {
    lhs.swap(rhs);
}

/*!
 * \brief Serialize the given matrix using the given serializer
 * \param os The serializer
 * \param matrix The matrix to serialize
 */
template <typename Stream, typename T, order SO, std::size_t D>
void serialize(serializer<Stream>& os, const dyn_matrix_impl<T, SO, D>& matrix){
    for(std::size_t i = 0; i < etl::dimensions(matrix); ++i){
        os << matrix.dim(i);
    }

    for(const auto& value : matrix){
        os << value;
    }
}

/*!
 * \brief Deserialize the given matrix using the given serializer
 * \param is The deserializer
 * \param matrix The matrix to deserialize
 */
template <typename Stream, typename T, order SO, std::size_t D>
void deserialize(deserializer<Stream>& is, dyn_matrix_impl<T, SO, D>& matrix){
    typename std::decay_t<decltype(matrix)>::dimension_storage_impl new_dimensions;

    for(auto& value : new_dimensions){
        is >> value;
    }

    matrix.resize_arr(new_dimensions);

    for(auto& value : matrix){
        is >> value;
    }
}

template <typename T, order SO, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_impl<T, SO, D>& mat) {
    if (D == 1) {
        return os << "V[" << mat.size() << "]";
    }

    os << "M[" << mat.dim(0);

    for (std::size_t i = 1; i < D; ++i) {
        os << "," << mat.dim(i);
    }

    return os << "]";
}

} //end of namespace etl
