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
struct dyn_matrix_impl final : dyn_base<T, D>, inplace_assignable<dyn_matrix_impl<T, SO, D>>, comparable<dyn_matrix_impl<T, SO, D>>, expression_able<dyn_matrix_impl<T, SO, D>>, value_testable<dyn_matrix_impl<T, SO, D>>, dim_testable<dyn_matrix_impl<T, SO, D>>, gpu_able<T, dyn_matrix_impl<T, SO, D>> {
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
    memory_type _memory; ///< Pointer to the allocated memory

    using base_type::release;
    using base_type::allocate;
    using base_type::check_invariants;

public:
    using base_type::dim;

    // Construction

    //Default constructor (constructs an empty matrix)
    dyn_matrix_impl() noexcept : base_type(), _memory(nullptr) {
        //Nothing else to init
    }

    //Copy constructor
    dyn_matrix_impl(const dyn_matrix_impl& rhs) noexcept : base_type(rhs), _memory(allocate(_size)) {
        std::copy_n(rhs._memory, _size, _memory);
    }

    //Copy constructor with different type
    //This constructor is necessary because the one from expression is explicit
    template <typename T2>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO, D>& rhs) noexcept : base_type(rhs), _memory(allocate(_size)) {
        //The type is different, so we must use assign
        assign_evaluate(rhs, *this);
    }

    //Move constructor
    dyn_matrix_impl(dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)), _memory(rhs._memory) {
        rhs._memory = nullptr;
    }

    //Initializer-list construction for vector
    dyn_matrix_impl(std::initializer_list<value_type> list) noexcept : base_type(list.size(), {{list.size()}}),
                                                                       _memory(allocate(_size)) {
        static_assert(n_dimensions == 1, "This constructor can only be used for 1D matrix");

        std::copy(list.begin(), list.end(), begin());
    }

    //Normal constructor with only sizes
    template <typename... S, cpp_enable_if(
                                 (sizeof...(S) == D),
                                 cpp::all_convertible_to<std::size_t, S...>::value,
                                 cpp::is_homogeneous<typename cpp::first_type<S...>::type, S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(sizes...), {{static_cast<std::size_t>(sizes)...}}),
                                                    _memory(allocate(_size)) {
        //Nothing to init
    }

    //Sizes followed by an initializer list
    template <typename... S, cpp_enable_if(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
                                                    _memory(allocate(_size)) {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        std::copy(list.begin(), list.end(), begin());
    }

    //Sizes followed by a values_t
    template <typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              cpp::is_specialization_of<values_t, typename cpp::last_type<std::size_t, S...>::type>::value)>
    explicit dyn_matrix_impl(std::size_t s1, S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                                     dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _memory(allocate(_size)) {
        auto list = cpp::last_value(sizes...).template list<value_type>();
        std::copy(list.begin(), list.end(), begin());
    }

    //Sizes followed by a value
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

    //Sizes followed by a generator_expr
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

    //Sizes followed by an init flag followed by the value
    template <typename... S, cpp_enable_if(dyn_detail::is_init_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
                                                    _memory(allocate(_size)) {
        static_assert(sizeof...(S) == D + 2, "Invalid number of dimensions");

        std::fill(begin(), end(), cpp::last_value(sizes...));
    }

    template <typename E, cpp_enable_if(
                              std::is_convertible<value_t<E>, value_type>::value,
                              is_etl_expr<E>::value)>
    explicit dyn_matrix_impl(E&& e) noexcept
            : base_type(e), _memory(allocate(_size)) {
        assign_evaluate(std::forward<E>(e), *this);
    }

    template <typename Container, cpp_enable_if(
                                      cpp::not_c<is_etl_expr<Container>>::value,
                                      std::is_convertible<typename Container::value_type, value_type>::value)>
    explicit dyn_matrix_impl(const Container& vec)
            : base_type(vec.size(), {_size}), _memory(allocate(_size)) {
        static_assert(D == 1, "Only 1D matrix can be constructed from containers");

        for (std::size_t i = 0; i < _size; ++i) {
            _memory[i] = vec[i];
        }
    }

    // Assignment

    //Copy assignment operator

    //Note: For now, this is the only constructor that is able to change the size and dimensions of the matrix
    dyn_matrix_impl& operator=(const dyn_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            if (!_size) {
                _size       = rhs._size;
                _dimensions = rhs._dimensions;
                _memory =   allocate(_size);
                std::copy_n(rhs._memory, _size, _memory);
            } else {
                validate_assign(*this, rhs);
                assign_evaluate(rhs, *this);
            }
        }

        check_invariants();

        return *this;
    }

    //Default move assignment operator
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

    //Construct from expression

    template <typename E, cpp_enable_if(!std::is_same<std::decay_t<E>, dyn_matrix_impl<T, SO, D>>::value, std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    dyn_matrix_impl& operator=(E&& e) noexcept {
        validate_assign(*this, e);

        assign_evaluate(e, *this);

        check_invariants();

        return *this;
    }

    //Allow copy from other containers

    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    dyn_matrix_impl& operator=(const Container& vec) {
        validate_assign(*this, vec);

        std::copy(vec.begin(), vec.end(), begin());

        check_invariants();

        return *this;
    }

    //Set the same value to each element of the matrix
    dyn_matrix_impl& operator=(const value_type& value) noexcept {
        std::fill(begin(), end(), value);

        check_invariants();

        return *this;
    }

    //Destructor

    /*!
     * \brief Destruct the matrix and release all its memory
     */
    ~dyn_matrix_impl() noexcept {
        if(_memory){
            release(_memory, _size);
        }
    }

    // Swap operations

    void swap(dyn_matrix_impl& other) {
        using std::swap;
        swap(_size, other._size);
        swap(_dimensions, other._dimensions);
        swap(_memory, other._memory);

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

    template <bool B = n_dimensions == 1, cpp_enable_if(B)>
    value_type& operator()(std::size_t i) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

    template <bool B = n_dimensions == 1, cpp_enable_if(B)>
    const value_type& operator()(std::size_t i) const noexcept {
        cpp_assert(i < dim(0), "Out of bounds");

        return _memory[i];
    }

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

    template <typename... S>
    std::size_t index(S... sizes) const noexcept {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        std::size_t index = 0;

        if (storage_order == order::RowMajor) {
            std::size_t subsize = _size;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    subsize /= dim(i++);
                    index += subsize * s;
                },
                sizes...);
        } else {
            std::size_t subsize = 1;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
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
    const value_type& operator()(S... sizes) const noexcept {
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
    value_type& operator()(S... sizes) noexcept {
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
        return V::load(_memory + i);
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
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return _memory + _size;
    }

    std::size_t& unsafe_dimension_access(std::size_t i) {
        cpp_assert(i < n_dimensions, "Out of bounds");
        return _dimensions[i];
    }
};

static_assert(std::is_nothrow_default_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<dyn_vector<double>>::value, "dyn_vector should be nothrow destructible");

template <typename T, order SO, std::size_t D>
void swap(dyn_matrix_impl<T, SO, D>& lhs, dyn_matrix_impl<T, SO, D>& rhs) {
    lhs.swap(rhs);
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
