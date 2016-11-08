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
struct dyn_matrix_impl final : dense_dyn_base<dyn_matrix_impl<T, SO, D>, T, SO, D>,
                               inplace_assignable<dyn_matrix_impl<T, SO, D>>,
                               comparable<dyn_matrix_impl<T, SO, D>>,
                               expression_able<dyn_matrix_impl<T, SO, D>>,
                               value_testable<dyn_matrix_impl<T, SO, D>>,
                               dim_testable<dyn_matrix_impl<T, SO, D>> {
    static constexpr std::size_t n_dimensions = D;                              ///< The number of dimensions
    static constexpr order storage_order      = SO;                             ///< The storage order
    static constexpr std::size_t alignment    = intrinsic_traits<T>::alignment; ///< The memory alignment

    using base_type              = dense_dyn_base<dyn_matrix_impl<T, SO, D>, T, SO, D>; ///< The base type
    using value_type             = T;                                               ///< The value type
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;           ///< The type used to store the dimensions
    using memory_type            = value_type*;                                     ///< The memory type
    using const_memory_type      = const value_type*;                               ///< The const memory type
    using iterator               = memory_type;                                     ///< The type of iterator
    using const_iterator         = const_memory_type;                               ///< The type of const iterator

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<T>;

private:
    using base_type::_size;
    using base_type::_dimensions;
    using base_type::_memory;

    mutable gpu_handler<T> _gpu_memory_handler; ///< The GPU memory handler

    using base_type::release;
    using base_type::allocate;
    using base_type::check_invariants;
    using base_type::index;

public:
    using base_type::dim;
    using base_type::memory_start;
    using base_type::memory_end;
    using base_type::begin;
    using base_type::end;

    // Construction

    /*!
     * \brief Construct an empty matrix
     *
     * This matrix don't have any memory nor dimensionsand most
     * operations will likely fail on it
     */
    dyn_matrix_impl() noexcept : base_type() {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    dyn_matrix_impl(const dyn_matrix_impl& rhs) noexcept : base_type(rhs) {
        _memory = allocate(alloc_size<T>(_size));

        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Move construct a matrix
     * \param rhs The matrix to move
     */
    dyn_matrix_impl(dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)) {
        _memory = rhs._memory;
        rhs._memory = nullptr;
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    template <typename T2, order SO2, std::size_t D2, cpp_enable_if(SO2 == SO)>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO2, D2>& rhs) noexcept : base_type(rhs){
        _memory = allocate(alloc_size<T>(_size));

        direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    template <typename T2, order SO2, std::size_t D2, cpp_disable_if(SO2 == SO)>
    dyn_matrix_impl(const dyn_matrix_impl<T2, SO2, D2>& rhs) noexcept : base_type(rhs){
        _memory = allocate(alloc_size<T>(_size));

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
            : base_type(e){
        _memory = allocate(alloc_size<T>(_size));

        assign_evaluate(e, *this);
    }

    /*!
     * \brief Construct a vector with the given values
     * \param list Initializer list containing all the values of the vector
     */
    dyn_matrix_impl(std::initializer_list<value_type> list) noexcept : base_type(list.size(), {{list.size()}}) {
        _memory = allocate(alloc_size<T>(_size));

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
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(sizes...), {{static_cast<std::size_t>(sizes)...}}) {
        _memory = allocate(alloc_size<T>(_size));
    }

    /*!
     * \brief Construct a matrix with the given dimensions and values
     * \param sizes The dimensions of the matrix followed by an initializer_list
     */
    template <typename... S, cpp_enable_if(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        _memory = allocate(alloc_size<T>(_size));

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
                                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        _memory = allocate(alloc_size<T>(_size));

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
                                                            ){
        _memory = allocate(alloc_size<T>(_size));

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
                                              std::is_convertible<std::size_t, S1>::value,                                              //The first type must be convertible to size_t
                                              cpp::is_sub_homogeneous<S1, S...>::value,                                                 //The first N-1 types must homegeneous
                                              cpp::is_specialization_of<generator_expr, typename cpp::last_type<S1, S...>::type>::value //The last type must be a generator expr
                                              )>
    explicit dyn_matrix_impl(S1 s1, S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                                     dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        _memory = allocate(alloc_size<T>(_size));

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
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-2)>(), sizes...)) {
        _memory = allocate(alloc_size<T>(_size));

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
            : base_type(container.size(), {{container.size()}}){
        _memory = allocate(alloc_size<T>(_size));

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
                _memory     = allocate(alloc_size<T>(_size));
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
            auto new_memory = allocate(alloc_size<T>(new_size));

            for (std::size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, _size);

            _memory = new_memory;
        } else {
            _memory     = allocate(alloc_size<T>(new_size));
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
            auto new_memory = allocate(alloc_size<T>(new_size));

            for (std::size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, _size);

            _memory = new_memory;
        } else {
            _memory     = allocate(alloc_size<T>(new_size));
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
        // It is possible that the matrix was not initialized before
        // In the case, get the the dimensions from the expression and
        // initialize the matrix
        if(!_memory){
            inherit(e);
        } else {
            validate_assign(*this, e);
        }

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
        if(_memory){
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

        //TODO swap is likely screwing up GPU memory!

        check_invariants();
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(void) store(const vec_type<V> in, std::size_t i) noexcept {
        V::store(_memory + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(void) storeu(const vec_type<V> in, std::size_t i) noexcept {
        V::storeu(_memory + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(void) stream(const vec_type<V> in, std::size_t i) noexcept {
        V::stream(_memory + i, in);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) load(std::size_t i) const noexcept {
        return V::load(_memory + i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) loadu(std::size_t i) const noexcept {
        return V::loadu(_memory + i);
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

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<T, n_dimensions> direct() const {
        return opaque_memory<T, n_dimensions>(memory_start(), _size, _dimensions, _gpu_memory_handler, SO);
    }

    /*!
     * \brief Inherit the dimensions of an ETL expressions, if the matrix has no
     * dimensions.
     * \param e The expression to get the dimensions from.
     */
    template <typename E>
    void inherit_if_null(const E& e){
        static_assert(n_dimensions == etl::dimensions(e), "Cannot inherit from an expression with different number of dimensions");
        static_assert(!etl::decay_traits<E>::is_generator, "Cannot inherit dimensions from a generator expression");

        if(!_memory){
            inherit(e);
        }
    }

private:
    /*!
     * \brief Inherit the dimensions of an ETL expressions.
     * This must only be called when the matrix has no dimensions
     * \param e The expression to get the dimensions from.
     */
    template <typename E, cpp_enable_if(etl::decay_traits<E>::is_generator)>
    void inherit(const E& e){
        cpp_assert(false, "Impossible to inherit dimensions from generators");
        cpp_unused(e);
    }

    /*!
     * \brief Inherit the dimensions of an ETL expressions.
     * This must only be called when the matrix has no dimensions
     * \param e The expression to get the dimensions from.
     */
    template <typename E, cpp_disable_if(etl::decay_traits<E>::is_generator)>
    void inherit(const E& e){
        cpp_assert(n_dimensions == etl::dimensions(e), "Invalid number of dimensions");

        // Compute the size and new dimensions
        _size = 1;
        for (std::size_t d = 0; d < n_dimensions; ++d) {
            _dimensions[d] = etl::dim(e, d);
            _size *= _dimensions[d];
        }

        // Allocate the new memory
        _memory = allocate(alloc_size<T>(_size));
    }
};

static_assert(std::is_nothrow_default_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<dyn_vector<double>>::value, "dyn_vector should be nothrow destructible");

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

/*!
 * \brief Print the description of the matrix to the given stream
 * \param os The output stream
 * \param mat The matrix to output the description to the stream
 * \return The given output stream
 */
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
