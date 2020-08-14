//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
#include "etl/direct_fill.hpp" //direct_fill with GPU support

namespace etl {

/*!
 * \brief Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, order SO, size_t D>
struct dyn_matrix_impl final : dense_dyn_base<dyn_matrix_impl<T, SO, D>, T, SO, D>,
                               inplace_assignable<dyn_matrix_impl<T, SO, D>>,
                               expression_able<dyn_matrix_impl<T, SO, D>>,
                               value_testable<dyn_matrix_impl<T, SO, D>>,
                               iterable<dyn_matrix_impl<T, SO, D>, SO == order::RowMajor>,
                               dim_testable<dyn_matrix_impl<T, SO, D>> {
    static constexpr size_t n_dimensions = D;                                      ///< The number of dimensions
    static constexpr order storage_order = SO;                                     ///< The storage order
    static constexpr size_t alignment    = default_intrinsic_traits<T>::alignment; ///< The memory alignment

    using this_type              = dyn_matrix_impl<T, SO, D>;                  ///< The type of this expression
    using base_type              = dense_dyn_base<this_type, T, SO, D>;        ///< The base type
    using iterable_base_type     = iterable<this_type, SO == order::RowMajor>; ///< The iterable base type
    using value_type             = T;                                          ///< The value type
    using dimension_storage_impl = std::array<size_t, n_dimensions>;           ///< The type used to store the dimensions
    using memory_type            = value_type*;                                ///< The memory type
    using const_memory_type      = const value_type*;                          ///< The const memory type

    using iterator       = std::conditional_t<SO == order::RowMajor, value_type*, etl::iterator<this_type>>;             ///< The iterator type
    using const_iterator = std::conditional_t<SO == order::RowMajor, const value_type*, etl::iterator<const this_type>>; ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

private:
    using base_type::_dimensions;
    using base_type::_memory;
    using base_type::_size;

    using base_type::allocate;
    using base_type::check_invariants;
    using base_type::release;

public:
    using base_type::dim;
    using base_type::memory_end;
    using base_type::memory_start;
    using iterable_base_type::begin;
    using iterable_base_type::end;

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
    dyn_matrix_impl(const dyn_matrix_impl& rhs) noexcept(assert_nothrow) : base_type(rhs) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

        if (rhs.is_cpu_up_to_date()) {
            direct_copy(rhs.memory_start(), rhs.memory_end(), memory_start());
        }

        cpp_assert(rhs.is_cpu_up_to_date() == this->is_cpu_up_to_date(), "dyn_matrix_impl(&) must preserve CPU status");
        cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "dyn_matrix_impl(&) must preserve GPU status");
    }

    /*!
     * \brief Move construct a matrix
     * \param rhs The matrix to move
     */
    dyn_matrix_impl(dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)) {
        _memory     = rhs._memory;

        rhs._size   = 0;
        rhs._memory = nullptr;
    }

    /*!
     * \brief Construct a vector with the given values
     * \param list Initializer list containing all the values of the vector
     */
    dyn_matrix_impl(std::initializer_list<value_type> list) noexcept : base_type(list.size(), {{list.size()}}) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

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
    template <typename... S, cpp_enable_iff((sizeof...(S) == D) && cpp::all_convertible_to_v<size_t, S...>)>
    explicit dyn_matrix_impl(S... sizes) noexcept : base_type(util::size(sizes...), {{static_cast<size_t>(sizes)...}}) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
    }

    /*!
     * \brief Construct a matrix with the given dimensions and initializer_list
     * \param sizes The dimensions of the matrix followed by an initializer_list
     */
    template <typename... S, cpp_enable_iff(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit dyn_matrix_impl(S... sizes) noexcept
            : base_type(util::size(std::make_index_sequence<(sizeof...(S) - 1)>(), sizes...),
                        dyn_detail::sizes(std::make_index_sequence<(sizeof...(S) - 1)>(), sizes...)) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        std::copy(list.begin(), list.end(), begin());
    }

    /*!
     * \brief Construct a matrix with the given dimensions and values
     * \param sizes The dimensions of the matrix followed by a values_t
     */
    template <typename... S, cpp_enable_iff((sizeof...(S) == D) && cpp::is_specialization_of_v<values_t, typename cpp::last_type<size_t, S...>::type>)>
    explicit dyn_matrix_impl(size_t s1, S... sizes) noexcept
            : base_type(util::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                        dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

        auto list = cpp::last_value(sizes...).template list<value_type>();
        std::copy(list.begin(), list.end(), begin());
    }

    /*!
     * \brief Construct a matrix with the given dimensions and a value
     * \param sizes The dimensions of the matrix followed by a values
     *
     * Every element of the matrix will be set to this value.
     */
    template <typename... S, cpp_enable_iff((sizeof...(S) == D) && !cpp::is_specialization_of_v<values_t, typename cpp::last_type<size_t, S...>::type>)>
    explicit dyn_matrix_impl(size_t s1, S... sizes) noexcept
            : base_type(util::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                        dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

        decltype(auto) value = cpp::last_value(s1, sizes...);
        std::fill(begin(), end(), value);
    }

    /*!
     * \brief Construct a vector from a Container
     * \param container A STL container
     *
     * Only possible for 1D matrices
     */
    template <typename Container, cpp_enable_iff(std::is_convertible_v<typename Container::value_type, value_type>)>
    explicit dyn_matrix_impl(const Container& container) : base_type(container.size(), {{container.size()}}) {
        _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

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
    dyn_matrix_impl& operator=(const dyn_matrix_impl& rhs) noexcept(assert_nothrow) {
        if (this != &rhs) {
            if (!_size) {
                _size       = rhs._size;
                _dimensions = rhs._dimensions;
                _memory     = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
            } else {
                validate_assign(*this, rhs);
            }

            // TODO Find a better solution
            const_cast<dyn_matrix_impl&>(rhs).assign_to(*this);
        }

        check_invariants();

        cpp_assert(rhs.is_cpu_up_to_date() == this->is_cpu_up_to_date(), "dyn_matrix_impl::operator= must preserve CPU status");
        cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "dyn_matrix_impl::operator= must preserve GPU status");

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
            if (_memory) {
                release(_memory, alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
            }

            _size       = rhs._size;
            _dimensions = std::move(rhs._dimensions);
            _memory     = rhs._memory;

            rhs._size   = 0;
            rhs._memory = nullptr;
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Resize with the new dimensions in the given array
     * \param dimensions The new dimensions
     */
    void resize_arr(const dimension_storage_impl& dimensions) {
        auto new_size = std::accumulate(dimensions.begin(), dimensions.end(), size_t(1), std::multiplies<size_t>());

        if (_memory) {
            auto new_memory = allocate(alloc_size_mat<T>(new_size, (dimensions.back())));

            for (size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

            _memory = new_memory;
        } else {
            _memory = allocate(alloc_size_mat<T>(new_size, (dimensions.back())));
        }

        _size       = new_size;
        _dimensions = dimensions;
    }

    /*!
     * \brief Resize with the new given dimensions
     * \param sizes The new dimensions
     */
    template <typename... Sizes>
    void resize(Sizes... sizes) {
        static_assert(sizeof...(Sizes), "Cannot change number of dimensions");

        auto new_size = util::size(sizes...);

        if (_memory) {
            auto new_memory = allocate(alloc_size_mat<T>(new_size, cpp::last_value(sizes...)));

            for (size_t i = 0; i < std::min(_size, new_size); ++i) {
                new_memory[i] = _memory[i];
            }

            release(_memory, alloc_size_mat<T>(_size, dim(n_dimensions - 1)));

            _memory = new_memory;
        } else {
            _memory = allocate(alloc_size_mat<T>(new_size, cpp::last_value(sizes...)));
        }

        _size       = new_size;
        _dimensions = dyn_detail::sizes(std::make_index_sequence<D>(), sizes...);
    }

    /*!
     * \brief Assign from an ETL expression.
     * \param e The expression containing the values to assign to the matrix
     * \return A reference to the matrix
     */
    template <typename E,
              cpp_enable_iff(!std::is_same_v<std::decay_t<E>, dyn_matrix_impl<T, SO, D>> && std::is_convertible_v<value_t<E>, value_type>
                             && is_etl_expr<E>)>
    dyn_matrix_impl& operator=(E&& e) noexcept {
        // It is possible that the matrix was not initialized before
        // In the case, get the the dimensions from the expression and
        // initialize the matrix
        if (!_memory) {
            inherit(e);
        } else {
            validate_assign(*this, e);
        }

        // Avoid aliasing issues
        if constexpr (!decay_traits<E>::is_linear) {
            if (e.alias(*this)) {
                // Create a temporary to hold the result
                this_type tmp;

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
     * \brief Assign from an STL container.
     * \param vec The container containing the values to assign to the matrix
     * \return A reference to the matrix
     */
    template <typename Container, cpp_enable_iff(!is_etl_expr<Container> && std::is_convertible_v<typename Container::value_type, value_type>)>
    dyn_matrix_impl& operator=(const Container& vec) {
        // Inherit from the dimensions if possible
        if (!_memory && D == 1) {
            // Compute the size and new dimensions
            _size          = vec.size();
            _dimensions[0] = vec.size();

            // Allocate the new memory
            _memory = allocate(alloc_size_mat<T>(_size, dim(0)));
        } else {
            validate_assign(*this, vec);
        }

        std::copy(vec.begin(), vec.end(), begin());

        this->validate_cpu();
        this->invalidate_gpu();

        check_invariants();

        return *this;
    }

    /*!
     * \brief Assign the same value to each element of the matrix
     * \param value The value to assign to each element of the matrix
     * \return A reference to the matrix
     */
    dyn_matrix_impl& operator=(const value_type& value) noexcept {
        direct_fill(*this, value);

        check_invariants();

        return *this;
    }

    /*!
     * \brief Destruct the matrix and release all its memory
     */
    ~dyn_matrix_impl() noexcept {
        if (_memory) {
            release(_memory, alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
        }
    }

    /*!
     * \brief Release all memory hold by the matrix.
     *
     * Using a matrix after it has been cleared is considered as Undefined Behaviour.
     */
    void clear() {
        if (_memory) {
            release(_memory, alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
        }

        _memory = nullptr;
        _size   = 0;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    auto& gpu_compute_hint([[maybe_unused]] Y& y) {
        this->ensure_gpu_up_to_date();
        return *this;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    const auto& gpu_compute_hint([[maybe_unused]] Y& y) const {
        this->ensure_gpu_up_to_date();
        return *this;
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
    ETL_STRONG_INLINE(void)
    store(const vec_type<V> in, size_t i) noexcept {
        V::store(_memory + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(void)
    storeu(const vec_type<V> in, size_t i) noexcept {
        V::storeu(_memory + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(void)
    stream(const vec_type<V> in, size_t i) noexcept {
        V::stream(_memory + i, in);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    load(size_t i) const noexcept {
        return V::load(_memory + i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    loadu(size_t i) const noexcept {
        return V::loadu(_memory + i);
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    size_t& unsafe_dimension_access(size_t i) {
        cpp_assert(i < n_dimensions, "Out of bounds");
        return _dimensions[i];
    }

    /*!
     * \brief Inherit the dimensions of an ETL expressions, if the matrix has no
     * dimensions.
     * \param e The expression to get the dimensions from.
     */
    template <typename E>
    void inherit_if_null(const E& e) {
        static_assert(n_dimensions == etl::decay_traits<E>::dimensions(), "Cannot inherit from an expression with different number of dimensions");
        static_assert(!etl::decay_traits<E>::is_generator, "Cannot inherit dimensions from a generator expression");

        if (!_memory) {
            inherit(e);
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Subtract from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit([[maybe_unused]] const detail::evaluator_visitor& visitor) const {}

private:
    /*!
     * \brief Inherit the dimensions of an ETL expressions.
     * This must only be called when the matrix has no dimensions
     * \param e The expression to get the dimensions from.
     */
    template <typename E>
    void inherit([[maybe_unused]] const E& e) {
        if constexpr (etl::decay_traits<E>::is_generator) {
            cpp_unreachable("Impossible to inherit dimensions from generators");
        } else {
            cpp_assert(n_dimensions == etl::dimensions(e), "Invalid number of dimensions");

            // Compute the size and new dimensions
            _size = 1;
            for (size_t d = 0; d < n_dimensions; ++d) {
                _dimensions[d] = etl::dim(e, d);
                _size *= _dimensions[d];
            }

            // Allocate the new memory
            _memory = allocate(alloc_size_mat<T>(_size, dim(n_dimensions - 1)));
        }
    }

    /*!
     * \brief Print the description of the matrix to the given stream
     * \param os The output stream
     * \param mat The matrix to output the description to the stream
     * \return The given output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const dyn_matrix_impl& mat) {
        if (D == 1) {
            return os << "V[" << mat.size() << "]";
        }

        os << "M[" << mat.dim(0);

        for (size_t i = 1; i < D; ++i) {
            os << "," << mat.dim(i);
        }

        return os << "]";
    }
};

#ifndef CPP_UTILS_ASSERT_EXCEPTION
static_assert(std::is_nothrow_default_constructible_v<dyn_vector<double>>, "dyn_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible_v<dyn_vector<double>>, "dyn_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible_v<dyn_vector<double>>, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable_v<dyn_vector<double>>, "dyn_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable_v<dyn_vector<double>>, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible_v<dyn_vector<double>>, "dyn_vector should be nothrow destructible");
#endif

/*!
 * \brief Helper to create a dyn matrix using the dimensions.
 *
 * This avoids having to set the number of dimensions.
 *
 * \tparam T The type contained in the matrix
 * \param sizes The dimensions
 *
 * \return A dyn matrix of the given dimensions.
 */
template <typename T, typename... Sizes>
etl::dyn_matrix<T, sizeof...(Sizes)> make_dyn_matrix(Sizes... sizes) {
    return etl::dyn_matrix<T, sizeof...(Sizes)>(sizes...);
}

/*!
 * \brief Swap two dyn matrix
 * \param lhs The first matrix
 * \param rhs The second matrix
 */
template <typename T, order SO, size_t D>
void swap(dyn_matrix_impl<T, SO, D>& lhs, dyn_matrix_impl<T, SO, D>& rhs) {
    lhs.swap(rhs);
}

/*!
 * \brief Serialize the given matrix using the given serializer
 * \param os The serializer
 * \param matrix The matrix to serialize
 */
template <typename Stream, typename T, order SO, size_t D>
void serialize(serializer<Stream>& os, const dyn_matrix_impl<T, SO, D>& matrix) {
    for (size_t i = 0; i < etl::dimensions(matrix); ++i) {
        os << matrix.dim(i);
    }

    for (const auto& value : matrix) {
        os << value;
    }
}

/*!
 * \brief Deserialize the given matrix using the given serializer
 * \param is The deserializer
 * \param matrix The matrix to deserialize
 */
template <typename Stream, typename T, order SO, size_t D>
void deserialize(deserializer<Stream>& is, dyn_matrix_impl<T, SO, D>& matrix) {
    typename std::decay_t<decltype(matrix)>::dimension_storage_impl new_dimensions;

    for (auto& value : new_dimensions) {
        is >> value;
    }

    matrix.resize_arr(new_dimensions);

    for (auto& value : matrix) {
        is >> value;
    }
}

} //end of namespace etl
