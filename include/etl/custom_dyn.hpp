//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime-sized custom matrix implementation
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
struct custom_dyn_matrix_impl final : dense_dyn_base<custom_dyn_matrix_impl<T, SO, D>, T, SO, D>,
                                      inplace_assignable<custom_dyn_matrix_impl<T, SO, D>>,
                                      comparable<custom_dyn_matrix_impl<T, SO, D>>,
                                      expression_able<custom_dyn_matrix_impl<T, SO, D>>,
                                      value_testable<custom_dyn_matrix_impl<T, SO, D>>,
                                      iterable<custom_dyn_matrix_impl<T, SO, D>, SO == order::RowMajor>,
                                      dim_testable<custom_dyn_matrix_impl<T, SO, D>> {
    static constexpr std::size_t n_dimensions = D;                              ///< The number of dimensions
    static constexpr order storage_order      = SO;                             ///< The storage order
    static constexpr std::size_t alignment    = intrinsic_traits<T>::alignment; ///< The memory alignment

    using this_type              = custom_dyn_matrix_impl<T, SO, D>;
    using iterable_base_type     = iterable<this_type, SO == order::RowMajor>;
    using base_type              = dense_dyn_base<custom_dyn_matrix_impl<T, SO, D>, T, SO, D>; ///< The base type
    using value_type             = T;                                                          ///< The value type
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;                      ///< The type used to store the dimensions
    using memory_type            = value_type*;                                                ///< The memory type
    using const_memory_type      = const value_type*;                                          ///< The const memory type

    using iterator       = std::conditional_t<SO == order::RowMajor, value_type*, etl::iterator<this_type>>;             ///< The iterator type
    using const_iterator = std::conditional_t<SO == order::RowMajor, const value_type*, etl::iterator<const this_type>>; ///< The const iterator type

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

public:
    using base_type::dim;
    using base_type::memory_start;
    using base_type::memory_end;
    using iterable_base_type::begin;
    using iterable_base_type::end;

    // Construction

    custom_dyn_matrix_impl(const custom_dyn_matrix_impl& rhs) noexcept : base_type(rhs) {
        _memory = rhs._memory;
    }

    /*!
     * \brief Move construct a matrix
     * \param rhs The matrix to move
     */
    custom_dyn_matrix_impl(custom_dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)), _gpu_memory_handler(std::move(rhs._gpu_memory_handler)) {
        _memory = rhs._memory;
        rhs._memory = nullptr;
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
    explicit custom_dyn_matrix_impl(value_type* memory, S... sizes) noexcept : base_type(dyn_detail::size(sizes...), {{static_cast<std::size_t>(sizes)...}})
                                                    {
        _memory = memory;
        //Nothing else to init
    }

    custom_dyn_matrix_impl& operator=(const custom_dyn_matrix_impl& rhs) noexcept {
        if (this != &rhs) {
            _size       = rhs._size;
            _dimensions = rhs._dimensions;
            _memory     = rhs._memory;
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
    custom_dyn_matrix_impl& operator=(custom_dyn_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            _size               = rhs._size;
            _dimensions         = std::move(rhs._dimensions);
            _memory             = rhs._memory;
            _gpu_memory_handler = std::move(rhs._gpu_memory_handler);

            rhs._size   = 0;
            rhs._memory = nullptr;
        }

        check_invariants();

        return *this;
    }

    /*!
     * \brief Assign from an ETL expression.
     * \param e The expression containing the values to assign to the matrix
     * \return A reference to the matrix
     */
    template <typename E, cpp_enable_if(!std::is_same<std::decay_t<E>, custom_dyn_matrix_impl<T, SO, D>>::value, std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    custom_dyn_matrix_impl& operator=(E&& e) noexcept {
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
    custom_dyn_matrix_impl& operator=(const Container& vec) {
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
    custom_dyn_matrix_impl& operator=(const value_type& value) noexcept {
        std::fill(begin(), end(), value);

        check_invariants();

        return *this;
    }

    /*!
     * \brief Destruct the matrix
     */
    ~custom_dyn_matrix_impl() noexcept {
        // Nothing to do
    }

    /*!
     * \brief Swap the content of the matrix with the content of the given matrix
     * \param other The other matrix to swap content with
     */
    void swap(custom_dyn_matrix_impl& other) {
        using std::swap;
        swap(_size, other._size);
        swap(_dimensions, other._dimensions);
        swap(_memory, other._memory);

        //TODO swap is likely screwing up GPU memory!

        check_invariants();
    }

    // Accessors

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        V::storeu(memory_start() + i, in);
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
        V::storeu(memory_start() + i, in);
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
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    vec_type<V> loadu(std::size_t i) const noexcept {
        return V::loadu(_memory + i);
    }

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<T, n_dimensions> direct() const {
        return opaque_memory<T, n_dimensions>(memory_start(), _size, _dimensions, _gpu_memory_handler, SO);
    }
};

static_assert(std::is_nothrow_move_constructible<dyn_vector<double>>::value, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable<dyn_vector<double>>::value, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible<dyn_vector<double>>::value, "dyn_vector should be nothrow destructible");

/*!
 * \brief Swap two dyn matrix
 * \param lhs The first matrix
 * \param rhs The second matrix
 */
template <typename T, order SO, std::size_t D>
void swap(custom_dyn_matrix_impl<T, SO, D>& lhs, custom_dyn_matrix_impl<T, SO, D>& rhs) {
    lhs.swap(rhs);
}

/*!
 * \brief Print the description of the matrix to the given stream
 * \param os The output stream
 * \param mat The matrix to output the description to the stream
 * \return The given output stream
 */
template <typename T, order SO, std::size_t D>
std::ostream& operator<<(std::ostream& os, const custom_dyn_matrix_impl<T, SO, D>& mat) {
    if (D == 1) {
        return os << "CV[" << mat.size() << "]";
    }

    os << "CM[" << mat.dim(0);

    for (std::size_t i = 1; i < D; ++i) {
        os << "," << mat.dim(i);
    }

    return os << "]";
}

} //end of namespace etl
