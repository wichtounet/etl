//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime matrix implementation
 */

#pragma once

#include "etl/dyn_base.hpp" //The base class and utilities

namespace etl {

/*!
 * \brief GPU special Matrix with run-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, order SO, size_t D>
struct gpu_dyn_matrix_impl final : dense_dyn_base<gpu_dyn_matrix_impl<T, SO, D>, T, SO, D>, dim_testable<gpu_dyn_matrix_impl<T, SO, D>> {
    static constexpr size_t n_dimensions = D;                                      ///< The number of dimensions
    static constexpr order storage_order = SO;                                     ///< The storage order
    static constexpr size_t alignment    = default_intrinsic_traits<T>::alignment; ///< The memory alignment

    using this_type              = gpu_dyn_matrix_impl<T, SO, D>;              ///< The type of this expression
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

    using base_type::check_invariants;

public:
    using base_type::_gpu;

    // Construction

    /*!
     * \brief Construct an empty matrix
     *
     * This matrix don't have any memory nor dimensionsand most
     * operations will likely fail on it
     */
    gpu_dyn_matrix_impl() noexcept : base_type() {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a matrix
     * \param rhs The matrix to copy
     */
    gpu_dyn_matrix_impl(const gpu_dyn_matrix_impl& rhs) noexcept(assert_nothrow) : base_type(rhs) {
        // The CPU status is discarded!

        cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "gpu_dyn_matrix_impl(&) must preserve GPU status");
    }

    /*!
     * \brief Move construct a matrix
     * \param rhs The matrix to move
     */
    gpu_dyn_matrix_impl(gpu_dyn_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)) {
        _memory     = rhs._memory;
        rhs._memory = nullptr;
    }

    /*!
     * \brief Copy assign from another matrix
     *
     * This operator can change the dimensions of the matrix
     *
     * \param rhs The matrix to copy from
     * \return A reference to the matrix
     */
    gpu_dyn_matrix_impl& operator=(const gpu_dyn_matrix_impl& rhs) noexcept(assert_nothrow) {
        if (this != &rhs) {
            if (!_size) {
                _size       = rhs._size;
                _dimensions = rhs._dimensions;
            } else {
                validate_assign(*this, rhs);
            }

            // TODO Find a better solution
            const_cast<gpu_dyn_matrix_impl&>(rhs).assign_to(*this);
        }

        check_invariants();

        cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "gpu_dyn_matrix_impl::operator= must preserve GPU status");

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
    gpu_dyn_matrix_impl& operator=(gpu_dyn_matrix_impl&& rhs) noexcept {
        if (this != &rhs) {
            _gpu = std::move(rhs._gpu);

            _size       = rhs._size;
            _dimensions = std::move(rhs._dimensions);

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
    template <etl_expr E>
    gpu_dyn_matrix_impl& operator=(E&& e) noexcept requires(!std::same_as<std::decay_t<E>, gpu_dyn_matrix_impl<T, SO, D>> && std::convertible_to<value_t<E>, value_type>) {
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
     * \brief Assign a value to all elements of the matrix
     * \param v The vlaue to assign to all elements
     */
    gpu_dyn_matrix_impl& operator=([[maybe_unused]] value_type v) noexcept {
        cpp_unreachable("Invalid call to operator=(value_type) for GPU dyn matrix");

        return *this;
    }

    /*!
     * \brief Destruct the matrix and release all its memory
     */
    ~gpu_dyn_matrix_impl() noexcept {
        // Nothing to do here
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

    // Assignment functions

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
     * \brief Subtract from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit([[maybe_unused]] const detail::evaluator_visitor& visitor) const {}

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
        }
    }

    /*!
     * \brief Resize the matrix as a scalar value.
     *
     * This must only be called when the matrix has no dimensions
     */
    void resize_scalar() {
        _size          = 1;
        _dimensions[0] = 1;
    }

    /*!
     * \brief Print the description of the matrix to the given stream
     * \param os The output stream
     * \param mat The matrix to output the description to the stream
     * \return The given output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const gpu_dyn_matrix_impl& mat) {
        if (D == 1) {
            return os << "GV[" << mat.size() << "]";
        }

        os << "GM[" << mat.dim(0);

        for (size_t i = 1; i < D; ++i) {
            os << "," << mat.dim(i);
        }

        return os << "]";
    }
};

#ifndef CPP_UTILS_ASSERT_EXCEPTION
static_assert(std::is_nothrow_default_constructible_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible_v<gpu_dyn_matrix<double>>, "dyn_vector should be nothrow destructible");
#endif

} //end of namespace etl
