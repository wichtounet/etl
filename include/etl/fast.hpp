//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
#include "etl/direct_fill.hpp" //direct_fill with GPU support

namespace etl {

/*!
 * \brief Matrix with compile-time fixed dimensions.
 *
 * The matrix support an arbitrary number of dimensions.
 */
template <typename T, typename ST, order SO, size_t... Dims>
struct fast_matrix_impl final : fast_matrix_base<fast_matrix_impl<T, ST, SO, Dims...>, T, ST, SO, Dims...>,
                                inplace_assignable<fast_matrix_impl<T, ST, SO, Dims...>>,
                                expression_able<fast_matrix_impl<T, ST, SO, Dims...>>,
                                value_testable<fast_matrix_impl<T, ST, SO, Dims...>>,
                                iterable<fast_matrix_impl<T, ST, SO, Dims...>, SO == order::RowMajor>,
                                dim_testable<fast_matrix_impl<T, ST, SO, Dims...>> {
    static_assert(sizeof...(Dims) > 0, "At least one dimension must be specified");

public:
    static constexpr size_t n_dimensions = sizeof...(Dims);                        ///< The number of dimensions
    static constexpr size_t etl_size     = (Dims * ...);                           ///< The size of the matrix
    static constexpr order storage_order = SO;                                     ///< The storage order
    static constexpr bool array_impl     = !matrix_detail::is_vector<ST>;          ///< true if the storage is an std::arraw, false otherwise
    static constexpr size_t alignment    = default_intrinsic_traits<T>::alignment; ///< The memory alignment

    using this_type          = fast_matrix_impl<T, ST, SO, Dims...>;            ///< this type
    using base_type          = fast_matrix_base<this_type, T, ST, SO, Dims...>; ///< The base type
    using iterable_base_type = iterable<this_type, SO == order::RowMajor>;      ///< The iterable base type
    using value_type         = T;                                               ///< The value type
    using storage_impl       = ST;                                              ///< The storage implementation
    using memory_type        = value_type*;                                     ///< The memory type
    using const_memory_type  = const value_type*;                               ///< The const memory type

    using base_type::dim;
    using base_type::memory_end;
    using base_type::memory_start;
    using base_type::size;
    using iterable_base_type::begin;
    using iterable_base_type::end;

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

private:
    using base_type::_data;

public:
    /// Construction

    /*!
     * \brief Construct an empty fast matrix
     */
    fast_matrix_impl() noexcept : base_type() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a fast matrix filled with the same value
     * \param value the value to fill the matrix with
     */
    template <typename VT, cpp_enable_iff(std::is_convertible_v<VT, value_type> || std::is_assignable_v<T&, VT>)>
    explicit fast_matrix_impl(const VT& value) noexcept : base_type() {
        // Fill the matrix
        std::fill(begin(), end(), value);
    }

    /*!
     * \brief Construct a fast matrix filled with the given values
     * \param l the list of values to fill the matrix with
     */
    fast_matrix_impl(std::initializer_list<value_type> l) : base_type() {
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
    fast_matrix_impl(const fast_matrix_impl& rhs) noexcept : base_type(rhs) {
        // Nothing else to init
    }

    /*!
     * \brief Move construct a fast matrix
     * \param rhs The fast matrix to move from
     */
    fast_matrix_impl(fast_matrix_impl&& rhs) noexcept : base_type(std::move(rhs)) {
        // Nothing else to init
    }

    /*!
     * \brief Construct a fast matrix from the given STL container
     * \param container The container to get values from
     */
    template <typename Container,
              cpp_enable_iff(!is_complex_t<Container> && std::is_convertible_v<typename Container::value_type, value_type> && !is_etl_expr<Container>)>
    explicit fast_matrix_impl(const Container& container) : base_type() {
        validate_assign(*this, container);
        std::copy(container.begin(), container.end(), begin());
    }

    // Assignment

    /*!
     * \brief Copy assign a fast matrix
     * \param rhs The fast matrix to copy from
     * \return a reference to the fast matrix
     */
    fast_matrix_impl& operator=(const fast_matrix_impl& rhs) noexcept(assert_nothrow) {
        // Avoid copy to self
        if (this != &rhs) {
            // This will handle the possible copy to GPU
            this->_gpu = rhs._gpu;

            // If necessary, perform the actual copy to CPU
            if (this->is_cpu_up_to_date()) {
                _data = rhs._data;
            }

            cpp_assert(rhs.is_cpu_up_to_date() == this->is_cpu_up_to_date(), "fast::operator= must preserve CPU status");
            cpp_assert(rhs.is_gpu_up_to_date() == this->is_gpu_up_to_date(), "fast::operator= must preserve GPU status");
        }

        return *this;
    }

    /*!
     * \brief Copy assign a fast matrix
     * \param rhs The fast matrix to copy from
     * \return a reference to the fast matrix
     */
    fast_matrix_impl& operator=(fast_matrix_impl&& rhs) noexcept {
        // Avoid move to self
        if (this != &rhs) {
            // This will handle the possible copy to GPU
            this->_gpu = std::move(rhs._gpu);

            // If necessary, perform the actual copy to CPU
            if (this->is_cpu_up_to_date()) {
                _data = std::move(rhs._data);
            }
        }

        return *this;
    }

    /*!
     * \brief Copy assign a fast matrix from a matrix fast matrix type
     * \param rhs The fast matrix to copy from
     * \return a reference to the fast matrix
     */
    template <size_t... SDims>
    fast_matrix_impl& operator=(const fast_matrix_impl<T, ST, SO, SDims...>& rhs) noexcept {
        // Make sure the assign is valid
        validate_assign(*this, rhs);

        // Since the type is different, it is handled by the
        // evaluator which will handle all the possible cases
        rhs.assign_to(*this);

        return *this;
    }

    /*!
     * \brief Assign the values of the STL container to the fast matrix
     * \param container The STL container to get the values from
     * \return a reference to the fast matrix
     */
    template <std_container Container>
    fast_matrix_impl& operator=(const Container& container) noexcept requires(std::convertible_to<typename Container::value_type, value_type>) {
        validate_assign(*this, container);
        std::copy(container.begin(), container.end(), begin());

        this->validate_cpu();
        this->invalidate_gpu();

        return *this;
    }

    /*!
     * \brief Assign the values of the ETL expression to the fast matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <typename E,
              cpp_enable_iff(is_etl_expr<E> && std::is_convertible_v<value_t<E>, value_type> && !std::is_same_v<std::decay_t<E>, this_type>)>
    fast_matrix_impl& operator=(E&& e) {
        validate_assign(*this, e);

        // Avoid aliasing issues
        if constexpr (!decay_traits<E>::is_linear) {
            if (e.alias(*this)) {
                // Create a temporary to hold the result
                auto tmp = force_temporary_dim_only(*this);

                // Assign the expression to the temporary
                e.assign_to(tmp);

                // Assign the temporary to this matrix
                *this = tmp;
            } else {
                e.assign_to(*this);
            }
        } else {
            // Direct assignment of the expression into this matrix
            e.assign_to(*this);
        }

        return *this;
    }

    /*!
     * \brief Assign the value to each element
     * \param value The value to assign to each element
     * \return a reference to the fast matrix
     */
    template <typename VT, cpp_enable_iff(std::is_convertible_v<VT, value_type> || std::is_assignable_v<T&, VT>)>
    fast_matrix_impl& operator=(const VT& value) noexcept {
        direct_fill(*this, value);

        return *this;
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
    void store(vec_type<V> in, size_t i) noexcept {
        V::store(memory_start() + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t i) noexcept {
        V::storeu(memory_start() + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t i) noexcept {
        V::stream(memory_start() + i, in);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t i) const noexcept {
        return V::load(memory_start() + i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t i) const noexcept {
        return V::loadu(memory_start() + i);
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
     * \brief Prints a fast matrix type (not the contents) to the given stream
     * \param os The output stream
     * \param matrix The fast matrix to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, [[maybe_unused]] const fast_matrix_impl& matrix) {
        if constexpr (sizeof...(Dims) == 1) {
            return os << "V[" << concat_sizes(Dims...) << "]";
        }

        return os << "M[" << concat_sizes(Dims...) << "]";
    }
};

#ifndef CPP_UTILS_ASSERT_EXCEPTION
static_assert(std::is_nothrow_default_constructible_v<fast_vector<double, 2>>, "fast_vector should be nothrow default constructible");
static_assert(std::is_nothrow_copy_constructible_v<fast_vector<double, 2>>, "fast_vector should be nothrow copy constructible");
static_assert(std::is_nothrow_move_constructible_v<fast_vector<double, 2>>, "fast_vector should be nothrow move constructible");
static_assert(std::is_nothrow_copy_assignable_v<fast_vector<double, 2>>, "fast_vector should be nothrow copy assignable");
static_assert(std::is_nothrow_move_assignable_v<fast_vector<double, 2>>, "fast_vector should be nothrow move assignable");
static_assert(std::is_nothrow_destructible_v<fast_vector<double, 2>>, "fast_vector should be nothrow destructible");
#endif

/*!
 * \brief Create a fast_matrix of the given dimensions over the given memory
 * \param memory The memory
 * \return A fast_matrix using the given memory
 *
 * The memory must be large enough to hold the matrix
 */
template <size_t... Dims, typename T>
fast_matrix_impl<T, std::span<T>, order::RowMajor, Dims...> fast_matrix_over(T* memory) {
    return fast_matrix_impl<T, std::span<T>, order::RowMajor, Dims...>(std::span<T>(memory, (Dims * ...)));
}

/*!
 * \brief Swaps the given two matrices
 * \param lhs The first matrix to swap
 * \param rhs The second matrix to swap
 */
template <typename T, typename ST, order SO, size_t... Dims>
void swap(fast_matrix_impl<T, ST, SO, Dims...>& lhs, fast_matrix_impl<T, ST, SO, Dims...>& rhs) {
    lhs.swap(rhs);
}

/*!
 * \brief Serialize the given matrix using the given serializer
 * \param os The serializer
 * \param matrix The matrix to serialize
 */
template <typename Stream, typename T, typename ST, order SO, size_t... Dims>
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
template <typename Stream, typename T, typename ST, order SO, size_t... Dims>
void deserialize(deserializer<Stream>& os, fast_matrix_impl<T, ST, SO, Dims...>& matrix) {
    for (auto& value : matrix) {
        os >> value;
    }
}

} //end of namespace etl
