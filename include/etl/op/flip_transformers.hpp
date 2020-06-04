//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/one_if_max_sub.hpp"

namespace etl {

/*!
 * \brief Transformer to implement one if max sub on 2D matrix.
 *
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct one_if_max_sub_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend etl_traits<one_if_max_sub_transformer>;

    static_assert(is_2d<sub_type>, "one_if_max_sub is only defined for 2D Matrix");

    static constexpr bool gpu_computable = impl::egblas::has_sone_if_max_sub && all_row_major<T> && all_floating<T>;

private:
    sub_type sub; ///< The subexpression

    std::vector<size_t> max_indices;

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit one_if_max_sub_transformer(sub_type expr) : sub(expr), max_indices(etl::dim<0>(expr)) {
        for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
            max_indices[i] = max_index(sub(i));
        }
    }

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        size_t i_i = i / dim<1>(sub);
        size_t i_j = i % dim<1>(sub);
        return i_j == max_indices[i_i] ? value_type(1) : value_type(0);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        size_t i_i = i / dim<1>(sub);
        size_t i_j = i % dim<1>(sub);
        return i_j == max_indices[i_i] ? value_type(1) : value_type(0);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return j == max_indices[i] ? value_type(1) : value_type(0);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename Y>
    auto gpu_compute_hint(Y& y) const noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(sub, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        impl::egblas::one_if_max_sub(etl::dim<0>(y), etl::dim<1>(y), 1, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        t2.validate_gpu();
        t2.invalidate_cpu();

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename Y>
    Y& gpu_compute(Y& y) const noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(sub, y);

        impl::egblas::one_if_max_sub(etl::dim<0>(y), etl::dim<1>(y), 1, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename V>
    void visit(V&& visitor) const {
        sub.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure sub value
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Display the transformer on the given stream
     * \param os The output stream
     * \param transformer The transformer to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const one_if_max_sub_transformer& transformer) {
        return os << "one_if_max_sub(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that flips a matrix horizontally
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct hflip_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend etl_traits<hflip_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit hflip_transformer(sub_type expr) : sub(expr) {}

    static constexpr bool matrix = is_2d<sub_type>; ///< INdicates if the sub type is a matrix or not

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        if constexpr (matrix) {
            size_t i_i = i / dim<1>(sub);
            size_t i_j = i % dim<1>(sub);
            return sub[i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j)];
        } else {
            return sub[etl::size(sub) - i - 1];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        if constexpr (matrix) {
            size_t i_i = i / dim<1>(sub);
            size_t i_j = i % dim<1>(sub);
            return sub.read_flat(i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j));
        } else {
            return sub.read_flat(etl::size(sub) - i - 1);
        }
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(size_t i) const {
        return sub(etl::size(sub) - 1 - i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return sub(i, columns(sub) - 1 - j);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename V>
    void visit(V&& visitor) const {
        sub.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure sub value
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Display the transformer on the given stream
     * \param os The output stream
     * \param transformer The transformer to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const hflip_transformer& transformer) {
        return os << "hflip(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that flips a matrix vertically
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct vflip_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend etl_traits<vflip_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit vflip_transformer(sub_type expr) : sub(expr) {}

    static constexpr bool matrix = is_2d<sub_type>; ///< Indicates if the sub type is a 2D matrix or not

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        if constexpr (matrix) {
            size_t i_i = i / dim<1>(sub);
            size_t i_j = i % dim<1>(sub);
            return sub[(dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j];
        } else {
            return sub[i];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        if constexpr (matrix) {
            size_t i_i = i / dim<1>(sub);
            size_t i_j = i % dim<1>(sub);
            return sub.read_flat((dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j);
        } else {
            return sub.read_flat(i);
        }
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(size_t i) const {
        return sub(i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return sub(rows(sub) - 1 - i, j);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename V>
    void visit(V&& visitor) const {
        sub.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure sub value
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Display the transformer on the given stream
     * \param os The output stream
     * \param transformer The transformer to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const vflip_transformer& transformer) {
        return os << "vflip(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that flips a matrix vertically and horizontally.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct fflip_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend etl_traits<fflip_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit fflip_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        if (dimensions(sub) == 1) {
            return sub[i];
        } else {
            return sub[etl::size(sub) - i - 1];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        if (dimensions(sub) == 1) {
            return sub.read_flat(i);
        } else {
            return sub.read_flat(etl::size(sub) - i - 1);
        }
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(size_t i) const {
        return sub(i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return sub(rows(sub) - 1 - i, columns(sub) - 1 - j);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename V>
    void visit(V&& visitor) const {
        sub.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure sub value
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Display the transformer on the given stream
     * \param os The output stream
     * \param transformer The transformer to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const fflip_transformer& transformer) {
        return os << "fflip(" << transformer.sub << ")";
    }
};

/*!
 * \brief Specialization for forwarding everything to the sub expression
 */
template <typename T>
struct etl_traits<
    T,
    std::enable_if_t<
           cpp::is_specialization_of_v<etl::hflip_transformer, std::decay_t<T>>
        || cpp::is_specialization_of_v<etl::vflip_transformer, std::decay_t<T>>
        || cpp::is_specialization_of_v<etl::fflip_transformer, std::decay_t<T>>
        || cpp::is_specialization_of_v<etl::one_if_max_sub_transformer, std::decay_t<T>>>> {
    using expr_t     = T;                                  ///< The expression type
    using sub_expr_t = std::decay_t<typename T::sub_type>; ///< The sub expression type
    using value_type = value_t<sub_expr_t>;                ///< The value type

    static constexpr bool  is_etl         = true;                                  ///< Indicates if the type is an ETL expression
    static constexpr bool  is_transformer = true;                                  ///< Indicates if the type is a transformer
    static constexpr bool  is_view        = false;                                 ///< Indicates if the type is a view
    static constexpr bool  is_magic_view  = false;                                 ///< Indicates if the type is a magic view
    static constexpr bool  is_fast        = etl_traits<sub_expr_t>::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool  is_linear      = false;                                 ///< Indicates if the expression is linear
    static constexpr bool  is_thread_safe = true;                                  ///< Indicates if the expression is thread safe
    static constexpr bool  is_value       = false;                                 ///< Indicates if the expression is of value type
    static constexpr bool  is_direct      = false;                                 ///< Indicates if the expression has direct memory access
    static constexpr bool  is_generator   = false;                                 ///< Indicates if the expression is a generated
    static constexpr bool  is_padded      = false;                                 ///< Indicates if the expression is padded
    static constexpr bool  is_aligned     = false;                                 ///< Indicates if the expression is padded
    static constexpr bool  is_temporary   = etl_traits<sub_expr_t>::is_temporary;  ///< Indicates if the expression needs an evaluator visitor
    static constexpr bool  gpu_computable = T::gpu_computable;                     ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order  = etl_traits<sub_expr_t>::storage_order; ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

} //end of namespace etl
