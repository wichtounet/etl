//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Transform (dynamic) that returns only the maximum elements from the
 * right dimensions.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct argmax_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<argmax_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit argmax_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        return max_index(sub(i));
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return max_index(sub(i));
    }

    /*!
     * \brief Returns the value at the given position (i, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t i, Sizes... /*sizes*/) const {
        return max_index(sub(i));
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
    friend std::ostream& operator<<(std::ostream& os, const argmax_transformer& transformer) {
        return os << "argmax(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that returns only the maximum elements from the
 * right dimensions.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct argmin_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<argmin_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit argmin_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        return min_index(sub(i));
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return min_index(sub(i));
    }

    /*!
     * \brief Returns the value at the given position (i, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t i, Sizes... /*sizes*/) const {
        return min_index(sub(i));
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
    friend std::ostream& operator<<(std::ostream& os, const argmin_transformer& transformer) {
        return os << "argmin(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that sums the expression from the right, effectively removing the right dimension.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct sum_r_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<sum_r_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit sum_r_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        return sum(sub(i));
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return sum(sub(i));
    }

    /*!
     * \brief Returns the value at the given position (i, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t i, Sizes... /*sizes*/) const {
        return sum(sub(i));
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
    friend std::ostream& operator<<(std::ostream& os, const sum_r_transformer& transformer) {
        return os << "sum_r(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that averages the expression from the right, effectively removing the right dimension.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct mean_r_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<mean_r_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit mean_r_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](size_t i) const {
        return mean(sub(i));
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return mean(sub(i));
    }

    /*!
     * \brief Returns the value at the given position (i, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t i, Sizes... /*sizes*/) const {
        return mean(sub(i));
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
    friend std::ostream& operator<<(std::ostream& os, const mean_r_transformer& transformer) {
        return os << "mean_r(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that sums the expression from the left, effectively removing the left dimension.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct sum_l_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<sum_l_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit sum_l_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param j The index
     * \return the value at the given index.
     */
    value_type operator[](size_t j) const {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub[j + i * (etl::size(sub) / dim<0>(sub))];
        }

        return m;
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub.read_flat(j + i * (etl::size(sub) / dim<0>(sub)));
        }

        return m;
    }

    /*!
     * \brief Access to the value at the given (j, sizes...) position
     * \param j The first index
     * \param sizes The remaining indices
     * \return The value at the position (j, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t j, Sizes... sizes) const {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub(i, j, sizes...);
        }

        return m;
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
    friend std::ostream& operator<<(std::ostream& os, const sum_l_transformer& transformer) {
        return os << "sum_l(" << transformer.sub << ")";
    }
};

/*!
 * \brief Transform (dynamic) that averages the expression from the left, effectively removing the left dimension.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct mean_l_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    friend struct etl_traits<mean_l_transformer>;

    static constexpr bool gpu_computable = false;

private:
    sub_type sub; ///< The subexpression

public:
    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit mean_l_transformer(sub_type expr) : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param j The index
     * \return the value at the given index.
     */
    value_type operator[](size_t j) const {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub[j + i * (etl::size(sub) / dim<0>(sub))];
        }

        return m / dim<0>(sub);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub.read_flat(j + i * (etl::size(sub) / dim<0>(sub)));
        }

        return m / dim<0>(sub);
    }

    /*!
     * \brief Access to the value at the given (j, sizes...) position
     * \param j The first index
     * \param sizes The remaining indices
     * \return The value at the position (j, sizes...)
     */
    template <typename... Sizes>
    value_type operator()(size_t j, Sizes... sizes) const {
        value_type m = 0.0;

        for (size_t i = 0; i < dim<0>(sub); ++i) {
            m += sub(i, j, sizes...);
        }

        return m / dim<0>(sub);
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
    friend std::ostream& operator<<(std::ostream& os, const mean_l_transformer& transformer) {
        return os << "mean_l(" << transformer.sub << ")";
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<
    T,
    std::enable_if_t<
        cpp::is_specialization_of_v<
            etl::argmax_transformer,
            std::decay_t<
                T>> || cpp::is_specialization_of_v<etl::argmin_transformer, std::decay_t<T>> || cpp::is_specialization_of_v<etl::sum_r_transformer, std::decay_t<T>> || cpp::is_specialization_of_v<etl::mean_r_transformer, std::decay_t<T>>>> {
    using expr_t     = T;                                                ///< The expression type
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>; ///< The sub expression type
    using value_type = value_t<sub_expr_t>;                              ///< The value type of the expression

    static constexpr bool is_etl         = true;                                   ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = true;                                   ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                  ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                  ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = etl_traits<sub_expr_t>::is_fast;        ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                  ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = etl_traits<sub_expr_t>::is_thread_safe; ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                  ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = false;                                  ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                  ///< Indicates if the expression is a generated
    static constexpr bool is_padded      = false;                                  ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                                  ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = etl_traits<sub_expr_t>::is_temporary;   ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool gpu_computable = false;                                  ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = etl_traits<sub_expr_t>::storage_order;  ///< The expression storage order

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
        return etl::dim<0>(v.sub);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, [[maybe_unused]] size_t d) {
        return etl::dim<0>(v.sub);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T,
                  std::enable_if_t<cpp::is_specialization_of_v<etl::sum_l_transformer,
                                                               std::decay_t<T>> || cpp::is_specialization_of_v<etl::mean_l_transformer, std::decay_t<T>>>> {
    using expr_t     = T;                                                ///< The expression type
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>; ///< The sub expression type
    using value_type = value_t<sub_expr_t>;                              ///< The value type of the expression

    static constexpr bool is_etl         = true;                                   ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = true;                                   ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                  ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                  ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = etl_traits<sub_expr_t>::is_fast;        ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                  ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = etl_traits<sub_expr_t>::is_thread_safe; ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                  ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = false;                                  ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                  ///< Indicates if the expression is a generated
    static constexpr bool is_padded      = false;                                  ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                                  ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = etl_traits<sub_expr_t>::is_temporary;   ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool gpu_computable = false;                                  ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = etl_traits<sub_expr_t>::storage_order;  ///< The expression storage order

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
        return etl::size(v.sub) / etl::dim<0>(v.sub);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return etl::dim(v.sub, d + 1);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

} //end of namespace etl
