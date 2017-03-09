//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/tmp.hpp"

//The transformers
#include "etl/op/flip_transformers.hpp"
#include "etl/op/rep_transformers.hpp"
#include "etl/op/reduc_transformers.hpp"

namespace etl {

/*!
 * \brief Transform (dynamic) that transposes a matrix.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct transpose_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    sub_type sub; ///< The subexpression

    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit transpose_transformer(sub_type expr)
            : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub[i];
        } else {
            if (decay_traits<sub_type>::storage_order == order::RowMajor) {
                std::size_t ii, jj;
                std::tie(ii, jj) = index_to_2d(sub, i);
                return sub(jj, ii);
            } else {
                return sub(i / dim<1>(sub), i % dim<1>(sub));
            }
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub.read_flat(i);
        } else {
            if (decay_traits<sub_type>::storage_order == order::RowMajor) {
                std::size_t ii, jj;
                std::tie(ii, jj) = index_to_2d(sub, i);
                return sub(jj, ii);
            } else {
                return sub(i / dim<1>(sub), i % dim<1>(sub));
            }
        }
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(j, i);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
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
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Transform that applies a lazy matrix multiplication to two matrices
 * \tparam L The left type on which the transformer is applied
 * \tparam R The right type on which the transformer is applied
 */
template <typename L, typename R>
struct mm_mul_transformer {
    using left_type  = L;          ///< The left side type
    using right_type = R;          ///< The right side type
    using value_type = value_t<L>; ///< The value type

    left_type left;   ///< The left expression
    right_type right; ///< The right expression

    /*!
     * \brief Construct a new transformer around the given expressions
     * \param left The left hand side sub expression
     * \param right The right hand side sub expression
     */
    mm_mul_transformer(left_type left, right_type right)
            : left(left), right(right) {
        check_mmul_sizes(left, right);
    }

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](std::size_t i) const {
        std::size_t i_i, i_j;
        std::tie(i_i, i_j) = index_to_2d(left, i);
        return operator()(i_i, i_j);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        std::size_t i_i, i_j;
        std::tie(i_i, i_j) = index_to_2d(left, i);
        return operator()(i_i, i_j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        value_type c = 0;

        for (std::size_t k = 0; k < columns(left); k++) {
            c += left(i, k) * right(k, j);
        }

        return c;
    }

    /*!
     * \brief Returns the left value on which the transformer is working.
     * \return A reference to the left value on which the transformer is working.
     */
    left_type& lhs() {
        return left;
    }

    /*!
     * \brief Returns the left value on which the transformer is working.
     * \return A reference to the left value on which the transformer is working.
     */
    const left_type& lhs() const {
        return left;
    }

    /*!
     * \brief Returns the right value on which the transformer is working.
     * \return A reference to the right value on which the transformer is working.
     */
    right_type& rhs() {
        return right;
    }

    /*!
     * \brief Returns the right value on which the transformer is working.
     * \return A reference to the right value on which the transformer is working.
     */
    const right_type& rhs() const {
        return right;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return left.alias(rhs) || right.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        lhs().visit(visitor);
        rhs().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        lhs().visit(visitor);
        rhs().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        lhs().visit(visitor);
        visitor.need_value = true;
        rhs().visit(visitor);
        visitor.need_value = old_need_value;
    }

private:
    template <typename A, typename B, cpp_disable_if(all_fast<A, B>::value)>
    void check_mmul_sizes(const A& a, const B& b) {
        cpp_assert(
            dim<1>(a) == dim<0>(b) //interior dimensions
            ,
            "Invalid sizes for multiplication");
        cpp_unused(a);
        cpp_unused(b);
    }

    template <typename A, typename B, cpp_enable_if(all_fast<A, B>::value)>
    void check_mmul_sizes(const A& /*a*/, const B& /*b*/) {
        static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>() //interior dimensions
            ,
            "Invalid sizes for multiplication");
    }
};

/*!
 * \brief Transform that applies a convmtx transformation on a matrix.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct dyn_convmtx_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    static_assert(decay_traits<T>::dimensions() == 1, "convmtx can only be applied on vectors");

    sub_type sub;  ///< The subexpression
    std::size_t h; ///< The convmtx transformation size

    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     * \param h The convmtx transformation size
     */
    dyn_convmtx_transformer(sub_type expr, std::size_t h)
            : sub(expr), h(h) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](std::size_t i) const {
        if (decay_traits<sub_type>::storage_order == order::RowMajor) {
            std::size_t i_i = i / (etl::size(sub) + h - 1);
            std::size_t i_j = i % (etl::size(sub) + h - 1);
            return operator()(i_i, i_j);
        } else {
            std::size_t i_i = i % h;
            std::size_t i_j = i / h;
            return operator()(i_i, i_j);
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return operator[](i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        if (j < i) {
            return value_type(0);
        } else if (j >= etl::size(sub) + i) {
            return value_type(0);
        } else {
            return sub(j - i);
        }
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
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
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Transform that applies a convmtx2 transformation on a matrix.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct dyn_convmtx2_transformer {
    using sub_type   = T;          ///< The type on which the expression works
    using value_type = value_t<T>; ///< The type of valuie

    static_assert(decay_traits<T>::dimensions() == 2, "convmtx2 can only be applied on matrices");

    sub_type sub;               ///< The subexpression
    const std::size_t k1;       ///< The first dimension of the kernel
    const std::size_t k2;       ///< The second dimension of the kernel
    std::size_t i1;             ///< The first dimension of the input
    std::size_t i2;             ///< The second dimension of the input
    std::size_t inner_paddings; ///< The inner padding sum
    std::size_t inner_padding;  ///< The inner padding

    dyn_convmtx2_transformer(sub_type sub, std::size_t k1, std::size_t k2)
            : sub(sub), k1(k1), k2(k2) {
        i1 = etl::dim<0>(sub);
        i2 = etl::dim<1>(sub);

        std::size_t c_height = (i1 + k1 - 1) * (i2 + k2 - 1);
        std::size_t c_width  = k1 * k2;

        auto max_fill  = c_height - ((i1 + k1 - 1) * ((c_width - 1) / k1) + (c_width - 1) % k1);
        inner_paddings = max_fill - (i1 * i2);
        inner_padding  = inner_paddings / (i2 - 1);
    }

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / (k1 * k2);
        std::size_t i_j = i % (k1 * k2);
        return (*this)(i_i, i_j);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        std::size_t i_i = i / (k1 * k2);
        std::size_t i_j = i % (k1 * k2);
        return (*this)(i_i, i_j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        auto top_padding = (i1 + k1 - 1) * (j / k1) + j % k1;

        if (i < top_padding || i >= top_padding + (i1 * i2) + inner_paddings) {
            return value_type(0);
        } else {
            auto inner = i - top_padding;
            auto col   = inner % (i1 + inner_padding);
            auto block = inner / (i1 + inner_padding);

            if (col >= i1) {
                return value_type(0);
            } else {
                return sub(col, block);
            }
        }
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
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
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
    }
};

//TODO Make a temporary_expr out of this

/*!
 * \brief Compute the convolution matrix of sub into m for a kernel of size (k1,k2)
 * \param m The output matrix
 * \param sub The input matrix
 * \param k1 The first dimension of ther kernel
 * \param k2 The second dimension of ther kernel
 */
template <typename A, typename M>
void convmtx2_direct_t(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const std::size_t c_height = (i1 + k1 - 1) * (i2 + k2 - 1);
    const std::size_t c_width  = k1 * k2;

    const auto max_fill       = c_height - ((i1 + k1 - 1) * ((c_width - 1) / k1) + (c_width - 1) % k1);
    const auto inner_paddings = max_fill - (i1 * i2);
    const auto inner_padding  = inner_paddings / (i2 - 1);

    auto* __restrict mm = m.memory_start();
    auto* __restrict ss = sub.memory_start();

    std::fill(mm, mm + etl::size(m), 0.0);

    for (std::size_t j = 0; j < c_width; ++j) {
        std::size_t big_i = (i1 + k1 - 1) * (j / k1) + j % k1;

        for (std::size_t ii = 0; ii < etl::dim<1>(sub); ++ii) {
            for (std::size_t jj = 0; jj < etl::dim<0>(sub); ++jj) {
                mm[j * c_width + big_i] = ss[jj * i2 + ii];
                ++big_i;
            }
            big_i += inner_padding;
        }
    }
}

//TODO Adapt this to an expression

/*!
 * \brief Convert an image to a sequence of image columns to be multiplied by kernels of size (k1,k2)
 * \param m The output matrix
 * \param sub The input image
 * \param k1 The first dimension of ther kernel
 * \param k2 The second dimension of ther kernel
 */
template <typename A, typename M, cpp_disable_if(all_dma<A, M>::value)>
void im2col_direct(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const std::size_t m_width = (i1 - k1 + 1) * (i2 - k2 + 1);

    for (std::size_t b = 0; b < m_width; ++b) {
        auto s_i = b % (i1 - k1 + 1);
        auto s_j = b / (i1 - k1 + 1);

        for (std::size_t b_i = 0; b_i < k1; ++b_i) {
            for (std::size_t b_j = 0; b_j < k2; ++b_j) {
                m(b_j * k1 + b_i, b) = sub(s_i + b_i, s_j + b_j);
            }
        }
    }
}

// This is a direct memory version
// On gcc and clang, this is significantly faster

/*!
 * \brief Convert an image to a sequence of image columns to be multiplied by kernels of size (k1,k2)
 * \param m The output matrix
 * \param sub The input image
 * \param k1 The first dimension of ther kernel
 * \param k2 The second dimension of ther kernel
 */
template <typename A, typename M, cpp_enable_if(all_dma<A, M>::value)>
void im2col_direct(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const auto m_width = (i1 - k1 + 1) * (i2 - k2 + 1);

    const auto mm = m.memory_start();
    const auto ss = sub.memory_start();

    for (std::size_t b = 0; b < m_width; ++b) {
        auto s_i = b % (i1 - k1 + 1);
        auto s_j = b / (i1 - k1 + 1);

        for (std::size_t b_i = 0; b_i < k1; ++b_i) {
            for (std::size_t b_j = 0; b_j < k2; ++b_j) {
                mm[(b_j * k1 + b_i) * m_width + b] = ss[(s_i + b_i) * i2 + s_j + b_j];
            }
        }
    }
}

//im2col version without any need for transpose

/*!
 * \brief Convert an image to a sequence of image columns to be multiplied by kernels of size (k1,k2).
 *
 * This special version does not require any transposition when used.
 *
 * \param m The output matrix
 * \param sub The input image
 * \param k1 The first dimension of ther kernel
 * \param k2 The second dimension of ther kernel
 */
template <typename A, typename M>
void im2col_direct_tr(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    static_assert(all_dma<A, M>::value, "im2col_direct_tr has only been implemented for direct memory access");

    const std::size_t i1 = etl::dim<0>(sub);
    const std::size_t i2 = etl::dim<1>(sub);

    const auto height = i1 - k1 + 1;
    const auto width  = i2 - k2 + 1;

    const auto mm = m.memory_start();
    const auto ss = sub.memory_start();

    for (std::size_t c = 0; c < k1 * k2; ++c) {
        const std::size_t w_source = c % k2;
        const std::size_t h_source = (c / k2) % k1;
        const std::size_t c_source = c / (k1 * k2);

        for (std::size_t h = 0; h < height; ++h) {
            const std::size_t block_source = (c_source * i1 + h + h_source) * i2 + w_source;
            const std::size_t block_target = (c * height + h) * width;

            direct_copy_n(ss + block_source, mm + block_target, width);
        }
    }
}

/*!
 * \brief Convert a sequence of images to a sequence of image columns to be multiplied by kernels of size (k1,k2).
 *
 * This special version does not require any transposition when used.
 *
 * \param m The output matrix
 * \param sub The input image
 * \param k1 The first dimension of ther kernel
 * \param k2 The second dimension of ther kernel
 */
template <typename A, typename M>
void im2col_direct_tr_multi(M& m, A&& sub, std::size_t k1, std::size_t k2) {
    static_assert(all_dma<A, M>::value, "im2col_direct_tr has only been implemented for direct memory access");

    const auto N  = etl::dim<0>(sub);
    const auto i1 = etl::dim<1>(sub);
    const auto i2 = etl::dim<2>(sub);

    const auto height = i1 - k1 + 1;
    const auto width  = i2 - k2 + 1;

    const auto mm = m.memory_start();
    const auto ss = sub.memory_start();

    for (std::size_t w = 0; w < k1 * k2; ++w) {
        const auto w_source = w % k2;
        const auto h_source = (w / k2) % k1;
        const auto c_source = w / (k1 * k2);

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t h = 0; h < height; ++h) {
                const auto block_source = ((c_source * i1 + h + h_source) * i2 + w_source) + (i) * (i1 * i2);
                const auto block_target = (w * N + i) * (height * width) + h * width;

                etl::direct_copy_n(ss + block_source, mm + block_target, width);
            }
        }
    }
}

/*!
 * \brief Specialization for tranpose_transformer
 */
template <typename T>
struct etl_traits<transpose_transformer<T>> {
    using expr_t     = etl::transpose_transformer<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;               ///< The sub expression type
    using value_type = value_t<sub_expr_t>;           ///< The value type

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = true;                                            ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                 ///< Indicates if the expression is fast
    static constexpr bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generated
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicaes if the expression needs an evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, 1 - d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<1 - D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for mm_mul_transformer
 */
template <typename LE, typename RE>
struct etl_traits<mm_mul_transformer<LE, RE>> {
    using expr_t       = etl::mm_mul_transformer<LE, RE>; ///< The expression type
    using left_expr_t  = std::decay_t<LE>;                ///< The left hand side expression type
    using right_expr_t = std::decay_t<RE>;                ///< The right hand side expression type
    using value_type   = value_t<left_expr_t>;            ///< The value type

    static constexpr bool is_etl         = true;                                                                  ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = true;                                                                  ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                                                 ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                                                 ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = etl_traits<left_expr_t>::is_fast && etl_traits<right_expr_t>::is_fast; ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                                                 ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                                                 ///< Indicates if the expression is of value type
    static constexpr bool is_direct       = false;                                                                 ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator = false;                                                                   ///< Indicates if the expression is a generated
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor =
        etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor; ///< Indicates if the expression needs an evaluator visitor
    static constexpr order storage_order = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order;

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return dim(v, 0) * dim(v, 1);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return etl::dim(v.left, 0);
        } else {
            cpp_assert(d == 1, "Only 2D mmul are supported");

            return etl::dim(v.right, 1);
        }
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<left_expr_t>::template dim<0>() * etl_traits<right_expr_t>::template dim<1>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(D < 2, "Only 2D mmul are supported");

        return D == 0 ? etl_traits<left_expr_t>::template dim<0>() : etl_traits<right_expr_t>::template dim<1>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx_transformer<E>> {
    using expr_t     = etl::dyn_convmtx_transformer<E>; ///< The expression type
    using sub_expr_t = std::decay_t<E>;                 ///< The sub expression type
    using value_type = value_t<sub_expr_t>;             ///< The value type

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = true;                                            ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;                                           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generated
    static constexpr bool is_padded               = false;                                           ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = false;                                           ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicaes if the expression needs an evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v.h * (etl::size(v.sub) + v.h - 1);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return v.h;
        } else {
            return etl::size(v.sub) + v.h - 1;
        }
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx2_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx2_transformer<E>> {
    using expr_t     = etl::dyn_convmtx2_transformer<E>; ///< The expression type
    using sub_expr_t = std::decay_t<E>;                  ///< The sub expression type
    using value_type = value_t<sub_expr_t>;

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = true;                                            ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generated
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicaes if the expression needs an evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        auto c_height = (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        auto c_width  = v.k1 * v.k2;
        return c_height * c_width;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        } else {
            return v.k1 * v.k2;
        }
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for forwarding everything to the sub expression
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::hflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::vflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::fflip_transformer, std::decay_t<T>>>::value>>
                         {
    using expr_t     = T;                                  ///< The expression type
    using sub_expr_t = std::decay_t<typename T::sub_type>; ///< The sub expression type
    using value_type = value_t<sub_expr_t>;

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = true;                                            ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                 ///< Indicates if the expression is fast
    static constexpr bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generated
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicaes if the expression needs an evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Display the transformer on the given stream
 * \param os The output stream
 * \param transformer The transformer to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const transpose_transformer<T>& transformer) {
    return os << "T(" << transformer.sub << ")";
}

/*!
 * \brief Display the transformer on the given stream
 * \param os The output stream
 * \param transformer The transformer to print
 * \return the output stream
 */
template <typename L, typename R>
std::ostream& operator<<(std::ostream& os, const mm_mul_transformer<L, R>& transformer) {
    return os << "mm_mul(" << transformer.left << "," << transformer.right << ")";
}

} //end of namespace etl
