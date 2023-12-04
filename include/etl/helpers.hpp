//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Return the number of dimensions of the given ETL expression
 * \param expr The expression to get the number of dimensions for
 * \return The number of dimensions of the given expression.
 */
template <typename E>
constexpr size_t dimensions([[maybe_unused]] const E& expr) noexcept {
    return etl_traits<E>::dimensions();
}

/*!
 * \brief Return the number of dimensions of the given ETL type
 * \tparam E The expression type to get the number of dimensions for
 * \return The number of dimensions of the given type.
 */
template <typename E>
constexpr size_t dimensions() noexcept {
    return decay_traits<E>::dimensions();
}

/*!
 * \brief Return the complexity of the expression
 * \param expr The expression to get the complexity for
 * \return The estimated complexity of the given expression.
 */
template <typename E>
constexpr int complexity([[maybe_unused]] const E& expr) noexcept {
    return etl_traits<E>::complexity();
}

/*!
 * \brief Return the complexity of the expression
 * \tparam E The expression type to get the complexity for
 * \return The estimated complexity of the given type.
 */
template <typename E>
constexpr int complexity() noexcept {
    return decay_traits<E>::complexity();
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <dyn_expr E>
size_t rows(const E& expr) {
    return etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <fast_expr E>
constexpr size_t rows(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::template dim<0>();
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <dyn_matrix_c E>
size_t columns(const E& expr) {
    return etl_traits<E>::dim(expr, 1);
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <fast_matrix_c E>
constexpr size_t columns(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::template dim<1>();
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <dyn_expr E>
size_t size(const E& expr) {
    return etl_traits<E>::size(expr);
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <fast_expr E>
constexpr size_t size(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::size();
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <dyn_matrix_c E>
size_t subsize(const E& expr) {
    return etl_traits<E>::size(expr) / etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <fast_matrix_c E>
constexpr size_t subsize(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <size_t D, dyn_expr E>
size_t dim(const E& e) noexcept {
    return etl_traits<E>::dim(e, D);
}

/*!
 * \brief Return the d dimension of e
 * \param e The expression to get the dimensions from
 * \param d The dimension to get
 * \return the dth dimension of e
 */
template <etl_expr E>
size_t dim(const E& e, size_t d) noexcept {
    return etl_traits<E>::dim(e, d);
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <size_t D, fast_expr E>
constexpr size_t dim(const E& e) noexcept {
    return (void)e, etl_traits<E>::template dim<D>();
}

/*!
 * \brief Return the D dimension of E
 * \return the Dth dimension of E
 */
template <size_t D, fast_expr E>
constexpr size_t dim() noexcept {
    return decay_traits<E>::template dim<D>();
}

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions_impl;

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions_impl<E> requires(generator<E>) : std::integral_constant<size_t, std::numeric_limits<size_t>::max()> {};

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions_impl<E> requires(!generator<E>) : std::integral_constant<size_t, etl_traits<E>::dimensions()> {};

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E, typename Enable = void>
constexpr size_t safe_dimensions = safe_dimensions_impl<E>::value;

/*!
 * \brief Convert a flat index into a 2D index
 * \param sub The matrix expression
 * \param i The flat index
 * \return a pair of indices for the equivalent 2D index
 */
template <typename E>
constexpr std::pair<size_t, size_t> index_to_2d(E&& sub, size_t i) {
    return decay_traits<E>::storage_order == order::RowMajor ? std::make_pair(i / dim<1>(sub), i % dim<1>(sub))
                                                             : std::make_pair(i % dim<0>(sub), i / dim<0>(sub));
}

/*!
 * \brief Returns the row stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the row stride of the given ETL matrix expression
 */
template <etl_2d E>
size_t row_stride(E&& expr) {
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<1>(expr) : 1;
}

/*!
 * \brief Returns the column stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the column stride of the given ETL matrix expression
 */
template <etl_2d E>
size_t col_stride(E&& expr) {
    return decay_traits<E>::storage_order == order::RowMajor ? 1 : etl::dim<0>(expr);
}

/*!
 * \brief Returns the minor stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the minor stride of the given ETL matrix expression
 */
template <etl_2d E>
size_t minor_stride(E&& expr) {
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<0>(expr) : etl::dim<1>(expr);
}

/*!
 * \brief Returns the major stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the major stride of the given ETL matrix expression
 */
template <etl_2d E>
size_t major_stride(E&& expr) {
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<1>(expr) : etl::dim<0>(expr);
}

/*!
 * \brief Test if two memory ranges overlap.
 * \param a_begin The beginning of the first range
 * \param a_end The end (exclusive) of the first range
 * \param b_begin The beginning of the second range
 * \param b_end The end (exclusive) of the second range
 *
 * The ranges must be ordered (begin <= end). This function is optimized so that only two comparisons are performed.
 *
 * \return true if the two ranges overlap, false otherwise
 */
template <typename P1, typename P2>
bool memory_alias(const P1* a_begin, const P1* a_end, const P2* b_begin, const P2* b_end) {
    cpp_assert(a_begin <= a_end, "memory_alias works on ordered ranges");
    cpp_assert(b_begin <= b_end, "memory_alias works on ordered ranges");

    return reinterpret_cast<uintptr_t>(a_begin) < reinterpret_cast<uintptr_t>(b_end)
           && reinterpret_cast<uintptr_t>(a_end) > reinterpret_cast<uintptr_t>(b_begin);
}

/*!
 * \brief Ensure that the CPU is up to date.
 *
 * \param expr The expression
 */
template <typename E>
void safe_ensure_cpu_up_to_date(E&& expr) {
    expr.ensure_cpu_up_to_date();
}

/*!
 * \brief Indicates if the CPU memory is up to date. If the expression does
 * not have direct memory access, return true
 *
 * \param expr The expression
 */
template <typename E>
bool safe_is_cpu_up_to_date(E&& expr) {
    if constexpr (is_dma<E>) {
        return expr.is_cpu_up_to_date();
    } else {
        return true;
    }
}

/*!
 * \brief Indicates if the GPU memory is up to date. If the expression does
 * not have direct memory access, return false
 *
 * \param expr The expression
 */
template <typename E>
bool safe_is_gpu_up_to_date(E&& expr) {
    if constexpr (is_dma<E>) {
        return expr.is_gpu_up_to_date();
    } else {
        return false;
    }
}

/*!
 * \brief Smart forwarding for a temporary expression.
 *
 * This is guaranteed to produce a DMA expression in the most
 * efficient way possible.
 *
 * \param expr the Expresison from which to create a temporary.
 *
 * \return a direct expression
 */
template <typename E>
decltype(auto) smart_forward(E& expr) {
    if constexpr (is_temporary_expr<E>) {
        return force_temporary(expr);
    } else {
        return make_temporary(expr);
    }
}

/*!
 * \brief Smart forwarding for a temporary expression that will be
 * computed in GPU.
 *
 * This is guaranteed to produce a DMA expression in the most
 * efficient way possible.
 *
 * \param expr the Expresison from which to create a temporary.
 *
 * \return a direct GPU-able expression
 */
template <typename E>
decltype(auto) smart_forward_gpu(E& expr) {
    if constexpr (is_temporary_expr<E>) {
        if constexpr (E::gpu_computable) {
            return force_temporary_gpu(expr);
        } else {
            return force_temporary(expr);
        }
    } else {
        return make_temporary(expr);
    }
}

// Unary smart_gpu_compute

/*!
 * \brief Compute the expression into a representation that is GPU up to date.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed.
 *
 * \param expr The expression that must be evaluated
 *
 * \return A gpu-computed expression reprensenting the results of the input expr
 */
template <typename E, typename Y>
decltype(auto) smart_gpu_compute_hint(E& expr, Y& y) {
    if constexpr (is_temporary_expr<E>) {
        if constexpr (E::gpu_computable) {
            return force_temporary_gpu(expr);
        } else {
            auto t = force_temporary(expr);
            t.ensure_gpu_up_to_date();
            return t;
        }
    } else {
        return expr.gpu_compute_hint(y);
    }
}

// Binary smart_gpu_compute

/*!
 * \brief Compute the expression into a representation that is GPU up to date
 * and store this representation in y.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed. Ideally, the result will be directly computed inside y.
 *
 * \param x The expression that must be evaluated
 * \param y The expression into which store the GPU result of x
 *
 * \return y
 */
template <typename X, typename Y>
decltype(auto) smart_gpu_compute(X& x, Y& y) {
    if constexpr (is_temporary_expr<X>) {
        if constexpr (X::gpu_computable) {
            return y = x;
        } else {
            auto t = force_temporary(x);
            t.ensure_gpu_up_to_date();
            y = t;
            return y;
        }
    } else {
        if constexpr (is_dma<X>) {
            x.ensure_gpu_up_to_date();
            y.ensure_gpu_allocated();
            y.gpu_copy_from(x.gpu_memory());
            return y;
        } else {
            return x.gpu_compute(y);
        }
    }
}

// Select version

/*!
 * \brief Compute the expression into a representation that is GPU up to date
 * and possibly store this representation in y.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed. Ideally, the result will be directly computed inside y.
 *
 * \param x The expression that must be evaluated
 * \param y The expression into which store the GPU result of x
 *
 * \return either a temporary of the result of x (possibly x) or y
 */
template <typename X, typename Y>
decltype(auto) select_smart_gpu_compute(X& x, Y& y) {
    if constexpr (should_gpu_compute_direct<X>) {
        return smart_gpu_compute_hint(x, y);
    } else {
        return smart_gpu_compute(x, y);
    }
}

/*!
 * \brief The space between two elements in GPU for the given type.
 *
 * This is 1 for every type but for scalar which are 0.
 */
template <typename T>
constexpr size_t gpu_inc = is_scalar<T> ? 0 : 1;

} //end of namespace etl
