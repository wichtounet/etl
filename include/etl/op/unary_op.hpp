//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the unary operators for the unary expression
 *
 * A unary operator is a simple class with a static function apply that
 * computes its result. If the operator is vectorizable, it also contains a
 * static function load that computes the result for several operands at a
 * time.
 */

#pragma once

#include <functional>
#include <ctime>

#include "etl/math.hpp"
#include "etl/impl/cudnn/sigmoid.hpp"
#include "etl/impl/egblas/sqrt.hpp"
#include "etl/impl/egblas/log.hpp"
#include "etl/impl/egblas/exp.hpp"
#include "etl/impl/egblas/relu_der_out.hpp"

namespace etl {

/*!
 * \brief Unary operation taking the absolute value
 * \tparam T The type of value
 */
template <typename T>
struct abs_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::abs(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::max(x, V::sub(V::template zero<T>(), x));
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "abs";
    }
};

/*!
 * \brief Unary operation rounding down the value
 * \tparam T The type of value
 */
template <typename T>
struct floor_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::floor(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "floor";
    }
};

/*!
 * \brief Unary operation rounding up the value
 * \tparam T The type of value
 */
template <typename T>
struct ceil_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::ceil(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ceil";
    }
};

/*!
 * \brief Unary operation taking the logarithmic value
 * \tparam T The type of value
 */
template <typename T>
struct log_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable =
                (V == vector_mode_t::SSE3 && is_single_precision_t<T>)
            ||  (V == vector_mode_t::AVX && is_single_precision_t<T>)
            ||  (intel_compiler && !is_complex_t<T>);

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_slog)
            || (is_double_precision_t<T> && impl::egblas::has_dlog)
            || (is_complex_single_t<T> && impl::egblas::has_clog)
            || (is_complex_double_t<T> && impl::egblas::has_zlog);

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return std::log(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::log(x);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::log(etl::size(x), &alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        // TODO Check this!
        if /* constexpr */ (is_dma<X>){
            decltype(auto) t1 = smart_gpu_compute(x);

            T alpha(1.0);
            impl::egblas::log(etl::size(x), &alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);
        } else {
            smart_gpu_compute(x, y);

            T alpha(1.0);
            impl::egblas::log(etl::size(x), &alpha, y.gpu_memory(), 1, y.gpu_memory(), 1);
        }

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "log";
    }
};

/*!
 * \brief Unary operation taking the square root value
 * \tparam T The type of value
 */
template <typename T>
struct sqrt_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = is_floating_t<T> && egblas_enabled;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return std::sqrt(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::sqrt(x);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);
        gpu_compute(t1, t2);
        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        T alpha(1.0);
        impl::egblas::sqrt(etl::size(x), &alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sqrt";
    }
};

/*!
 * \brief Unary operation taking the inverse square root value
 * \tparam T The type of value
 */
template <typename T>
struct invsqrt_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return T(1) / std::sqrt(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "invsqrt";
    }
};

/*!
 * \brief Unary operation taking the cubic root value
 * \tparam T The type of value
 */
template <typename T>
struct cbrt_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return std::cbrt(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cbrt";
    }
};

/*!
 * \brief Unary operation taking the inverse cubic root value
 * \tparam T The type of value
 */
template <typename T>
struct invcbrt_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return T(1) / std::cbrt(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "invcbrt";
    }
};

/*!
 * \brief Unary operation computing the exponential
 * \tparam T The type of value
 */
template <typename T>
struct exp_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable =
            (V == vector_mode_t::SSE3 && !is_complex_t<T>)
        ||  (V == vector_mode_t::AVX && !is_complex_t<T>)
        ||  (intel_compiler && !is_complex_t<T>);

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_sexp)
            || (is_double_precision_t<T> && impl::egblas::has_dexp)
            || (is_complex_single_t<T> && impl::egblas::has_cexp)
            || (is_complex_double_t<T> && impl::egblas::has_zexp);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return std::exp(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::exp(x);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::exp(etl::size(x), &alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        // TODO Check this!
        if /* constexpr */ (is_dma<X>){
            decltype(auto) t1 = smart_gpu_compute(x);

            T alpha(1.0);
            impl::egblas::exp(etl::size(x), &alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);
        } else {
            smart_gpu_compute(x, y);

            T alpha(1.0);
            impl::egblas::exp(etl::size(x), &alpha, y.gpu_memory(), 1, y.gpu_memory(), 1);
        }

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "exp";
    }
};

/*!
 * \brief Unary operation computing the sign
 * \tparam T The type of value
 */
template <typename T>
struct sign_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return math::sign(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sign";
    }
};

/*!
 * \brief Unary operation computing the logistic sigmoid
 * \tparam T The type of value
 */
template <typename T>
struct sigmoid_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable =
            (V == vector_mode_t::SSE3 && !is_complex_t<T>)
        ||  (V == vector_mode_t::AVX && !is_complex_t<T>)
        ||  (intel_compiler && !is_complex_t<T>);

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = is_floating<E> && cudnn_enabled;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return math::logistic_sigmoid(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        auto one = V::set(T(1));

        auto t1 = V::minus(x);
        auto t2 = V::exp(t1);
        auto t3 = V::add(one, t2);
        return V::div(one, t3);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);
        impl::cudnn::sigmoid(t1, t2);
        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        impl::cudnn::sigmoid(t1, y);

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sigmoid";
    }
};

/*!
 * \brief Unary operation computing the softplus
 * \tparam T The type of value
 */
template <typename T>
struct softplus_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return math::softplus(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "softplus";
    }
};

/*!
 * \brief Unary operation computing the minus operation
 * \tparam T The type of value
 */
template <typename T>
struct minus_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return -x;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::minus(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "-";
    }
};

/*!
 * \brief Unary operation computing the plus operation
 * \tparam T The type of value
 */
template <typename T>
struct plus_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return +x;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return x;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "+";
    }
};

/*!
 * \brief Unary operation computing a fast sigmoid approximation
 * \tparam T The type of value
 */
template <typename T>
struct fast_sigmoid_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param v The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& v) {
        auto x = 0.5 * v;

        T z;
        if (x >= 0) {
            if (x < 1.7) {
                z = (1.5 * x / (1 + x));
            } else if (x < 3) {
                z = (0.935409070603099 + 0.0458812946797165 * (x - 1.7));
            } else {
                z = 0.99505475368673;
            }
        } else {
            auto xx = -x;
            if (xx < 1.7) {
                z = (1.5 * xx / (1 + xx));
            } else if (xx < 3) {
                z = (0.935409070603099 + 0.0458812946797165 * (xx - 1.7));
            } else {
                z = 0.99505475368673;
            }
            z = -z;
        }

        return 0.5 * (z + 1.0);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "fast_sigmoid";
    }
};

/*!
 * \brief Unary operation computing the tangent
 * \tparam T The type of value
 */
template <typename T>
struct tan_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::tan(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "tan";
    }
};

/*!
 * \brief Unary operation computing the cosinus
 * \tparam T The type of value
 */
template <typename T>
struct cos_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = (V == vector_mode_t::SSE3 || V == vector_mode_t::AVX) && is_single_precision_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::cos(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::cos(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cos";
    }
};

/*!
 * \brief Unary operation computing the sinus
 * \tparam T The type of value
 */
template <typename T>
struct sin_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = (V == vector_mode_t::SSE3 || V == vector_mode_t::AVX) && is_single_precision_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::sin(x);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::sin(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sin";
    }
};

/*!
 * \brief Unary operation computing the hyperbolic tangent
 * \tparam T The type of value
 */
template <typename T>
struct tanh_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::tanh(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "tanh";
    }
};

/*!
 * \brief Unary operation computing the hyperbolic cosinus
 * \tparam T The type of value
 */
template <typename T>
struct cosh_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::cosh(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cosh";
    }
};

/*!
 * \brief Unary operation computing the hyperbolic sinus
 * \tparam T The type of value
 */
template <typename T>
struct sinh_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return std::sinh(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sinh";
    }
};

/*!
 * \brief Unary operation extracting the real part of a complex number
 * \tparam T The type of value
 */
template <typename T>
struct real_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr typename T::value_type apply(const T& x) noexcept {
        return get_real(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "real";
    }
};

/*!
 * \brief Unary operation extracting the imag part of a complex number
 * \tparam T The type of value
 */
template <typename T>
struct imag_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr typename T::value_type apply(const T& x) noexcept {
        return get_imag(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "imag";
    }
};

/*!
 * \brief Unary operation computing the conjugate value of complex number
 * \tparam T The type of value
 */
template <typename T>
struct conj_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return get_conj(x);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "conj";
    }
};

/*!
 * \brief Unary operation computing the RELU operation
 * \tparam T The type of value
 */
template <typename T>
struct relu_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = is_floating<E> && cudnn_enabled;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        return std::max(x, T(0.0));
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::max(x, V::set(T(0.0)));
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);
        impl::cudnn::relu(t1, t2);
        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        impl::cudnn::relu(t1, y);

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "relu";
    }
};

/*!
 * \brief Unary operation computing the derivate of the RELU operation
 * \tparam T The type of value
 */
template <typename T>
struct relu_derivative_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = is_floating_t<T> && impl::egblas::has_srelu_der_out && impl::egblas::has_drelu_der_out;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        return x > 0.0 ? 1.0 : 0.0;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return V::round_up(V::min(V::set(T(1.0)), x));
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    static auto gpu_compute(const X& x) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu_dim_only(t1);
        gpu_compute(t1, t2);
        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        T alpha(1.0);
        impl::egblas::relu_der_out(etl::size(x), &alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "relu_derivative";
    }
};

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename T>
struct bernoulli_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a Bernoulli distribution
 * \tparam T The type of value
 */
template <typename G, typename T>
struct bernoulli_unary_g_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not

private:

    G& rand_engine; ///< The custom random engine

public:

    /*!
     * \brief Construct a new bernoulli_unary_g_op
     */
    explicit bernoulli_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 1.0 : 0.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli";
    }
};

/*!
 * \brief Unary operation sampling with a reverse Bernoulli distribution
 * \tparam T The type of value
 */
template <typename T>
struct reverse_bernoulli_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 0.0 : 1.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

/*!
 * \brief Unary operation sampling with a reverse Bernoulli distribution
 * \tparam T The type of value
 */
template <typename G, typename T>
struct reverse_bernoulli_unary_g_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

private:

    G& rand_engine; ///< The custom random engine

public:

    /*!
     * \brief Construct a new reverse_bernoulli_unary_g_op
     */
    explicit reverse_bernoulli_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        return x > distribution(rand_engine) ? 0.0 : 1.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "bernoulli_reverse";
    }
};

/*!
 * \brief Unary operation applying an uniform noise (0.0, 1.0(
 * \tparam T The type of value
 */
template <typename T>
struct uniform_noise_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::uniform_real_distribution<double> real_distribution(0.0, 1.0);

        return x + real_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

/*!
 * \brief Unary operation applying an uniform noise (0.0, 1.0(
 * \tparam T The type of value
 */
template <typename G, typename T>
struct uniform_noise_unary_g_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

private:

    G& rand_engine; ///< The custom random engine

public:

    /*!
     * \brief Construct a new uniform_noise_unary_g_op
     */
    explicit uniform_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::uniform_real_distribution<double> real_distribution(0.0, 1.0);

        return x + real_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "uniform_noise";
    }
};

/*!
 * \brief Unary operation applying a normal noise
 * \tparam T The type of value
 */
template <typename T>
struct normal_noise_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);

        return x + normal_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "normal_noise";
    }
};

/*!
 * \brief Unary operation applying a normal noise
 * \tparam T The type of value
 */
template <typename G, typename T>
struct normal_noise_unary_g_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

private:

    G& rand_engine; ///< The custom random engine

public:

    /*!
     * \brief Construct a new normal_noise_unary_g_op
     */
    explicit normal_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::normal_distribution<double> normal_distribution(0.0, 1.0);

        return x + normal_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "normal_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise
 * \tparam T The type of value
 */
template <typename T>
struct logistic_noise_unary_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static T apply(const T& x) {
        static random_engine rand_engine(std::time(nullptr));

        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

/*!
 * \brief Unary operation applying a logistic noise
 * \tparam T The type of value
 */
template <typename G, typename T>
struct logistic_noise_unary_g_op {
    static constexpr bool linear = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = false;  ///< Indicates if the operator is thread safe or not

private:

    G& rand_engine; ///< The custom random engine

public:

    /*!
     * \brief Construct a new logistic_noise_unary_g_op
     */
    explicit logistic_noise_unary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    T apply(const T& x) const {
        std::normal_distribution<double> noise_distribution(0.0, math::logistic_sigmoid(x));

        return x + noise_distribution(rand_engine);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "logistic_noise";
    }
};

/*!
 * \brief Unary operation applying the min between the value and a scalar
 * \tparam T the type of value
 * \tparam S the type of scalar
 */
template <typename T, typename S>
struct min_scalar_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    S s; ///< The scalar value

    /*!
     * \brief Construct a new min_scalar_op with the given value
     * \param s The scalar value
     */
    explicit min_scalar_op(S s)
            : s(s) {}

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    constexpr T apply(const T& x) const noexcept {
        return std::min(x, s);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::min(lhs, V::set(s));
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "min";
    }
};

/*!
 * \brief Unary operation applying the max between the value and a scalar
 * \tparam T the type of value
 * \tparam S the type of scalar
 */
template <typename T, typename S>
struct max_scalar_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    S s; ///< The scalar value

    /*!
     * \brief Construct a new max_scalar_op with the given value
     * \param s The scalar value
     */
    explicit max_scalar_op(S s)
            : s(s) {}

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    constexpr T apply(const T& x) const noexcept {
        return std::max(x, s);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::max(lhs, V::set(s));
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "max";
    }
};

/*!
 * \brief Unary operation that clips all values between two scalars
 * \tparam T the type of value
 * \tparam S the type of scalar
 */
template <typename T, typename S>
struct clip_scalar_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = intel_compiler && !is_complex_t<T>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;

    S min; ///< The minimum for clipping
    S max; ///< The maximum for clipping

    /*!
     * \brief Builds a new operator
     * \param min The minimum for clipping
     * \param max The maximum for clipping
     */
    clip_scalar_op(S min, S max)
            : min(min), max(max) {}

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    constexpr T apply(const T& x) const noexcept {
        return std::min(std::max(x, min), max);
    }

#ifdef __INTEL_COMPILER
    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::min(V::max(lhs, V::set(min)), V::set(max));
    }
#endif

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "clip";
    }
};

} //end of namespace etl
