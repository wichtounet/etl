//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/cudnn/sigmoid.hpp"
#include "etl/impl/egblas/sigmoid.hpp"

namespace etl {

/*!
 * \brief Unary operation computing the logistic sigmoid
 * \tparam T The type of value
 */
template <typename T>
struct sigmoid_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable =
        (V == vector_mode_t::SSE3 && !is_complex_t<T>) || (V == vector_mode_t::AVX && !is_complex_t<T>) || (intel_compiler && !is_complex_t<T>);

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = is_floating<E>&& cudnn_enabled;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

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
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        auto n = etl::size(x);

        if (n < 8 * 1024 * 1024 && is_single_precision<Y> && impl::egblas::has_ssigmoid) {
            impl::egblas::sigmoid(n, 1, t1.gpu_memory(), 1, t2.gpu_memory(), 1);
        } else if (n < 1024 * 1024 && is_double_precision<Y> && impl::egblas::has_dsigmoid) {
            impl::egblas::sigmoid(n, 1, t1.gpu_memory(), 1, t2.gpu_memory(), 1);
        } else {
            impl::cudnn::sigmoid(t1, t2);
        }

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
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        auto n = etl::size(x);

        if (n < 8 * 1024 * 1024 && is_single_precision<Y> && impl::egblas::has_ssigmoid) {
            impl::egblas::sigmoid(n, 1, t1.gpu_memory(), 1, y.gpu_memory(), 1);

            y.validate_gpu();
            y.invalidate_cpu();
        } else if (n < 1024 * 1024 && is_double_precision<Y> && impl::egblas::has_dsigmoid) {
            impl::egblas::sigmoid(n, 1, t1.gpu_memory(), 1, y.gpu_memory(), 1);

            y.validate_gpu();
            y.invalidate_cpu();
        } else {
            impl::cudnn::sigmoid(t1, y);
        }

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
 * \brief Unary operation computing a fast sigmoid approximation
 * \tparam T The type of value
 */
template <typename T>
struct fast_sigmoid_unary_op {
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

} //end of namespace etl
