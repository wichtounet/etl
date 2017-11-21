//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/relu_der_out.hpp"

namespace etl {

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
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::relu_der_out(etl::size(x), &alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

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

} //end of namespace etl
