//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Binary operator for sigmoid derivative
 */
template <typename T>
struct sigmoid_derivative_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = false; ///< Indicates if the description must be printed as function

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
    template <typename L, typename R>
    static constexpr bool gpu_computable = cudnn_enabled;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs * (1.0 - lhs) * rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static ETL_STRONG_INLINE(vec_type<V>) load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        auto one = V::set(T(1.0));
        auto t1  = V::sub(one, lhs);
        auto t2  = V::mul(lhs, t1);

        return V::mul(t2, rhs);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y>
    static auto gpu_compute_hint(const L& lhs, const R& rhs, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(lhs, y);
        decltype(auto) t2 = smart_gpu_compute_hint(rhs, y);
        decltype(auto) t3 = force_temporary_gpu_dim_only(t2);

        impl::cudnn::sigmoid_backward(t1, t2, t3);

        return t3;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(lhs, y);
        decltype(auto) t2 = smart_gpu_compute_hint(rhs, y);

        impl::cudnn::sigmoid_backward(t1, t2, y);

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "sigmoid_back";
    }
};

} //end of namespace etl
