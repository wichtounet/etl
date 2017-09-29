//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/max.hpp"

namespace etl {

/*!
 * \brief Binary operator for scalar maximum
 */
template <typename T, typename E>
struct max_binary_op {
    static constexpr bool linear    = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = true; ///< Indicates if the description must be printed as function

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
    template<typename L, typename R>
    static constexpr bool gpu_computable =
               (all_single_precision<L,R> && impl::egblas::has_smax)
            || (all_double_precision<L,R> && impl::egblas::has_dmax)
            || (all_complex_single_precision<L,R> && impl::egblas::has_cmax)
            || (all_complex_double_precision<L,R> && impl::egblas::has_zmax);

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::max(x, value);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::max(lhs, rhs);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    static auto gpu_compute(const X& x, const Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);
        decltype(auto) t2 = smart_gpu_compute(y);

        auto t3 = force_temporary_gpu(t1);

        T alpha(1);
        impl::egblas::max(etl::size(x), &alpha, t2.gpu_memory(), 1, t3.gpu_memory(), 1);

        return t3;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y, typename YY>
    static YY& gpu_compute(const X& x, const Y& y, YY& yy) noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);
        smart_gpu_compute(y, yy);

        T alpha(1);
        impl::egblas::max(etl::size(x), &alpha, t1.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "max";
    }
};

} //end of namespace etl
