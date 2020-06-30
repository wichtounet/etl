//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/or.hpp"

namespace etl {

/*!
 * \brief Binary operator for elementwise logical OR computation
 */
template <typename T>
struct logical_or_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = false; ///< Indicates if the description must be printed as function

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
    template <typename L, typename R>
    static constexpr bool gpu_computable = impl::egblas::has_bor;

    /*!
     * \brief Estimate the complexity of operator
     * \return An estimation of the complexity of the operator
     */
    static constexpr int complexity() {
        return 1;
    }

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs || rhs;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y, typename YY>
    static auto gpu_compute_hint(const X& x, const Y& y, YY& yy) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, yy);
        decltype(auto) t2 = smart_gpu_compute_hint(y, yy);

        auto t3 = force_temporary_gpu_dim_only_t<bool>(t1);

        impl::egblas::logical_or(etl::size(yy), t1.gpu_memory(), 1, t2.gpu_memory(), 1, t3.gpu_memory(), 1);

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
        decltype(auto) t1 = smart_gpu_compute_hint(x, yy);
        decltype(auto) t2 = smart_gpu_compute_hint(y, yy);

        impl::egblas::logical_or(etl::size(yy), t1.gpu_memory(), 1, t2.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "||";
    }
};

} //end of namespace etl
