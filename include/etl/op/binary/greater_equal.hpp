//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/greater_equal.hpp"

namespace etl {

/*!
 * \brief Binary operator for element greater than or equal comparison
 */
template <typename T>
struct greater_equal_binary_op {
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
    static constexpr bool gpu_computable = (all_single_precision<L, R> && impl::egblas::has_sgreater_equal)
                                           || (all_double_precision<L, R> && impl::egblas::has_dgreater_equal)
                                           || (all_complex_single_precision<L, R> && impl::egblas::has_cgreater_equal)
                                           || (all_complex_double_precision<L, R> && impl::egblas::has_zgreater_equal);

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
        return lhs >= rhs;
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

        constexpr size_t inca = gpu_inc<decltype(x)>;
        constexpr size_t incb = gpu_inc<decltype(y)>;

        auto t3 = force_temporary_gpu_dim_only_t<bool>(t1);

        impl::egblas::greater_equal(etl::size(yy), t1.gpu_memory(), inca, t2.gpu_memory(), incb, t3.gpu_memory(), 1);

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

        constexpr size_t inca = gpu_inc<decltype(x)>;
        constexpr size_t incb = gpu_inc<decltype(y)>;

        impl::egblas::greater_equal(etl::size(yy), t1.gpu_memory(), inca, t2.gpu_memory(), incb, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return ">";
    }
};

} //end of namespace etl
