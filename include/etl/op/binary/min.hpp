//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/min.hpp"

namespace etl {

/*!
 * \brief Binary operator for scalar minimum
 */
template <typename LT, typename RT>
struct min_binary_op {
    static constexpr bool linear    = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = !is_complex_t<LT>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template<typename L, typename R>
    static constexpr bool gpu_computable =
               (all_single_precision<L,R> && impl::egblas::has_smin3)
            || (all_double_precision<L,R> && impl::egblas::has_dmin3)
            || (all_complex_single_precision<L,R> && impl::egblas::has_cmin3)
            || (all_complex_double_precision<L,R> && impl::egblas::has_zmin3);

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<LT>;


    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr LT apply(const LT& x, const RT& value) noexcept {
        if(x < value){
            return x;
        } else {
            return value;
        }
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
        return V::min(lhs, rhs);
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

        auto t3 = force_temporary_gpu_dim_only(t1);

        LT alpha(1);
        impl::egblas::min(etl::size(yy), alpha, t1.gpu_memory(), inca, t2.gpu_memory(), incb, t3.gpu_memory(), 1);

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

        LT alpha(1);
        impl::egblas::min(etl::size(yy), alpha, t1.gpu_memory(), inca, t2.gpu_memory(), incb, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "min";
    }
};

} //end of namespace etl
