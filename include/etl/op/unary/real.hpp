//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/real.hpp"

namespace etl {

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
    static constexpr bool gpu_computable =
               (is_complex_single_t<T> && impl::egblas::has_creal)
            || (is_complex_double_t<T> && impl::egblas::has_zreal);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr typename T::value_type apply(const T& x) noexcept {
        return get_real(x);
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

        auto t2 = etl::force_temporary_gpu_dim_only_t<typename T::value_type>(t1);

        typename T::value_type alpha(1.0);
        impl::egblas::real(etl::size(x), &alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

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
        // Note: Cannot use select here since x and y don't have the same type
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        typename T::value_type alpha(1.0);
        impl::egblas::real(etl::size(x), &alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "real";
    }
};

} //end of namespace etl
