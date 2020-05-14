//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/clip_value.hpp"

namespace etl {

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
    using vec_type = typename V::template vec_type<T>;

    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_sclip_value)
                                           || (is_double_precision_t<T> && impl::egblas::has_dclip_value)
                                           || (is_complex_single_t<T> && impl::egblas::has_cclip_value)
                                           || (is_complex_double_t<T> && impl::egblas::has_zclip_value);

    S min; ///< The minimum for clipping
    S max; ///< The maximum for clipping

    /*!
     * \brief Builds a new operator
     * \param min The minimum for clipping
     * \param max The maximum for clipping
     */
    clip_scalar_op(S min, S max) : min(min), max(max) {}

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
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    auto gpu_compute_hint(const X& x, Y& y) const noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu(t1);
        impl::egblas::clip_value(etl::size(y), T(1), min, max, t2.gpu_memory(), 1);
        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    Y& gpu_compute(const X& x, Y& y) const noexcept {
        smart_gpu_compute(x, y);

        impl::egblas::clip_value(etl::size(y), T(1), min, max, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "clip";
    }
};

} //end of namespace etl
