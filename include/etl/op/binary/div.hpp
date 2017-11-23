//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Binary operator for scalar division
 */
template <typename T>
struct div_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the given vector mode
     * \tparam V The vector mode
     *
     * Note: Integer division is not yet supported
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = is_floating_t<T> || (is_complex_t<T> && V != vector_mode_t::AVX512);

    /*!
     * \brief Indicates if the operator can be computd on GPU
     */
    template<typename L, typename R>
    static constexpr bool gpu_computable =
            (
                    (!is_scalar<L> && !is_scalar<R>)
                &&  egblas_enabled
                &&  (
                        (is_single_precision_t<T> && impl::egblas::has_saxdy)
                    ||  (is_double_precision_t<T> && impl::egblas::has_daxdy)
                    ||  (is_complex_single_t<T> && impl::egblas::has_caxdy)
                    ||  (is_complex_double_t<T> && impl::egblas::has_zaxdy))
            )
        ||  (
                    (!is_scalar<L> && is_scalar<R>)
                &&  cublas_enabled
            )
        ||  (
                    (is_scalar<L> && !is_scalar<R>)
                && (
                        (is_single_precision_t<T> && impl::egblas::has_scalar_sdiv)
                    ||  (is_double_precision_t<T> && impl::egblas::has_scalar_ddiv)
                    ||  (is_complex_single_t<T> && impl::egblas::has_scalar_cdiv)
                    ||  (is_complex_double_t<T> && impl::egblas::has_scalar_zdiv))
            )
            ;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs / rhs;
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
        return V::div(lhs, rhs);
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
        cpp_unused(y);

        auto t3 = force_temporary_gpu_dim_only(y);
        gpu_compute(lhs, rhs, t3);
        return t3;
    }

#ifdef ETL_EGBLAS_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && !is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        smart_gpu_compute(lhs, y);

        decltype(auto) t2 = smart_gpu_compute_hint(rhs, y);

        value_t<L> alpha(1);

        impl::egblas::axdy(etl::size(y), alpha, t2.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

#endif

#ifdef ETL_CUBLAS_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        auto s = T(1) / rhs.value;

        smart_gpu_compute(lhs, y);

        decltype(auto) handle = impl::cublas::start_cublas();
        impl::cublas::cublas_scal(handle.get(), etl::size(y), s, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

#endif

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_scalar<L> && !is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        auto s = lhs.value;

        smart_gpu_compute(rhs, y);

        impl::egblas::scalar_div(s, y.gpu_memory(), etl::size(y), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "/";
    }
};

} //end of namespace etl
