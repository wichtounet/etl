//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/pow.hpp"
#include "etl/impl/egblas/pow_yx.hpp"

namespace etl {

/*!
 * \brief Binary operator for scalar power
 */
template <typename T, typename E>
struct pow_binary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable =
        (V == vector_mode_t::SSE3 && is_single_precision_t<T>) || (V == vector_mode_t::AVX && is_single_precision_t<T>) || (intel_compiler && !is_complex_t<T>);

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename L, typename R>
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_spow_yx) || (is_double_precision_t<T> && impl::egblas::has_dpow_yx)
                                           || (is_complex_single_t<T> && impl::egblas::has_cpow_yx) || (is_complex_double_t<T> && impl::egblas::has_zpow_yx);

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::pow(x, value);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The left hand side vector
     * \param y The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static ETL_STRONG_INLINE(vec_type<V>) load(const vec_type<V>& x, const vec_type<V>& y) noexcept {
        // Use pow(x, y) = exp(y * log(x))
        auto t1 = V::log(x);
        auto t2 = V::mul(y, t1);
        return V::exp(t2);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y, typename YY>
    static auto gpu_compute_hint([[maybe_unused]] const X& x, const Y& y, YY& yy) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, yy);

        auto t2 = force_temporary_gpu(t1);

#ifdef ETL_CUDA
        T power_cpu(y.value);
        auto power_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(power_gpu.get(), &power_cpu, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::pow_yx(etl::size(yy), alpha, power_gpu.get(), 0, t2.gpu_memory(), 1);
#endif

        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y, typename YY>
    static YY& gpu_compute(const X& x, [[maybe_unused]] const Y& y, YY& yy) noexcept {
        smart_gpu_compute(x, yy);

#ifdef ETL_CUDA
        T power_cpu(y.value);
        auto power_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(power_gpu.get(), &power_cpu, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::pow_yx(etl::size(yy), alpha, power_gpu.get(), 0, yy.gpu_memory(), 1);
#endif

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "pow";
    }
};

/*!
 * \brief Binary operator for scalar power with stable precision.
 */
template <typename T, typename E>
struct precise_pow_binary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true; ///< Indicates if the description must be printed as function

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
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::pow(x, value);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "pow_precise";
    }
};

/*!
 * \brief Binary operator for scalar power with an integer as the exponent.
 */
template <typename T, typename E>
struct integer_pow_binary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true; ///< Indicates if the description must be printed as function

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
    static constexpr bool gpu_computable = (is_single_precision_t<T> && impl::egblas::has_spow_yx) || (is_double_precision_t<T> && impl::egblas::has_dpow_yx)
                                           || (is_complex_single_t<T> && impl::egblas::has_cpow_yx) || (is_complex_double_t<T> && impl::egblas::has_zpow_yx);

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        T r(1);

        for (size_t i = 0; i < value; ++i) {
            r *= x;
        }

        return r;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y, typename YY>
    static auto gpu_compute_hint(const X& x, [[maybe_unused]] const Y& y, YY& yy) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, yy);

        auto t2 = force_temporary_gpu(t1);

#ifdef ETL_CUDA
        T power_cpu(y.value);
        auto power_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(power_gpu.get(), &power_cpu, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::pow_yx(etl::size(yy), alpha, power_gpu.get(), 0, t2.gpu_memory(), 1);
#endif

        return t2;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y, typename YY>
    static YY& gpu_compute(const X& x, [[maybe_unused]] const Y& y, YY& yy) noexcept {
        smart_gpu_compute(x, yy);

#ifdef ETL_CUDA
        T power_cpu(y.value);
        auto power_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(power_gpu.get(), &power_cpu, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::pow_yx(etl::size(yy), alpha, power_gpu.get(), 0, yy.gpu_memory(), 1);
#endif

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "pow";
    }
};

} //end of namespace etl
