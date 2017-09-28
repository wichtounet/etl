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
 * \brief Unary operation applying the max between the value and a scalar
 * \tparam T the type of value
 * \tparam S the type of scalar
 */
template <typename T, typename S>
struct max_scalar_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool linear      = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

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
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_smax)
            || (is_double_precision_t<T> && impl::egblas::has_dmax)
            || (is_complex_single_t<T> && impl::egblas::has_cmax)
            || (is_complex_double_t<T> && impl::egblas::has_zmax);

    S s; ///< The scalar value

    /*!
     * \brief Construct a new max_scalar_op with the given value
     * \param s The scalar value
     */
    explicit max_scalar_op(S s)
            : s(s) {}

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    constexpr T apply(const T& x) const noexcept {
        return std::max(x, s);
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    vec_type<V> load(const vec_type<V>& lhs) const noexcept {
        return V::max(lhs, V::set(s));
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X>
    auto gpu_compute(const X& x) const noexcept {
        decltype(auto) t1 = smart_gpu_compute(x);

        auto t2 = force_temporary_gpu(t1);

#ifdef ETL_CUDA
        auto s_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(s_gpu.get(), &s, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::max(etl::size(x), &alpha, s_gpu.get(), 0, t2.gpu_memory(), 1);
#endif

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

#ifdef ETL_CUDA
        auto s_gpu = impl::cuda::cuda_allocate_only<T>(1);
        cuda_check(cudaMemcpy(s_gpu.get(), &s, 1 * sizeof(T), cudaMemcpyHostToDevice));

        T alpha(1.0);
        impl::egblas::max(etl::size(x), &alpha, s_gpu.get(), 0, y.gpu_memory(), 1);
#else
        cpp_unused(y);
#endif

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
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
