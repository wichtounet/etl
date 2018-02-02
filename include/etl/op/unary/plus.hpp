//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Unary operation computing the plus operation
 * \tparam T The type of value
 */
template <typename T>
struct plus_unary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

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
    template <typename E>
    static constexpr bool gpu_computable = cuda_enabled && !is_scalar<E>;

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) noexcept {
        return +x;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param x The vector on which to operate
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& x) noexcept {
        return x;
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

        return force_temporary_gpu_dim_only(t1);
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        return smart_gpu_compute(x, y);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "+";
    }
};

} //end of namespace etl
