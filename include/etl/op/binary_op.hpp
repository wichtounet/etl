//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains binary operators
 */

#pragma once

#include <functional>
#include <ctime>

#include "etl/math.hpp"

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"
#include "etl/impl/cublas/axpy.hpp"

#endif

namespace etl {

/*!
 * \brief Binary operator for scalar addition
 */
template <typename T>
struct plus_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    static constexpr bool gpu_computable = cublas_enabled; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear         = true;           ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe    = true;           ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func      = false;          ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs + rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static ETL_STRONG_INLINE(vec_type<V>) load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::add(lhs, rhs);
    }

#ifdef ETL_CUBLAS_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R>
    static auto gpu_compute(const L& lhs, const R& rhs) noexcept {
        decltype(auto) t1 = lhs.gpu_compute();
        decltype(auto) t2 = rhs.gpu_compute();

        t1.ensure_gpu_up_to_date();
        t2.ensure_gpu_up_to_date();

        auto t3 = force_temporary(t2);
        t3.ensure_gpu_up_to_date();

        decltype(auto) handle = impl::cublas::start_cublas();

        value_t<L> alpha(1);

        impl::cublas::cublas_axpy(handle.get(), size(lhs), &alpha, t1.gpu_memory(), 1, t3.gpu_memory(), 1);

        t3.validate_gpu();
        t3.invalidate_cpu();

        return t3;
    }

#endif

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "+";
    }
};

/*!
 * \brief Binary operator for scalar subtraction
 */
template <typename T>
struct minus_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    static constexpr bool gpu_computable = cublas_enabled; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear         = true;           ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe    = true;           ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func      = false;          ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs - rhs;
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
        return V::sub(lhs, rhs);
    }

#ifdef ETL_CUBLAS_MODE

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R>
    static auto gpu_compute(const L& lhs, const R& rhs) noexcept {
        decltype(auto) t1 = lhs.gpu_compute();
        decltype(auto) t2 = rhs.gpu_compute();

        t1.ensure_gpu_up_to_date();
        t2.ensure_gpu_up_to_date();

        auto t3 = force_temporary(t1);
        t3.ensure_gpu_up_to_date();

        decltype(auto) handle = impl::cublas::start_cublas();

        value_t<L> alpha(-1);

        impl::cublas::cublas_axpy(handle.get(), size(lhs), &alpha, t2.gpu_memory(), 1, t3.gpu_memory(), 1);

        t3.validate_gpu();
        t3.invalidate_cpu();

        return t3;
    }

#endif

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "-";
    }
};

/*!
 * \brief Binary operator for scalar multiplication
 */
template <typename T>
struct mul_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = V == vector_mode_t::AVX512 ? !is_complex_t<T> : true;

    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear    = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs * rhs;
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
        return V::mul(lhs, rhs);
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "*";
    }
};

/*!
 * \brief Binary operator for scalar division
 */
template <typename T>
struct div_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Indicates if the expression is vectorizable using the given vector mode
     * \tparam V The vector mode
     *
     * Note: Integer division is not yet supported
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = is_floating_t<T> || (is_complex_t<T> && V != vector_mode_t::AVX512);

    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear    = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = false; ///< Indicates if the description must be printed as function

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
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "/";
    }
};

/*!
 * \brief Binary operator for scalar modulo
 */
template <typename T>
struct mod_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear    = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs % rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "%";
    }
};

/*!
 * \brief Binary operator for element wise equality
 */
template <typename T>
struct equal_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs == rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "==";
    }
};

/*!
 * \brief Binary operator for element wise inequality
 */
template <typename T>
struct not_equal_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs != rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "!=";
    }
};

/*!
 * \brief Binary operator for element less than comparison
 */
template <typename T>
struct less_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs < rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "<";
    }
};

/*!
 * \brief Binary operator for element less than or equal comparison
 */
template <typename T>
struct less_equal_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs <= rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "<";
    }
};

/*!
 * \brief Binary operator for element greater than comparison
 */
template <typename T>
struct greater_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs > rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return ">";
    }
};

/*!
 * \brief Binary operator for element greater than or equal comparison
 */
template <typename T>
struct greater_equal_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs >= rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return ">";
    }
};

/*!
 * \brief Binary operator for elementwise logical and computation
 */
template <typename T>
struct logical_and_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs && rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "&&";
    }
};

/*!
 * \brief Binary operator for elementwise logical OR computation
 */
template <typename T>
struct logical_or_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs || rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "||";
    }
};

/*!
 * \brief Binary operator for elementwise logical XOR computation
 */
template <typename T>
struct logical_xor_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr bool apply(const T& lhs, const T& rhs) noexcept {
        return lhs != rhs;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "^";
    }
};

/*!
 * \brief Binary operator for ranged noise generation
 *
 * This operator adds noise from N(0,1) to x. If x is 0 or the rhs
 * value, x is not modified.
 */
template <typename G, typename T, typename E>
struct ranged_noise_binary_g_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true;  ///< Indicates if the description must be printed as function

    G& rand_engine; ///< The random engine

    /*!
     * \brief Construct a new ranged_noise_binary_g_op
     */
    ranged_noise_binary_g_op(G& rand_engine) : rand_engine(rand_engine) {
        //Nothing else to init
    }

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    T apply(const T& x, E value) {
        std::normal_distribution<double> normal_distribution(0.0, 1.0);
        auto noise = std::bind(normal_distribution, rand_engine);

        if (x == 0.0 || x == value) {
            return x;
        } else {
            return x + noise();
        }
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

/*!
 * \brief Binary operator for ranged noise generation
 *
 * This operator adds noise from N(0,1) to x. If x is 0 or the rhs
 * value, x is not modified.
 */
template <typename T, typename E>
struct ranged_noise_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = false; ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = true;  ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static T apply(const T& x, E value) {
        static random_engine rand_engine(std::time(nullptr));
        static std::normal_distribution<double> normal_distribution(0.0, 1.0);
        static auto noise = std::bind(normal_distribution, rand_engine);

        if (x == 0.0 || x == value) {
            return x;
        } else {
            return x + noise();
        }
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "ranged_noise";
    }
};

/*!
 * \brief Binary operator for scalar maximum
 */
template <typename T, typename E>
struct max_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "max";
    }
};

/*!
 * \brief Binary operator for scalar minimum
 */
template <typename T, typename E>
struct min_binary_op {
    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
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
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return std::min(x, value);
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
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "min";
    }
};

/*!
 * \brief Binary operator for scalar power
 */
template <typename T, typename E>
struct pow_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear    = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

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
        return "pow";
    }
};

/*!
 * \brief Binary operator to get 1.0 if x equals to rhs value, 0 otherwise
 */
template <typename T, typename E>
struct one_if_binary_op {
    static constexpr bool gpu_computable = false; ///< Indicates if the operator can be computed on GPU
    static constexpr bool linear    = true; ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func = true; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param x The left hand side value on which to apply the operator
     * \param value The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& x, E value) noexcept {
        return x == value ? 1.0 : 0.0;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "one_if";
    }
};

} //end of namespace etl
