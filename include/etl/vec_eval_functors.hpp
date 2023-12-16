//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file eval_functors.hpp
 * \brief Contains the vectorized functors used by the evaluator to
 * perform its actions.
 */

#pragma once

namespace etl::detail {

/*!
 * \brief Common base for vectorized functors
 */
template <vector_mode_t V>
struct vectorized_base {
    using vect_impl = typename get_vector_impl<V>::type; ///< The vectorization type

    /*!
     * \brief Load a vector from lhs at position i
     * \param i The index where to start loading from
     * \return a vector from lhs starting at position i
     */
    template <typename T>
    static inline auto load(T&& x, size_t i) {
        return x.template load<vect_impl>(i);
    }
};

/*!
 * \brief Functor for vectorized assign
 *
 * The result is computed in a vectorized fashion with several
 * operations per cycle and written directly to the memory of lhs.
 */
template <vector_mode_t V>
struct VectorizedAssign : vectorized_base<V> {
    using base_t = vectorized_base<V>; ///< The base type
    using base_t::load;
    using vect_impl = typename base_t::vect_impl; ///< The vectorization type

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        using IT = typename get_intrinsic_traits<V>::template type<value_t<R_Expr>>;

        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>;

        const size_t last = remainder ? prev_multiple(N, IT::size) : N;

        size_t i = 0;

        // 0. If possible and interesting, use streaming stores

        if constexpr (streaming) {
            if (N > stream_threshold / (sizeof(value_t<L_Expr>) * 3) && !rhs.alias(lhs)) {
                for (; i < last; i += IT::size) {
                    lhs.template stream<vect_impl>(load(rhs, i), i);
                }

                for (; remainder && i < N; ++i) {
                    lhs_mem[i] = rhs[i];
                }

                return;
            }
        }

        // 1. In the default case, simple unrolled vectorization

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(load(rhs, i + 0 * IT::size), i + 0 * IT::size);
            lhs.template store<vect_impl>(load(rhs, i + 1 * IT::size), i + 1 * IT::size);
            lhs.template store<vect_impl>(load(rhs, i + 2 * IT::size), i + 2 * IT::size);
            lhs.template store<vect_impl>(load(rhs, i + 3 * IT::size), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(load(rhs, i), i);
        }

        for (; remainder && i < N; ++i) {
            lhs_mem[i] = rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign add
 */
template <vector_mode_t V>
struct VectorizedAssignAdd : vectorized_base<V> {
    using base_t = vectorized_base<V>; ///< The base type
    using base_t::load;
    using vect_impl = typename base_t::vect_impl; ///< The vectorization type

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        using IT = typename get_intrinsic_traits<V>::template type<value_t<R_Expr>>;

        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>;

        const size_t last = remainder ? prev_multiple(N, IT::size) : N;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::add(load(lhs, i + 0 * IT::size), load(rhs, i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(load(lhs, i + 1 * IT::size), load(rhs, i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(load(lhs, i + 2 * IT::size), load(rhs, i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(load(lhs, i + 3 * IT::size), load(rhs, i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::add(load(lhs, i), load(rhs, i)), i);
        }

        for (; remainder && i < N; ++i) {
            lhs_mem[i] += rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign sub
 */
template <vector_mode_t V>
struct VectorizedAssignSub : vectorized_base<V> {
    using base_t = vectorized_base<V>; ///< The base type
    using base_t::load;
    using vect_impl = typename base_t::vect_impl; ///< The vectorization type

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        using IT = typename get_intrinsic_traits<V>::template type<value_t<R_Expr>>;

        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>;

        const size_t last = remainder ? prev_multiple(N, IT::size) : N;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::sub(load(lhs, i + 0 * IT::size), load(rhs, i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(load(lhs, i + 1 * IT::size), load(rhs, i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(load(lhs, i + 2 * IT::size), load(rhs, i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(load(lhs, i + 3 * IT::size), load(rhs, i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::sub(load(lhs, i), load(rhs, i)), i);
        }

        for (; remainder && i < N; ++i) {
            lhs_mem[i] -= rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign mul
 */
template <vector_mode_t V>
struct VectorizedAssignMul : vectorized_base<V> {
    using base_t = vectorized_base<V>; ///< The base type
    using base_t::load;
    using vect_impl = typename base_t::vect_impl; ///< The vectorization type

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        using IT = typename get_intrinsic_traits<V>::template type<value_t<R_Expr>>;

        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>;

        const size_t last = remainder ? prev_multiple(N, IT::size) : N;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::mul(load(lhs, i + 0 * IT::size), load(rhs, i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(load(lhs, i + 1 * IT::size), load(rhs, i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(load(lhs, i + 2 * IT::size), load(rhs, i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(load(lhs, i + 3 * IT::size), load(rhs, i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::mul(load(lhs, i), load(rhs, i)), i);
        }

        for (; remainder && i < N; ++i) {
            lhs_mem[i] *= rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign div
 */
template <vector_mode_t V>
struct VectorizedAssignDiv : vectorized_base<V> {
    using base_t = vectorized_base<V>; ///< The base type
    using base_t::load;
    using vect_impl = typename base_t::vect_impl; ///< The vectorization type

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        using IT = typename get_intrinsic_traits<V>::template type<value_t<R_Expr>>;

        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>;

        const size_t last = remainder ? prev_multiple(N, IT::size) : N;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::div(load(lhs, i + 0 * IT::size), load(rhs, i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(load(lhs, i + 1 * IT::size), load(rhs, i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(load(lhs, i + 2 * IT::size), load(rhs, i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(load(lhs, i + 3 * IT::size), load(rhs, i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::div(load(lhs, i), load(rhs, i)), i);
        }

        for (; remainder && i < N; ++i) {
            lhs_mem[i] /= rhs[i];
        }
    }
};

} //end of namespace etl::detail
