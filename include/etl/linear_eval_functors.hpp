//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file linear_eval_functors.hpp
 * \brief Contains the linear functors used by the evaluator to
 * perform its actions.
 */

#pragma once

namespace etl::detail {

/*!
 * \brief Functor for simple assign
 *
 * The result is written to lhs with operator[] and read from rhs
 * with read_flat
 */
struct Assign {
    /*!
     * \brief Assign rhs to lhs
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        size_t i = 0;

        if constexpr (unroll_normal_loops) {
            const size_t iend = N & size_t(-4);

            for (; i < iend; i += 4) {
                lhs_mem[i]     = rhs.read_flat(i);
                lhs_mem[i + 1] = rhs.read_flat(i + 1);
                lhs_mem[i + 2] = rhs.read_flat(i + 2);
                lhs_mem[i + 3] = rhs.read_flat(i + 3);
            }
        }

        for (; i < N; ++i) {
            lhs_mem[i] = rhs.read_flat(i);
        }
    }
};

/*!
 * \brief Functor for simple compound assign add
 */
struct AssignAdd {
    /*!
     * \brief Assign rhs to lhs
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        size_t i = 0;

        if constexpr (unroll_normal_loops) {
            const size_t iend = (N & size_t(-4));

            for (; i < iend; i += 4) {
                lhs_mem[i] += rhs[i];
                lhs_mem[i + 1] += rhs[i + 1];
                lhs_mem[i + 2] += rhs[i + 2];
                lhs_mem[i + 3] += rhs[i + 3];
            }
        }

        for (; i < N; ++i) {
            lhs_mem[i] += rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign sub
 */
struct AssignSub {
    /*!
     * \brief Assign rhs to lhs
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        size_t i = 0;

        if constexpr (unroll_normal_loops) {
            const size_t iend = (N & size_t(-4));

            for (; i < iend; i += 4) {
                lhs_mem[i] -= rhs[i];
                lhs_mem[i + 1] -= rhs[i + 1];
                lhs_mem[i + 2] -= rhs[i + 2];
                lhs_mem[i + 3] -= rhs[i + 3];
            }
        }

        for (; i < N; ++i) {
            lhs_mem[i] -= rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign mul
 */
struct AssignMul {
    /*!
     * \brief Assign rhs to lhs
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        size_t i = 0;

        if constexpr (unroll_normal_loops) {
            const size_t iend = (N & size_t(-4));

            for (; i < iend; i += 4) {
                lhs_mem[i] *= rhs[i];
                lhs_mem[i + 1] *= rhs[i + 1];
                lhs_mem[i + 2] *= rhs[i + 2];
                lhs_mem[i + 3] *= rhs[i + 3];
            }
        }

        for (; i < N; ++i) {
            lhs_mem[i] *= rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign div
 */
struct AssignDiv {
    /*!
     * \brief Assign rhs to lhs
     */
    template <typename L_Expr, typename R_Expr>
    static void apply(L_Expr&& lhs, R_Expr&& rhs) {
        const size_t N = etl::size(lhs);

        auto* lhs_mem = lhs.memory_start();

        size_t i = 0;

        if constexpr (unroll_normal_loops) {
            const size_t iend = (N & size_t(-4));

            for (; i < iend; i += 4) {
                lhs_mem[i] /= rhs[i];
                lhs_mem[i + 1] /= rhs[i + 1];
                lhs_mem[i + 2] /= rhs[i + 2];
                lhs_mem[i + 3] /= rhs[i + 3];
            }
        }

        for (; i < N; ++i) {
            lhs_mem[i] /= rhs[i];
        }
    }
};

} //end of namespace etl::detail
