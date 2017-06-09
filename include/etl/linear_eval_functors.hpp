//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace detail {

/*!
 * \brief Functor for simple assign
 *
 * The result is written to lhs with operator[] and read from rhs
 * with read_flat
 */
template <typename L_Expr, typename R_Expr>
struct Assign {
    using value_type = value_t<L_Expr>; ///< The type of value

    value_type* lhs;         ///< The left hand side
    R_Expr rhs;              ///< The right hand side
    const size_t _size; ///< The size to assign

    /*!
     * \brief Constuct a new Assign
     * \param lhs The lhs memory
     * \param rhs The rhs memory
     */
    Assign(L_Expr lhs, R_Expr rhs) : lhs(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Assign rhs to lhs
     */
    void operator()(){
        size_t iend = 0;

        if (unroll_normal_loops) {
            iend = _size & size_t(-4);

            for (size_t i = 0; i < iend; i += 4) {
                lhs[i]     = rhs.read_flat(i);
                lhs[i + 1] = rhs.read_flat(i + 1);
                lhs[i + 2] = rhs.read_flat(i + 2);
                lhs[i + 3] = rhs.read_flat(i + 3);
            }
        }

        for (size_t i = iend; i < _size; ++i) {
            lhs[i] = rhs.read_flat(i);
        }
    }
};

/*!
 * \brief Functor for simple compound assign add
 */
template <typename L_Expr, typename R_Expr>
struct AssignAdd {
    using value_type = value_t<L_Expr>; ///< The type of value

    value_type* lhs;         ///< The left hand side
    R_Expr rhs;              ///< The right hand side
    const size_t _size;  ///< The size to assign

    /*!
     * \brief Constuct a new AssignAdd
     * \param lhs The lhs memory
     * \param rhs The rhs expression
     */
    AssignAdd(L_Expr lhs, R_Expr rhs) : lhs(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Assign rhs to lhs
     */
    void operator()(){
        size_t iend = 0;

        if (unroll_normal_loops) {
            iend = (_size & size_t(-4));

            for (size_t i = 0; i < iend; i += 4) {
                lhs[i] += rhs[i];
                lhs[i + 1] += rhs[i + 1];
                lhs[i + 2] += rhs[i + 2];
                lhs[i + 3] += rhs[i + 3];
            }
        }

        for (size_t i = iend; i < _size; ++i) {
            lhs[i] += rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign sub
 */
template <typename L_Expr, typename R_Expr>
struct AssignSub {
    using value_type = value_t<L_Expr>; ///< The type of value

    value_type* lhs;         ///< The left hand side
    R_Expr rhs;              ///< The right hand side
    const size_t _size;  ///< The size to assign

    /*!
     * \brief Constuct a new AssignSub
     * \param lhs The lhs memory
     * \param rhs The rhs expression
     */
    AssignSub(L_Expr lhs, R_Expr rhs) : lhs(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Assign rhs to lhs
     */
    void operator()(){
        size_t iend = 0;

        if (unroll_normal_loops) {
            iend = (_size & size_t(-4));

            for (size_t i = 0; i < iend; i += 4) {
                lhs[i] -= rhs[i];
                lhs[i + 1] -= rhs[i + 1];
                lhs[i + 2] -= rhs[i + 2];
                lhs[i + 3] -= rhs[i + 3];
            }
        }

        for (size_t i = iend; i < _size; ++i) {
            lhs[i] -= rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign mul
 */
template <typename L_Expr, typename R_Expr>
struct AssignMul {
    using value_type = value_t<L_Expr>; ///< The type of value

    value_type* lhs;         ///< The left hand side
    R_Expr rhs;              ///< The right hand side
    const size_t _size;  ///< The size to assign

    /*!
     * \brief Constuct a new AssignMul
     * \param lhs The lhs memory
     * \param rhs The rhs expression
     */
    AssignMul(L_Expr lhs, R_Expr rhs) : lhs(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Assign rhs to lhs
     */
    void operator()(){
        size_t iend = 0;

        if (unroll_normal_loops) {
            iend = (_size & size_t(-4));

            for (size_t i = 0; i < iend; i += 4) {
                lhs[i] *= rhs[i];
                lhs[i + 1] *= rhs[i + 1];
                lhs[i + 2] *= rhs[i + 2];
                lhs[i + 3] *= rhs[i + 3];
            }
        }

        for (size_t i = iend; i < _size; ++i) {
            lhs[i] *= rhs[i];
        }
    }
};

/*!
 * \brief Functor for compound assign div
 */
template <typename L_Expr, typename R_Expr>
struct AssignDiv {
    using value_type = value_t<L_Expr>; ///< The type of value

    value_type* lhs;         ///< The left hand side
    R_Expr rhs;              ///< The right hand side
    const size_t _size;  ///< The size to assign

    /*!
     * \brief Constuct a new AssignDiv
     * \param lhs The lhs memory
     * \param rhs The rhs expression
     */
    AssignDiv(L_Expr lhs, R_Expr rhs) : lhs(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Assign rhs to lhs
     */
    void operator()(){
        size_t iend = 0;

        if (unroll_normal_loops) {
            iend = (_size & size_t(-4));

            for (size_t i = 0; i < iend; i += 4) {
                lhs[i] /= rhs[i];
                lhs[i + 1] /= rhs[i + 1];
                lhs[i + 2] /= rhs[i + 2];
                lhs[i + 3] /= rhs[i + 3];
            }
        }

        for (size_t i = iend; i < _size; ++i) {
            lhs[i] /= rhs[i];
        }
    }
};

} //end of namespace detail

} //end of namespace etl
