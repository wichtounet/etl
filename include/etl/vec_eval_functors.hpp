//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace detail {

/*!
 * \brief Common base for vectorized functors
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct vectorized_base {
    using memory_type = value_t<L_Expr>*; ///< The memory type

    L_Expr lhs;              ///< The left hand side
    memory_type lhs_m;       ///< The left hand side memory
    R_Expr rhs;              ///< The right hand side
    const size_t _size; ///< The size to assign

    /*!
     * \brief The RHS value type
     */
    using lhs_value_type = value_t<L_Expr>;

    /*!
     * \brief The RHS value type
     */
    using rhs_value_type = value_t<R_Expr>;

    /*!
     * \brief The intrinsic type for the value type
     */
    using IT = typename get_intrinsic_traits<V>::template type<rhs_value_type>;

    /*!
     * \brief The vector implementation to use
     */
    using vect_impl = typename get_vector_impl<V>::type;

    /*!
     * \brief Constuct a new vectorized_base
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    vectorized_base(L_Expr lhs, R_Expr rhs) : lhs(lhs), lhs_m(lhs.memory_start()), rhs(rhs), _size(etl::size(lhs)) {
        //Nothing else
    }

    /*!
     * \brief Load a vector from lhs at position i
     * \param i The index where to start loading from
     * \return a vector from lhs starting at position i
     */
    inline auto lhs_load(size_t i) const {
        return lhs.template load<vect_impl>(i);
    }

    /*!
     * \brief Load a vector from rhs at position i
     * \param i The index where to start loading from
     * \return a vector from rhs starting at position i
     */
    inline auto rhs_load(size_t i) const {
        return rhs.template load<vect_impl>(i);
    }
};

/*!
 * \brief Functor for vectorized assign
 *
 * The result is computed in a vectorized fashion with several
 * operations per cycle and written directly to the memory of lhs.
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct VectorizedAssign : vectorized_base<V, L_Expr, R_Expr> {
    using base_t    = vectorized_base<V, L_Expr, R_Expr>; ///< The base type
    using IT        = typename base_t::IT;                ///< The intrisic type
    using vect_impl = typename base_t::vect_impl;         ///< The vector implementation

    using base_t::lhs_m;
    using base_t::lhs;
    using base_t::rhs;
    using base_t::_size;
    using base_t::rhs_load;

    /*!
     * \brief Constuct a new VectorizedAssign
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    VectorizedAssign(L_Expr lhs, R_Expr rhs) : base_t(lhs, rhs) {
        //Nothing else
    }

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    void operator()(){
        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>::value;

        const size_t last = remainder ? (_size & size_t(-IT::size)) : _size;

        size_t i = 0;

        if(streaming && _size > stream_threshold / (sizeof(typename base_t::lhs_value_type) * 3) && !rhs.alias(lhs)){
            for (; i < last; i += IT::size) {
                lhs.template stream<vect_impl>(rhs_load(i), i);
            }

            for (; remainder && i < _size; ++i) {
                lhs_m[i] = rhs[i];
            }
        } else {
            for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
                lhs.template store<vect_impl>(rhs_load(i + 0 * IT::size), i + 0 * IT::size);
                lhs.template store<vect_impl>(rhs_load(i + 1 * IT::size), i + 1 * IT::size);
                lhs.template store<vect_impl>(rhs_load(i + 2 * IT::size), i + 2 * IT::size);
                lhs.template store<vect_impl>(rhs_load(i + 3 * IT::size), i + 3 * IT::size);
            }

            for (; i < last; i += IT::size) {
                lhs.template store<vect_impl>(rhs_load(i), i);
            }

            for (; remainder && i < _size; ++i) {
                lhs_m[i] = rhs[i];
            }
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign add
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct VectorizedAssignAdd : vectorized_base<V, L_Expr, R_Expr> {
    using base_t    = vectorized_base<V, L_Expr, R_Expr>; ///< The base type
    using IT        = typename base_t::IT;                ///< The intrisic type
    using vect_impl = typename base_t::vect_impl;         ///< The vector implementation

    using base_t::lhs;
    using base_t::lhs_m;
    using base_t::rhs;
    using base_t::_size;
    using base_t::lhs_load;
    using base_t::rhs_load;

    /*!
     * \brief Constuct a new VectorizedAssignAdd
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    VectorizedAssignAdd(L_Expr lhs, R_Expr rhs) : base_t(lhs, rhs) {
        //Nothing else
    }

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    void operator()(){
        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>::value;

        const size_t last = remainder ? (_size & size_t(-IT::size)) : _size;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::add(lhs_load(i + 0 * IT::size), rhs_load(i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(lhs_load(i + 1 * IT::size), rhs_load(i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(lhs_load(i + 2 * IT::size), rhs_load(i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::add(lhs_load(i + 3 * IT::size), rhs_load(i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::add(lhs_load(i), rhs_load(i)), i);
        }

        for (; remainder && i < _size; ++i) {
            lhs_m[i] += rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign sub
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct VectorizedAssignSub : vectorized_base<V, L_Expr, R_Expr> {
    using base_t    = vectorized_base<V, L_Expr, R_Expr>; ///< The base type
    using IT        = typename base_t::IT;                ///< The intrisic type
    using vect_impl = typename base_t::vect_impl;         ///< The vector implementation

    using base_t::lhs;
    using base_t::lhs_m;
    using base_t::rhs;
    using base_t::_size;
    using base_t::lhs_load;
    using base_t::rhs_load;

    /*!
     * \brief Constuct a new VectorizedAssignSub
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    VectorizedAssignSub(L_Expr lhs, R_Expr rhs) : base_t(lhs, rhs) {
        //Nothing else
    }

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    void operator()() {
        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>::value;

        const size_t last = remainder ? (_size & size_t(-IT::size)) : _size;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::sub(lhs_load(i + 0 * IT::size), rhs_load(i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(lhs_load(i + 1 * IT::size), rhs_load(i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(lhs_load(i + 2 * IT::size), rhs_load(i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::sub(lhs_load(i + 3 * IT::size), rhs_load(i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::sub(lhs_load(i), rhs_load(i)), i);
        }

        for (; remainder && i < _size; ++i) {
            lhs_m[i] -= rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign mul
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct VectorizedAssignMul : vectorized_base<V, L_Expr, R_Expr> {
    using base_t    = vectorized_base<V, L_Expr, R_Expr>; ///< The base type
    using IT        = typename base_t::IT;                ///< The intrisic type
    using vect_impl = typename base_t::vect_impl;         ///< The vector implementation

    using base_t::lhs;
    using base_t::lhs_m;
    using base_t::rhs;
    using base_t::_size;
    using base_t::lhs_load;
    using base_t::rhs_load;

    /*!
     * \brief Constuct a new VectorizedAssignMul
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    VectorizedAssignMul(L_Expr lhs, R_Expr rhs) : base_t(lhs, rhs) {
        //Nothing else
    }

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    void operator()(){
        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>::value;

        const size_t last = remainder ? (_size & size_t(-IT::size)) : _size;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::mul(lhs_load(i + 0 * IT::size), rhs_load(i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(lhs_load(i + 1 * IT::size), rhs_load(i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(lhs_load(i + 2 * IT::size), rhs_load(i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::mul(lhs_load(i + 3 * IT::size), rhs_load(i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::mul(lhs_load(i), rhs_load(i)), i);
        }

        for (; remainder && i < _size; ++i) {
            lhs_m[i] *= rhs[i];
        }
    }
};

/*!
 * \brief Functor for vectorized compound assign div
 */
template <vector_mode_t V, typename L_Expr, typename R_Expr>
struct VectorizedAssignDiv : vectorized_base<V, L_Expr, R_Expr> {
    using base_t    = vectorized_base<V, L_Expr, R_Expr>; ///< The base type
    using IT        = typename base_t::IT;                ///< The intrisic type
    using vect_impl = typename base_t::vect_impl;         ///< The vector implementation

    using base_t::lhs;
    using base_t::lhs_m;
    using base_t::rhs;
    using base_t::_size;
    using base_t::lhs_load;
    using base_t::rhs_load;

    /*!
     * \brief Constuct a new VectorizedAssignDiv
     * \param lhs The lhs expression
     * \param rhs The rhs expression
     */
    VectorizedAssignDiv(L_Expr lhs, R_Expr rhs) : base_t(lhs, rhs) {
        //Nothing else
    }

    /*!
     * \brief Compute the vectorized iterations of the loop using aligned store operations
     */
    void operator()(){
        constexpr bool remainder = !padding || !all_padded<L_Expr, R_Expr>::value;

        const size_t last = remainder ? (_size & size_t(-IT::size)) : _size;

        size_t i = 0;

        for (; i + (IT::size * 3) < last; i += 4 * IT::size) {
            lhs.template store<vect_impl>(vect_impl::div(lhs_load(i + 0 * IT::size), rhs_load(i + 0 * IT::size)), i + 0 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(lhs_load(i + 1 * IT::size), rhs_load(i + 1 * IT::size)), i + 1 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(lhs_load(i + 2 * IT::size), rhs_load(i + 2 * IT::size)), i + 2 * IT::size);
            lhs.template store<vect_impl>(vect_impl::div(lhs_load(i + 3 * IT::size), rhs_load(i + 3 * IT::size)), i + 3 * IT::size);
        }

        for (; i < last; i += IT::size) {
            lhs.template store<vect_impl>(vect_impl::div(lhs_load(i), rhs_load(i)), i);
        }

        for (; remainder && i < _size; ++i) {
            lhs_m[i] /= rhs[i];
        }
    }
};

} //end of namespace detail

} //end of namespace etl
