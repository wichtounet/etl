//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file eval_visitors.hpp
 * \brief Contains the visitors used by the evaluator to process the
 * expression trees.
*/

#pragma once

#include "etl/visitor.hpp" //visitor of the expressions

namespace etl {

namespace detail {

/*!
 * \brief Visitor to allocate temporary when needed
 */
struct temporary_allocator_visitor {
    // Simple tag
};

/*!
 * \brief Visitor to evict GPU temporaries from the Expression tree
 */
struct gpu_clean_static_visitor {
    // Simple tag
};

/*!
 * \brief Visitor to perform lcoal evaluation when necessary
 */
struct evaluator_static_visitor {
    /*!
     * \brief Indicates if the visitor is necessary for the given expression
     */
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_evaluator_visitor>;

    mutable bool need_value = false; ///< Indicates if the value if necessary for the next visits

    /*!
     * \brief Visit the given temporary unary expression
     * \param v The temporary unary expression
     */
    template <typename D, typename T, typename A, typename R>
    void operator()(etl::temporary_expr_un<D, T, A, R>& v) const {
        bool old_need_value = need_value;

        need_value = decay_traits<D>::is_gpu;
        (*this)(v.a());

        v.evaluate();

        if (old_need_value) {
            v.direct().gpu_copy_from_if_necessary();
        }

        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given temporary binary expression
     * \param v The temporary binary expression
     */
    template <typename D, typename T, typename A, typename B, typename R>
    void operator()(etl::temporary_expr_bin<D, T, A, B, R>& v) const {
        bool old_need_value = need_value;

        need_value = decay_traits<D>::is_gpu;
        (*this)(v.a());

        need_value = decay_traits<D>::is_gpu;
        (*this)(v.b());

        v.evaluate();

        if (old_need_value) {
            v.direct().gpu_copy_from_if_necessary();
        }

        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given unary expression
     * \param v The unary expression
     */
    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(v.value());
        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given binary expression
     * \param v The binary expression
     */
    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(v.lhs());
        need_value = true;
        (*this)(v.rhs());
        need_value = old_need_value;
    }

    template <typename T, cpp_enable_if(fast_sub_view_able<T>::value)>
    void operator()(etl::sub_view<T>& v) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(v.value());
        v.late_init();
        need_value = old_need_value;
    }

    template <typename T, cpp_disable_if(fast_sub_view_able<T>::value)>
    void operator()(etl::sub_view<T>& v) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(v.value());
        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given view
     * \param view The view
     */
    template <typename T, cpp_enable_if(etl::is_view<T>::value)>
    void operator()(T& view) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(view.value());
        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given matrix-multiplication transformer
     * \param transformer The matrix-multiplication transformer
     */
    template <typename L, typename R>
    void operator()(mm_mul_transformer<L, R>& transformer) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(transformer.lhs());
        need_value = true;
        (*this)(transformer.rhs());
        need_value = old_need_value;
    }

    /*!
     * \brief Visit the given transformer
     * \param transformer The transformer
     */
    template <typename T, cpp_enable_if(etl::is_transformer<T>::value)>
    void operator()(T& transformer) const {
        bool old_need_value = need_value;
        need_value = true;
        (*this)(transformer.value());
        need_value = old_need_value;
    }

    //The leaves don't need any special handling

    /*!
     * \brief Visit the given generator
     * \param generator The generator
     */
    template <typename Generator>
    void operator()(const generator_expr<Generator>& generator) const {
        cpp_unused(generator);
        //Leaf
    }

    /*!
     * \brief Visit the given magic view
     * \param view The magic view
     */
    template <typename T, cpp_enable_if(etl::is_magic_view<T>::value)>
    void operator()(const T& view) const {
        cpp_unused(view);
        //Leaf
    }

    /*!
     * \brief Visit the given value class
     * \param v The value class
     */
    template <typename T, cpp_enable_if(etl::is_etl_value<T>::value)>
    void operator()(const T& v) const {
        cpp_unused(v);
        //Leaf
    }

    /*!
     * \brief Visit the given scalar
     * \param s The scalar
     */
    template <typename T>
    void operator()(const etl::scalar<T>& s) const {
        cpp_unused(s);
        //Leaf
    }
};

} //end of namespace detail

} //end of namespace etl
