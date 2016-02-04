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

#include "etl/visitor.hpp"        //visitor of the expressions

namespace etl {

namespace detail {

struct temporary_allocator_static_visitor : etl_visitor<temporary_allocator_static_visitor, false, true> {
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_temporary_visitor>;

    using etl_visitor<temporary_allocator_static_visitor, false, true>::operator();

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        v.allocate_temporary();

        (*this)(v.a());
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        v.allocate_temporary();

        (*this)(v.a());
        (*this)(v.b());
    }
};

struct evaluator_static_visitor {
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_evaluator_visitor>;

    mutable bool need_value = false;

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        bool old_need_value = need_value;

        need_value = Op::is_gpu;
        (*this)(v.a());

        v.evaluate();

        if(old_need_value){
            v.gpu_copy_from_if_necessary();
        }

        need_value = old_need_value;
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        bool old_need_value = need_value;

        need_value = Op::is_gpu;
        (*this)(v.a());

        need_value = Op::is_gpu;
        (*this)(v.b());

        v.evaluate();

        if(old_need_value){
            v.gpu_copy_from_if_necessary();
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

    template <typename Generator>
    void operator()(const generator_expr<Generator>& /*unused*/) const {
        //Leaf
    }

    template <typename T, cpp_enable_if(etl::is_magic_view<T>::value)>
    void operator()(const T& /*unused*/) const {
        //Leaf
    }

    template <typename T, cpp_enable_if(etl::is_etl_value<T>::value)>
    void operator()(const T& /*unused*/) const {
        //Leaf
    }

    template <typename T>
    void operator()(const etl::scalar<T>& /*unused*/) const {
        //Leaf
    }
};

struct gpu_clean_static_visitor : etl_visitor<gpu_clean_static_visitor, false, false> {
#ifdef ETL_CUDA
    template <typename E>
    using enabled = cpp::bool_constant<true>;
#else
    template <typename E>
    using enabled = cpp::bool_constant<false>;
#endif

    using etl_visitor<gpu_clean_static_visitor, false, false>::operator();

    template <typename T, cpp_enable_if(etl::is_etl_value<T>::value && !etl::is_sparse_matrix<T>::value)>
    void operator()(const T& value) const {
        value.gpu_evict();
    }

    template <typename T, cpp_enable_if(etl::is_sparse_matrix<T>::value)>
    void operator()(const T& /*value*/) const {
        //Nothing to do: no GPU support for sparse matrix
    }

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        (*this)(v.a());

        v.gpu_evict();
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        (*this)(v.a());
        (*this)(v.b());

        v.gpu_evict();
    }
};

} //end of namespace detail

} //end of namespace etl
