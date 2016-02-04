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

struct evaluator_static_visitor : etl_visitor<evaluator_static_visitor, false, true> {
    template <typename E>
    using enabled = cpp::bool_constant<decay_traits<E>::needs_evaluator_visitor>;

    using etl_visitor<evaluator_static_visitor, false, true>::operator();

    template <typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        (*this)(v.a());

        v.evaluate();
    }

    template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        (*this)(v.a());
        (*this)(v.b());

        v.evaluate();
    }
};

struct gpu_clean_static_visitor : etl_visitor<gpu_clean_static_visitor, true, false> {
#ifdef ETL_CUDA
    template <typename E>
    using enabled = cpp::bool_constant<true>;
#else
    template <typename E>
    using enabled = cpp::bool_constant<false>;
#endif

    using etl_visitor<gpu_clean_static_visitor, true, false>::operator();

    template <typename T, cpp_enable_if(etl::is_etl_value<T>::value && !etl::is_sparse_matrix<T>::value)>
    void operator()(const T& value) const {
        value.gpu_evict();
    }

    template <typename T, cpp_enable_if(etl::is_sparse_matrix<T>::value)>
    void operator()(const T& /*value*/) const {
        //Nothing to do: no GPU support for sparse matrix
    }
};

} //end of namespace detail

} //end of namespace etl
