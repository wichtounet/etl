//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_EVALUATOR_HPP
#define ETL_EVALUATOR_HPP

#include "traits_fwd.hpp"   //forward declaration of the traits

namespace etl {

namespace detail {

//TODO Disable completely the visitor if there are no temporary_expr in the tree (use TMP)

struct temporary_allocator_static_visitor {
    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        (*this)(v.value());
    }

    template <typename T, typename Expr>
    void operator()(etl::stable_transform_expr<T, Expr>& v) const {
        (*this)(v.value());
    }

    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        (*this)(v.lhs());
        (*this)(v.rhs());
    }

    template <typename T, typename AExpr, typename BExpr, typename Op>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op>& v) const {
        v.allocate_temporary();

        (*this)(v.a());
        (*this)(v.b());
    }

    //TODO Handle transformers and views

    template<typename T>
    void operator()(T&) const {
        //fallback
    }
};

} //end of namespace detail

template<typename Expr, typename Result>
struct standard_evaluator {
    template<typename E, typename R, cpp_disable_if(cpp::is_specialization_of<etl::temporary_binary_expr, std::decay_t<E>>::value)>
    static void evaluate(E&& expr, R&& result){
        detail::temporary_allocator_static_visitor visitor;
        visitor(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] = expr[i];
        }
    }

    template<typename E, typename R, cpp_enable_if(cpp::is_specialization_of<etl::temporary_binary_expr, std::decay_t<E>>::value)>
    static void evaluate(E&& expr, R&& result){
        detail::temporary_allocator_static_visitor visitor;
        visitor(expr.a());
        visitor(expr.b());

        expr.direct_evaluate(result);
    }
};

template<typename Expr, typename Result>
void evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

} //end of namespace etl

#endif
