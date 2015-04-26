//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_EVALUATOR_HPP
#define ETL_EVALUATOR_HPP

#include "traits_lite.hpp"   //forward declaration of the traits

namespace etl {

namespace detail {

//TODO Disable completely the visitor if there are no temporary_expr in the tree (use TMP)
struct temporary_allocator_static_visitor {
    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        (*this)(v.value());
    }

    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        (*this)(v.lhs());
        (*this)(v.rhs());
    }
    
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

    template<typename T, cpp_enable_if(etl::is_view<T>::value)>
    void operator()(T& view) const {
        (*this)(view.value());
    }

    template<typename L, typename R>
    void operator()(mm_mul_transformer<L,R>& transformer) const {
        (*this)(transformer.lhs());
        (*this)(transformer.rhs());
    }

    template<typename T, cpp_enable_if(etl::is_transformer<T>::value)>
    void operator()(T& transformer) const {
        (*this)(transformer.value());
    }

    template <typename Generator>
    void operator()(const generator_expr<Generator>&) const {
        //Leaf
    }


    template<typename T, cpp_enable_if(etl::is_magic_view<T>::value)>
    void operator()(const T&) const {
        //Leaf
    }

    template<typename T, cpp_enable_if(etl::is_etl_value<T>::value)>
    void operator()(const T&) const {
        //Leaf
    }

    template <typename T>
    void operator()(const etl::scalar<T>&) const {
        //Leaf
    }
};

//TODO Disable completely the visitor if there are no temporary_expr in the tree (use TMP)
struct evaluator_static_visitor {
    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        (*this)(v.value());
    }

    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        (*this)(v.lhs());
        (*this)(v.rhs());
    }

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

    template<typename T, cpp_enable_if(etl::is_view<T>::value)>
    void operator()(T& view) const {
        (*this)(view.value());
    }

    template<typename L, typename R>
    void operator()(mm_mul_transformer<L,R>& transformer) const {
        (*this)(transformer.lhs());
        (*this)(transformer.rhs());
    }

    template<typename T, cpp_enable_if(etl::is_transformer<T>::value)>
    void operator()(T& transformer) const {
        (*this)(transformer.value());
    }

    template <typename Generator>
    void operator()(const generator_expr<Generator>&) const {
        //Leaf
    }

    template<typename T, cpp_enable_if(etl::is_magic_view<T>::value)>
    void operator()(const T&) const {
        //Leaf
    }

    template<typename T, cpp_enable_if(etl::is_etl_value<T>::value)>
    void operator()(const T&) const {
        //Leaf
    }

    template <typename T>
    void operator()(const etl::scalar<T>&) const {
        //Leaf
    }
};

} //end of namespace detail

template<typename Expr, typename Result>
struct standard_evaluator {
    template<typename E, typename R, cpp_disable_if(is_temporary_expr<E>::value)>
    static void evaluate(E&& expr, R&& result){
        detail::temporary_allocator_static_visitor allocator_visitor;
        allocator_visitor(expr);

        detail::evaluator_static_visitor evaluator_visitor;
        evaluator_visitor(expr);

        for(std::size_t i = 0; i < etl::size(result); ++i){
            result[i] = expr[i];
        }
    }

    //Note: In case of direct evaluation, temporary_expr must be
    //evaluated by the static_visitor, otherwise, the result would
    //be evaluated twice and a temporary would be allocated for nothing

    template<typename E, typename R, cpp_enable_if(is_temporary_unary_expr<E>::value)>
    static void evaluate(E&& expr, R&& result){
        detail::temporary_allocator_static_visitor allocator_visitor;
        allocator_visitor(expr.a());

        detail::evaluator_static_visitor evaluator_visitor;
        evaluator_visitor(expr.a());

        expr.direct_evaluate(result);
    }

    template<typename E, typename R, cpp_enable_if(is_temporary_binary_expr<E>::value)>
    static void evaluate(E&& expr, R&& result){
        detail::temporary_allocator_static_visitor allocator_visitor;
        allocator_visitor(expr.a());
        allocator_visitor(expr.b());

        detail::evaluator_static_visitor evaluator_visitor;
        evaluator_visitor(expr.a());
        evaluator_visitor(expr.b());

        expr.direct_evaluate(result);
    }

    template<typename E, cpp_enable_if(is_temporary_expr<E>::value)>
    static void evaluate(E&& expr){
        detail::temporary_allocator_static_visitor allocator_visitor;
        allocator_visitor(expr);

        detail::evaluator_static_visitor evaluator_visitor;
        evaluator_visitor(expr);
    }
};

template<typename Expr, typename Result>
void evaluate(Expr&& expr, Result&& result){
    standard_evaluator<Expr, Result>::evaluate(std::forward<Expr>(expr), std::forward<Result>(result));
}

template<typename Expr>
void force(Expr&& expr){
    standard_evaluator<Expr, void>::evaluate(std::forward<Expr>(expr));
}

} //end of namespace etl

#endif
