//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_VISITOR_HPP
#define ETL_VISITOR_HPP

namespace etl {

template<typename D>
struct etl_visitor {
    using derived_t = D;

    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        as_derived()(v.value());
    }

    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        as_derived()(v.lhs());
        as_derived()(v.rhs());
    }

    template<typename T, typename AExpr, typename Op, typename Forced>
    void operator()(etl::temporary_unary_expr<T, AExpr, Op, Forced>& v) const {
        as_derived()(v.a());
    }

    template<typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
    void operator()(etl::temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& v) const {
        as_derived()(v.a());
        as_derived()(v.b());
    }

    template<typename T, cpp_enable_if(etl::is_view<T>::value)>
    void operator()(T& view) const {
        as_derived()(view.value());
    }

    template<typename L, typename R>
    void operator()(mm_mul_transformer<L,R>& transformer) const {
        as_derived()(transformer.lhs());
        as_derived()(transformer.rhs());
    }

    template<typename T, cpp_enable_if(etl::is_transformer<T>::value)>
    void operator()(T& transformer) const {
        as_derived()(transformer.value());
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

} //end of namespace etl

#endif
