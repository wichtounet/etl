//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Simple CRTP class to implement a visitor
 * \tparam D the type of the derived visitor class
 *
 * The default implementation is simply to visit the whole tree while not doing anything
 */
template <typename D, bool V_T = true, bool V_V = true>
struct etl_visitor {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    /*!
     * \brief Visit the given unary expression
     * \param v The unary expression
     */
    template <typename T, typename Expr, typename UnaryOp>
    void operator()(etl::unary_expr<T, Expr, UnaryOp>& v) const {
        as_derived()(v.value());
    }

    /*!
     * \brief Visit the given binary expression
     * \param v The binary expression
     */
    template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
    void operator()(etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v) const {
        as_derived()(v.lhs());
        as_derived()(v.rhs());
    }

    /*!
     * \brief Visit the given view
     * \param view The view
     */
    template <typename T, cpp_enable_if(etl::is_view<T>::value)>
    void operator()(T& view) const {
        as_derived()(view.value());
    }

    /*!
     * \brief Visit the given matrix-multiplication transformer
     * \param transformer The matrix-multiplication transformer
     */
    template <typename L, typename R>
    void operator()(mm_mul_transformer<L, R>& transformer) const {
        as_derived()(transformer.lhs());
        as_derived()(transformer.rhs());
    }

    /*!
     * \brief Visit the given transformer
     * \param transformer The transformer
     */
    template <typename T, cpp_enable_if(etl::is_transformer<T>::value)>
    void operator()(T& transformer) const {
        as_derived()(transformer.value());
    }

    /*!
     * \brief Visit the given temporary unary expr
     * \param v The temporary unary expr
     */
    template <typename T, cpp_enable_if_cst(V_T && is_temporary_unary_expr<T>::value)>
    void operator()(T& v){
        as_derived()(v.a());
    }

    /*!
     * \brief Visit the given temporary binary expr
     * \param v The temporary binary expr
     */
    template <typename T, cpp_enable_if_cst(V_T && is_temporary_binary_expr<T>::value)>
    void operator()(T& v) const {
        as_derived()(v.a());
        as_derived()(v.b());
    }

    /*!
     * \brief Visit the given generator expr
     * \param v The the generator expr
     */
    template <typename Generator>
    void operator()(const generator_expr<Generator>& v) const {
        cpp_unused(v);
        //Leaf
    }

    /*!
     * \brief Visit the given magic view.
     * \param v The the magic view.
     */
    template <typename T, cpp_enable_if(etl::is_magic_view<T>::value)>
    void operator()(const T& v) const {
        cpp_unused(v);
        //Leaf
    }

    /*!
     * \brief Visit the given value class.
     * \param v The the value class.
     */
    template <typename T, cpp_enable_if(V_V && etl::is_etl_value<T>::value)>
    void operator()(const T& v) const {
        cpp_unused(v);
        //Leaf
    }

    /*!
     * \brief Visit the given scalar.
     * \param v The the scalar.
     */
    template <typename T>
    void operator()(const etl::scalar<T>& v) const {
        cpp_unused(v);
        //Leaf
    }
};

/*!
 * \brief Apply the given visitor to the given expression
 * \param visitor The visitor to use
 * \param expr The expression to visit
 */
template <typename Visitor, typename Expr, cpp_enable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(const Visitor& visitor, Expr& expr) {
    visitor(expr);
}

/*!
 * \brief Apply the given visitor to the given expression
 * \param visitor The visitor to use
 * \param expr The expression to visit
 */
template <typename Visitor, typename Expr, cpp_enable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(Visitor& visitor, Expr& expr) {
    visitor(expr);
}

/*!
 * \brief Apply the given visitor to the given expression
 * \param expr The expression to visit
 * \tparam Visitor The visitor to use, will be constructed on the stack
 */
template <typename Visitor, typename Expr, cpp_enable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(Expr& expr) {
    Visitor visitor;
    visitor(expr);
}

template <typename Visitor, typename Expr, cpp_disable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(const Visitor& /*visitor*/, Expr& /*expr*/) {}

template <typename Visitor, typename Expr, cpp_disable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(Visitor& /*visitor*/, Expr& /*expr*/) {}

template <typename Visitor, typename Expr, cpp_disable_if(Visitor::template enabled<Expr>::value)>
void apply_visitor(Expr& /*expr*/) {}

} //end of namespace etl
