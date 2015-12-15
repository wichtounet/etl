//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd> //For stream support

#include "etl/iterator.hpp"

// CRTP Classes
#include "etl/crtp/comparable.hpp"
#include "etl/crtp/value_testable.hpp"
#include "etl/crtp/dim_testable.hpp"

namespace etl {

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct binary_expr final : comparable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>, dim_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>, value_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
private:
    static_assert(cpp::or_c<
                      cpp::and_c<std::is_same<LeftExpr, scalar<T>>, std::is_same<RightExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<LeftExpr>, std::is_same<RightExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<RightExpr>, std::is_same<LeftExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<LeftExpr>, is_etl_expr<RightExpr>>>::value,
                  "One argument must be an ETL expression and the other one convertible to T");

    using this_type = binary_expr<T, LeftExpr, BinaryOp, RightExpr>;

    LeftExpr _lhs;
    RightExpr _rhs;

public:
    using value_type        = T;
    using memory_type       = void;
    using const_memory_type = void;

    template<typename V = default_vec>
    using vec_type          = typename V::template vec_type<T>;

    //Cannot be constructed with no args
    binary_expr() = delete;

    //Construct a new expression
    binary_expr(LeftExpr l, RightExpr r)
            : _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    binary_expr(const binary_expr& e) = default;
    binary_expr(binary_expr&& e) noexcept = default;

    //Expressions are invariant
    binary_expr& operator=(const binary_expr& e) = delete;
    binary_expr& operator=(binary_expr&& e) = delete;

    //Accessors

    std::add_lvalue_reference_t<LeftExpr> lhs() {
        return _lhs;
    }

    cpp::add_const_lvalue_t<LeftExpr> lhs() const {
        return _lhs;
    }

    std::add_lvalue_reference_t<RightExpr> rhs() {
        return _rhs;
    }

    cpp::add_const_lvalue_t<RightExpr> rhs() const {
        return _rhs;
    }

    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return _lhs.alias(rhs) || _rhs.alias(rhs);
    }

    //Apply the expression

    value_type operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }

    value_type read_flat(std::size_t i) const {
        return BinaryOp::apply(lhs().read_flat(i), rhs().read_flat(i));
    }

    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const {
        return BinaryOp::template load<V>(lhs().template load<V>(i), rhs().template load<V>(i));
    }

    template <typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<this_type>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return BinaryOp::apply(lhs()(args...), rhs()(args...));
    }

    //TODO Simplify the next two SFINAE functions

    template <typename ST = T, typename L = LeftExpr, typename B = BinaryOp, typename R = RightExpr, cpp_enable_if((sub_size_compare<binary_expr<ST, L, B, R>>::value > 1))>
    auto operator()(std::size_t i) {
        return sub(*this, i);
    }

    template <typename ST = T, typename L = LeftExpr, typename B = BinaryOp, typename R = RightExpr, cpp_enable_if((sub_size_compare<binary_expr<ST, L, B, R>>::value > 1))>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }
};

/*!
 * \brief Specialization for binary_expr.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    using expr_t       = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>;
    using left_expr_t  = std::decay_t<LeftExpr>;
    using right_expr_t = std::decay_t<RightExpr>;

    static constexpr const bool left_directed = cpp::not_u<etl_traits<left_expr_t>::is_generator>::value;

    using sub_expr_t = std::conditional_t<left_directed, left_expr_t, right_expr_t>;

    static constexpr const bool is_etl         = true;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view        = false;
    static constexpr const bool is_magic_view  = false;
    static constexpr const bool is_fast        = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_linear      = etl_traits<left_expr_t>::is_linear && etl_traits<right_expr_t>::is_linear && BinaryOp::linear;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator =
        etl_traits<left_expr_t>::is_generator && etl_traits<right_expr_t>::is_generator;
    static constexpr const bool vectorizable = etl_traits<left_expr_t>::vectorizable && etl_traits<right_expr_t>::vectorizable && BinaryOp::vectorizable;
    static constexpr const bool needs_temporary_visitor =
        etl_traits<left_expr_t>::needs_temporary_visitor || etl_traits<right_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor =
        etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order;

    template <bool B = left_directed, cpp_enable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.lhs();
    }

    template <bool B = left_directed, cpp_disable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.rhs();
    }

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(get(v));
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(get(v), d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};


template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
std::ostream& operator<<(std::ostream& os, const binary_expr<T, LeftExpr, BinaryOp, RightExpr>& expr) {
    if (simple_operator<BinaryOp>::value) {
        return os << "(" << expr.lhs() << ' ' << BinaryOp::desc() << ' ' << expr.rhs() << ")";
    } else {
        return os << BinaryOp::desc() << "(" << expr.lhs() << ", " << expr.rhs() << ")";
    }
}

} //end of namespace etl
