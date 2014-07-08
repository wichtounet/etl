//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_STOP_HPP
#define ETL_STOP_HPP

#include "tmp.hpp"

namespace etl {

template<typename T, std::size_t Rows>
struct fast_vector;

template<typename T>
struct dyn_vector;

template<typename T>
struct dyn_matrix;

template<typename T, size_t Rows, size_t Columns>
struct fast_matrix;

template <typename T, typename Expr, typename UnaryOp>
class unary_expr;

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr;

//TODO Implement it for fast_XXX

template<typename T, typename Enable = void>
struct stop;

template <typename T, typename Expr, typename UnaryOp>
struct stop <unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<unary_expr<T, Expr, UnaryOp>>::is_vector>> {
    template<typename TT>
    static dyn_vector<T> s(TT&& value){
        return {std::forward<TT>(value)};
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct stop <unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<unary_expr<T, Expr, UnaryOp>>::is_matrix>> {
    template<typename TT>
    static dyn_matrix<T> s(TT&& value){
        return {std::forward<TT>(value)};
    }
};

template<typename T>
auto s(const T& value){
    return stop<T>::s(value);
}

} //end of namespace etl

#endif
