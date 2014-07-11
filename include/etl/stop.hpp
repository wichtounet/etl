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

//TODO Implement it for fast_XXX

template<typename T, typename Enable = void>
struct stop;

template <typename T, typename Expr, typename UnaryOp>
struct stop <unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<unary_expr<T, Expr, UnaryOp>>::is_vector>> {
    template<typename TT, typename T2 = Expr, disable_if_u<etl_traits<T2>::is_fast> = detail::dummy>
    static dyn_vector<T> s(TT&& value){
        return {std::forward<TT>(value)};
    }

    template<typename TT, typename T2 = remove_cv_t<remove_reference_t<Expr>>, enable_if_u<etl_traits<T2>::is_fast> = detail::dummy>
    static auto s(TT&& value){
        return fast_vector<T, etl_traits<T2>::size()>(std::forward<TT>(value));
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct stop <unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<unary_expr<T, Expr, UnaryOp>>::is_matrix>> {
    template<typename TT, typename T2 = Expr, disable_if_u<etl_traits<T2>::is_fast> = detail::dummy>
    static dyn_matrix<T> s(TT&& value){
        return {std::forward<TT>(value)};
    }

    template<typename TT, typename T2 = remove_cv_t<remove_reference_t<Expr>>, enable_if_u<etl_traits<T2>::is_fast> = detail::dummy>
    static auto s(TT&& value){
        return fast_matrix<T, etl_traits<Expr>::rows(), etl_traits<Expr>::columns()>(std::forward<TT>(value));
    }
};

template<typename T>
auto s(const T& value){
    return stop<T>::s(value);
}

} //end of namespace etl

#endif
