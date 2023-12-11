//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template <typename Expr>
struct optimizable;

template <typename Expr>
struct optimizer;

template <typename Expr>
struct transformer;

struct identity_op;

struct transform_op;

template <typename Sub>
struct stateful_op;

template <typename T, expr_or_scalar<T> Expr, typename UnaryOp>
struct unary_expr;

template <typename T, expr_or_scalar<T> LeftExpr, typename BinaryOp, expr_or_scalar<T> RightExpr>
struct binary_expr;

template <typename Generator>
class generator_expr;

template <typename Expr>
struct optimized_expr;

template <typename Expr>
struct serial_expr;

template <typename Selector, Selector V, typename Expr>
struct selected_expr;

template <typename Expr>
struct parallel_expr;

template <typename Expr, typename R = etl::nanoseconds>
struct timed_expr;

template <typename D, typename T, typename A, typename B, typename R>
struct temporary_expr_bin;

template <typename T, size_t D>
struct dim_view;

template <typename T, bool Aligned>
struct sub_view;

template <typename T, bool Aligned>
struct sub_matrix_2d;

template <typename T, bool Aligned>
struct sub_matrix_3d;

template <typename T, bool Aligned>
struct sub_matrix_4d;

template <typename T>
struct slice_view;

template <typename T, bool Aligned>
struct memory_slice_view;

template <typename T, bool DMA, size_t... Dims>
struct fast_matrix_view;

template <typename T, size_t D>
struct dyn_matrix_view;

template <typename A>
struct transpose_expr;

template <etl_expr A>
struct transpose_front_expr;

template <typename D, bool Fast>
struct base_temporary_expr;

template <typename T>
struct scalar;

} //end of namespace etl
