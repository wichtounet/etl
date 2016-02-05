//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

struct identity_op;

struct transform_op;

template <typename Sub>
struct stateful_op;

template <typename T, typename Expr, typename UnaryOp>
struct unary_expr;

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct binary_expr;

template <typename Generator>
class generator_expr;

template <typename Expr>
struct optimized_expr;

template <typename Expr>
struct serial_expr;

template <typename Expr, typename R = etl::nanoseconds>
struct timed_expr;

template <typename T, typename AExpr, typename Op, typename Forced>
struct temporary_unary_expr;

template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
struct temporary_binary_expr;

template <typename T, std::size_t D>
struct dim_view;

template <typename T>
struct sub_view;

template <typename T, std::size_t... Dims>
struct fast_matrix_view;

template <typename T>
struct dyn_vector_view;

template <typename T>
struct dyn_matrix_view;

} //end of namespace etl
