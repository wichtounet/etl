//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <vector>
#include <array>

namespace etl {

template <typename T, typename ST, order SO, std::size_t... Dims>
struct fast_matrix_impl;

template <typename T, order SO, std::size_t D = 2>
struct dyn_matrix_impl;

template <typename T, std::size_t... Dims>
using fast_matrix = fast_matrix_impl<T, std::array<T, mul_all<Dims...>::value>, order::RowMajor, Dims...>;

template <typename T, std::size_t... Dims>
using fast_matrix_cm = fast_matrix_impl<T, std::array<T, mul_all<Dims...>::value>, order::ColumnMajor, Dims...>;

template <typename T, std::size_t Rows>
using fast_vector = fast_matrix_impl<T, std::array<T, Rows>, order::RowMajor, Rows>;

template <typename T, std::size_t Rows>
using fast_vector_cm = fast_matrix_impl<T, std::array<T, Rows>, order::ColumnMajor, Rows>;

template <typename T, std::size_t Rows>
using fast_dyn_vector = fast_matrix_impl<T, std::vector<T>, order::RowMajor, Rows>;

template <typename T, std::size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, std::vector<T>, order::RowMajor, Dims...>;

template <typename T, std::size_t D = 2>
using dyn_matrix                    = dyn_matrix_impl<T, order::RowMajor, D>;

template <typename T, std::size_t D = 2>
using dyn_matrix_cm                 = dyn_matrix_impl<T, order::ColumnMajor, D>;

template <typename T>
using dyn_vector = dyn_matrix_impl<T, order::RowMajor, 1>;

template <typename T, sparse_storage SS, std::size_t D>
struct sparse_matrix_impl;

template <typename T, std::size_t D = 2>
using sparse_matrix = sparse_matrix_impl<T, sparse_storage::COO, D>;

//TODO Move exprssions in expr_fwd

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
