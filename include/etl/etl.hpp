//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_ETL_HPP
#define ETL_ETL_HPP

// The operators
#include "generators.hpp"
#include "transformers.hpp"
#include "views.hpp"
#include "virtual_views.hpp"
#include "unary_op.hpp"
#include "binary_op.hpp"

// The expressions
#include "binary_expr.hpp"
#include "unary_expr.hpp"
#include "stable_transform_expr.hpp"
#include "generator_expr.hpp"
#include "temporary_expr.hpp"

namespace etl {

template<typename T, typename ST, std::size_t... Dims>
struct fast_matrix_impl;

template<typename T, std::size_t... Dims>
using fast_matrix = fast_matrix_impl<T, std::array<T, mul_all<Dims...>::value>, Dims...>;

template<typename T, std::size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, std::vector<T>, Dims...>;

template<typename T, std::size_t D = 2>
struct dyn_matrix;

template<typename T>
using dyn_vector = dyn_matrix<T, 1>;

}

// The complex expressions
#include "mmul_expr.hpp"
#include "conv_expr.hpp"

#include "fast_expr.hpp"

// The value classes
#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/fast_dyn_matrix.hpp"
#include "etl/fast_dyn_vector.hpp"

// The traits
#include "traits.hpp"

// The expressions building
#include "fast_expr.hpp"

// to_string support
#include "print.hpp"

#endif
