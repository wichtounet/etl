//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <complex>

// Utilities
#include "etl/order.hpp"
#include "etl/sparse_storage.hpp"
#include "etl/config.hpp"
#include "etl/vectorization.hpp"
#include "etl/complex.hpp"

// The operators
#include "etl/op/scalar.hpp"
#include "etl/op/generators.hpp"
#include "etl/op/transformers.hpp"
#include "etl/op/views.hpp"
#include "etl/op/virtual_views.hpp"
#include "etl/op/unary_op.hpp"
#include "etl/op/binary_op.hpp"

//Global test functions
#include "etl/globals.hpp"

// The expressions
#include "etl/expr/binary_expr.hpp"
#include "etl/expr/unary_expr.hpp"
#include "etl/expr/generator_expr.hpp"
#include "etl/expr/temporary_expr.hpp"
#include "etl/expr/optimized_expr.hpp"

//Forward value classes for expressions
#include "etl/value_fwd.hpp"

// The expressions building
#include "etl/checks.hpp"
#include "etl/builder/expression_builder.hpp"

// The evaluator and optimizer
#include "etl/evaluator.hpp"
#include "etl/optimizer.hpp"

// The value classes implementation
#include "etl/fast.hpp"
#include "etl/dyn.hpp"
#include "etl/sparse.hpp"

// The traits
#include "etl/traits.hpp"

// to_string support
#include "etl/print.hpp"
