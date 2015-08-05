//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include<complex>

// Utilities
#include "order.hpp"
#include "config.hpp"
#include "vectorization.hpp"

// The operators
#include "op/scalar.hpp"
#include "op/generators.hpp"
#include "op/transformers.hpp"
#include "op/views.hpp"
#include "op/virtual_views.hpp"
#include "op/unary_op.hpp"
#include "op/binary_op.hpp"

// The expressions
#include "expr/binary_expr.hpp"
#include "expr/unary_expr.hpp"
#include "expr/generator_expr.hpp"
#include "expr/temporary_expr.hpp"
#include "expr/optimized_expr.hpp"

//Forward value classes for expressions
#include "value_fwd.hpp"

// The expressions building
#include "checks.hpp"
#include "builder/expression_builder.hpp"

// The evaluator and optimizer
#include "etl/evaluator.hpp"
#include "etl/optimizer.hpp"

// The value classes implementation
#include "etl/fast.hpp"
#include "etl/dyn.hpp"

// The traits
#include "traits.hpp"

// to_string support
#include "print.hpp"
