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
#include "scalar.hpp"
#include "generators.hpp"
#include "transformers.hpp"
#include "views.hpp"
#include "virtual_views.hpp"
#include "unary_op.hpp"
#include "binary_op.hpp"

// The expressions
#include "binary_expr.hpp"
#include "unary_expr.hpp"
#include "generator_expr.hpp"
#include "temporary_expr.hpp"

//Forward value classes for expressions
#include "value_fwd.hpp"

// The complex expressions
#include "mmul_expr.hpp"
#include "outer_product_expr.hpp"
#include "fft_expr.hpp"
#include "conv_expr.hpp"
#include "convmtx2_expr.hpp"

// The expressions building
#include "checks.hpp"
#include "expression_builder.hpp"
#include "conv_expression_builder.hpp"

// The value classes implementation
#include "etl/fast.hpp"
#include "etl/dyn.hpp"

// The traits
#include "traits.hpp"

// to_string support
#include "print.hpp"
