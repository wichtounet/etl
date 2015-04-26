//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_ETL_HPP
#define ETL_ETL_HPP

#include<complex>

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
#include "generator_expr.hpp"
#include "temporary_expr.hpp"

//Forward value classes for expressions
#include "value_fwd.hpp"

// The complex expressions
#include "mmul_expr.hpp"
#include "outer_product_expr.hpp"
#include "fft_expr.hpp"
#include "conv_expr.hpp"

// The expressions building
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

// to_string support
#include "print.hpp"

#endif
