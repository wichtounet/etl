//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Header with most of the features of the ETL library.
 */

#pragma once

//Common STL includes
#include "etl/std.hpp"

//Metaprogramming utilities
#include "etl/tmp.hpp"

// Utilities
#include "etl/impl_enums.hpp"
#include "etl/order.hpp"
#include "etl/sparse_storage.hpp"
#include "etl/config.hpp"
#include "etl/context.hpp"
#include "etl/parallel_session.hpp"
#include "etl/complex.hpp"
#include "etl/vectorization.hpp"
#include "etl/random.hpp"
#include "etl/duration.hpp"
#include "etl/threshold.hpp"
#include "etl/thread_engine.hpp"
#include "etl/parallel.hpp"
#include "etl/memory.hpp"
#include "etl/allocator.hpp"
#include "etl/iterator.hpp"
#include "etl/util/counters.hpp"

//Forward declarations
#include "etl/value_fwd.hpp"
#include "etl/expr_fwd.hpp"

// The traits
#include "etl/traits.hpp"

// Opaque memory container
#include "etl/gpu_handler.hpp"

// The operators
#include "etl/eval_visitors.hpp"  //Evaluation visitors
#include "etl/op/scalar.hpp"
#include "etl/op/generators.hpp"
#include "etl/op/transformers.hpp"
#include "etl/op/virtual_views.hpp"
#include "etl/op/unary_op.hpp"
#include "etl/op/binary_op.hpp"

//Global test functions
#include "etl/globals.hpp"

// The evaluator
#include "etl/evaluator.hpp"

// CRTP classes
#include "etl/crtp/assignable.hpp"
#include "etl/crtp/inplace_assignable.hpp"
#include "etl/crtp/value_testable.hpp"
#include "etl/crtp/dim_testable.hpp"
#include "etl/crtp/iterable.hpp"

// The complex expressions
#include "etl/expr/detail.hpp"
#include "etl/expr/transpose_expr.hpp"
#include "etl/expr/bias_batch_mean_expr.hpp"

// The expressions building
#include "etl/checks.hpp"
#include "etl/builder/expression_builder.hpp"

// The parallel utilies
#include "etl/parallel_support.hpp"

// The expressions
#include "etl/op/dim_view.hpp"
#include "etl/op/slice_view.hpp"
#include "etl/op/memory_slice_view.hpp"
#include "etl/op/sub_view.hpp"
#include "etl/op/sub_matrix_2d.hpp"
#include "etl/op/dyn_matrix_view.hpp"
#include "etl/op/fast_matrix_view.hpp"
#include "etl/expr/binary_expr.hpp"
#include "etl/expr/unary_expr.hpp"
#include "etl/expr/generator_expr.hpp"
#include "etl/expr/temporary_expr.hpp"
#include "etl/expr/optimized_expr.hpp"
#include "etl/expr/serial_expr.hpp"
#include "etl/expr/selected_expr.hpp"
#include "etl/expr/parallel_expr.hpp"
#include "etl/expr/timed_expr.hpp"

// The optimizer
#include "etl/optimizer.hpp"

// Necessary for the matrices
#include "etl/impl/direct_op.hpp"

// The value classes implementation
#include "etl/crtp/expression_able.hpp"
#include "etl/fast.hpp"
#include "etl/dyn.hpp"
#include "etl/sparse.hpp"
#include "etl/custom_dyn.hpp"
#include "etl/custom_fast.hpp"

// The adapters
#include "etl/adapters/symmetric.hpp"
#include "etl/adapters/hermitian.hpp"
#include "etl/adapters/diagonal.hpp"
#include "etl/adapters/lower.hpp"
#include "etl/adapters/strictly_lower.hpp"
#include "etl/adapters/uni_lower.hpp"
#include "etl/adapters/upper.hpp"
#include "etl/adapters/strictly_upper.hpp"
#include "etl/adapters/uni_upper.hpp"

// Serialization support
#include "etl/serializer.hpp"
#include "etl/deserializer.hpp"

// to_string support
#include "etl/print.hpp"
