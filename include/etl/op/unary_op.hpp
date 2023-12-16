//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the unary operators for the unary expression
 *
 * A unary operator is a simple class with a static function apply that
 * computes its result. If the operator is vectorizable, it also contains a
 * static function load that computes the result for several operands at a
 * time.
 */

#pragma once

#include <functional>
#include <ctime>

#include "etl/math.hpp"
#include "etl/temporary.hpp"

#include "etl/op/unary/minus.hpp"
#include "etl/op/unary/plus.hpp"
#include "etl/op/unary/abs.hpp"
#include "etl/op/unary/floor.hpp"
#include "etl/op/unary/ceil.hpp"
#include "etl/op/unary/log.hpp"
#include "etl/op/unary/log2.hpp"
#include "etl/op/unary/log10.hpp"
#include "etl/op/unary/sqrt.hpp"
#include "etl/op/unary/invsqrt.hpp"
#include "etl/op/unary/cbrt.hpp"
#include "etl/op/unary/invcbrt.hpp"
#include "etl/op/unary/tan.hpp"
#include "etl/op/unary/sin.hpp"
#include "etl/op/unary/cos.hpp"
#include "etl/op/unary/tanh.hpp"
#include "etl/op/unary/sinh.hpp"
#include "etl/op/unary/cosh.hpp"
#include "etl/op/unary/exp.hpp"
#include "etl/op/unary/sigmoid.hpp"
#include "etl/op/unary/sign.hpp"
#include "etl/op/unary/softplus.hpp"
#include "etl/op/unary/real.hpp"
#include "etl/op/unary/imag.hpp"
#include "etl/op/unary/conj.hpp"
#include "etl/op/unary/relu.hpp"
#include "etl/op/unary/relu_derivative.hpp"
#include "etl/op/unary/bernoulli.hpp"
#include "etl/op/unary/noise.hpp"
#include "etl/op/unary/clip.hpp"
