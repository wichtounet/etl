//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains binary operators
 */

#pragma once

#include <functional>

#include "etl/math.hpp"
#include "etl/temporary.hpp"

#ifdef ETL_CUBLAS_MODE
#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"
#include "etl/impl/cublas/axpy.hpp"
#include "etl/impl/cublas/scal.hpp"
#endif

#include "etl/impl/egblas/apxdbpy.hpp"
#include "etl/impl/egblas/apxdbpy_3.hpp"
#include "etl/impl/egblas/apxdby.hpp"
#include "etl/impl/egblas/apxdby_3.hpp"
#include "etl/impl/egblas/axdbpy.hpp"
#include "etl/impl/egblas/axdbpy_3.hpp"
#include "etl/impl/egblas/axdy.hpp"
#include "etl/impl/egblas/axdy_3.hpp"
#include "etl/impl/egblas/axmy.hpp"
#include "etl/impl/egblas/axmy_3.hpp"
#include "etl/impl/egblas/axpby.hpp"
#include "etl/impl/egblas/axpby_3.hpp"
#include "etl/impl/egblas/axpy.hpp"
#include "etl/impl/egblas/axpy_3.hpp"

#include "etl/impl/egblas/scalar_add.hpp"
#include "etl/impl/egblas/scalar_div.hpp"
#include "etl/impl/egblas/scalar_mul.hpp"

#include "etl/op/binary/plus.hpp"
#include "etl/op/binary/minus.hpp"
#include "etl/op/binary/mul.hpp"
#include "etl/op/binary/div.hpp"
#include "etl/op/binary/mod.hpp"
#include "etl/op/binary/equal.hpp"
#include "etl/op/binary/not_equal.hpp"
#include "etl/op/binary/less.hpp"
#include "etl/op/binary/less_equal.hpp"
#include "etl/op/binary/greater.hpp"
#include "etl/op/binary/greater_equal.hpp"
#include "etl/op/binary/logical_and.hpp"
#include "etl/op/binary/logical_or.hpp"
#include "etl/op/binary/logical_xor.hpp"
#include "etl/op/binary/min.hpp"
#include "etl/op/binary/max.hpp"
#include "etl/op/binary/one_if.hpp"
#include "etl/op/binary/ranged_noise.hpp"
#include "etl/op/binary/sigmoid_derivative.hpp"
#include "etl/op/binary/relu_derivative.hpp"
#include "etl/op/binary/pow.hpp"
