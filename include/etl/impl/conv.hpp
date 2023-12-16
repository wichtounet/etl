//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the convolution implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available. The selection of
 * parallel versus serial is also done at this level. The implementation
 * functions should never be used directly, only functions of this header can
 * be used directly.
 *
 * Ideas for improvements:
 *  * Parallel dispatching for SSE/AVX implementation is not perfect, it should be done inside the micro kernel main loop
 */

#pragma once

//Include the implementations
#include "etl/impl/std/conv.hpp"
#include "etl/impl/vec/conv.hpp"
#include "etl/impl/cudnn/conv.hpp"
#include "etl/impl/egblas/conv_1d.hpp"

#include "etl/impl/conv_select.hpp" // The selection functions

// All the descriptors
#include "etl/impl/conv_2d.hpp"
#include "etl/impl/conv_4d.hpp"
#include "etl/impl/conv_multi.hpp"
