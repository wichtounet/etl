//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains utilities for random generation
 */

#pragma once

#include <random>

namespace etl {

/*!
 * \brief The random engine used by the library
 */
using random_engine = std::mt19937_64;

} //end of namespace etl
