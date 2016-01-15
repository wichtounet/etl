//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// STL
#include <complex>
#include <vector>
#include <array>
#include <algorithm>   //For std::find_if
#include <iosfwd>      //For stream support
#include <type_traits> //For static assertions tests
#include <tuple>     //For TMP stuff
#include <thread>

// cpp_utils
#include "cpp_utils/tmp.hpp"
#include "cpp_utils/likely.hpp"
#include "cpp_utils/assert.hpp"
#include "cpp_utils/parallel.hpp"
