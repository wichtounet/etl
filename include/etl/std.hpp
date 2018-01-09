//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// STL
#include <complex>
#include <vector>
#include <array>
#include <algorithm>   //For value_testable
#include <iosfwd>      //For stream support
#include <type_traits> //For static assertions tests
#include <tuple>       //For TMP stuff
#include <thread>

// cpp_utils
#include "cpp_utils/compat.hpp"
#include "cpp_utils/tmp.hpp"
#include "cpp_utils/likely.hpp"
#include "cpp_utils/assert.hpp"
#include "cpp_utils/parallel.hpp"

// Macro to handle noexcept and cpp_assert

namespace etl {

#ifdef NDEBUG
constexpr bool assert_nothrow = true;
#else
#ifdef CPP_UTILS_ASSERT_EXCEPTION
constexpr bool assert_nothrow = false;
#else
constexpr bool assert_nothrow = true;
#endif
#endif

/*!
 * \brief Alignment flag to aligned expressions
 *
 * This can be used to make expressions more clear.
 */
constexpr bool aligned = false;

/*!
 * \brief Alignment flag to unaligned expressions.
 *
 * This can be used to make expressions more clear.
 */
constexpr bool unaligned = false;

/*!
 * \brief The current major version number of the library
 */
constexpr size_t version_major = 1;

/*!
 * \brief The current minor version number of the library
 */
constexpr size_t version_minor = 3;

/*!
 * \brief The current revision version number of the library
 */
constexpr size_t version_revision = 0;

} //end of namespace etl

/*!
 * \brief String representation of the current version of the library.
 */
#define ETL_VERSION_STR "1.3.0"

/*!
 * \brief The current major version number of the library
 */
#define ETL_VERSION_MAJOR 1

/*!
 * \brief The current minor version number of the library
 */
#define ETL_VERSION_MINOR 3

/*!
 * \brief The current revision version number of the library
 */
#define ETL_VERSION_REVISION 0
