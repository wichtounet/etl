//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief This file contains macros to abstract some incompatibilities between compilers
 */

#pragma once

//Workaround for use of this in noexcept specifier in gcc

#ifdef __clang__
#define noexcept_this(...) noexcept(__VA_ARGS__)
#else
//GCC is bugged (Bug: 52869), it is therefore impossible to use this in noexcept
#define noexcept_this(...)
#endif

//Stupid trick to allow auto functions to have debug symbols in some compilers

#ifdef __clang__
#define ETL_DEBUG_AUTO_TRICK template <typename E = void>
#else
#define ETL_DEBUG_AUTO_TRICK
#endif

//Only clang and GCC >=5 supports constexpr functions in their C++14 full fashion

#ifdef __clang__ //clang version
#define cpp14_constexpr constexpr
#elif __GNUC__ >= 5 //GCC5+ version
#define cpp14_constexpr constexpr
#else //other compilers
#define cpp14_constexpr
#endif

//Fix an assertion failed in Intel C++ Compiler

#ifdef __INTEL_COMPILER
#define intel_decltype_auto auto
#else
#define intel_decltype_auto decltype(auto)
#endif
