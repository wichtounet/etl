//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Restrict macros.
 */

#pragma once

#ifdef __GNUC__

#define ETL_RESTRICT __restrict

#elif defined(__clang__)

#define ETL_RESTRICT __restrict__

#else

#define ETL_RESTRICT

#endif
