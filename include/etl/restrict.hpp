//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

#else

#define ETL_RESTRICT

#endif
