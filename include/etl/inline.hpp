//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Inlining macros.
 */

#pragma once

#ifdef __clang__
#define ETL_INLINE_ATTR_VEC __attribute__((__always_inline__, __nodebug__))

#define ETL_INLINE(RRRR) inline RRRR __attribute__((__always_inline__, __nodebug__))
#define ETL_STRONG_INLINE(RRRR) inline RRRR __attribute__((__always_inline__))
#define ETL_STATIC_INLINE(RRRR) static ETL_INLINE(RRRR)
#define ETL_TMP_INLINE(RRRR) static inline RRRR __attribute__((__always_inline__, __nodebug__))
#define ETL_OUT_INLINE(RRRR) inline RRRR __attribute__((__always_inline__, __nodebug__))
#else
#define ETL_INLINE_ATTR_VEC __attribute__((__always_inline__, __artificial__))

#define ETL_INLINE(RRRR) inline RRRR __attribute__((__always_inline__, __gnu_inline__, __artificial__))
#define ETL_STRONG_INLINE(RRRR) inline RRRR __attribute__((__always_inline__, __gnu_inline__))
#define ETL_STATIC_INLINE(RRRR) static ETL_INLINE(RRRR)
#define ETL_TMP_INLINE(RRRR) static inline RRRR __attribute__((__always_inline__, __artificial__))
#define ETL_OUT_INLINE(RRRR) inline RRRR __attribute__((__always_inline__, __artificial__))
#endif
