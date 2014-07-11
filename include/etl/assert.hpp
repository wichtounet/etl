//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_ASSERT_HPP
#define ETL_ASSERT_HPP

#include <iostream>

#include "likely.hpp"

#define etl_unused(x) ((void)x)

#ifdef NDEBUG

#define etl_assert(condition, message) ((void)0)

#if defined __clang__

#if __has_builtin(__builtin_unreachable)
#define etl_unreachable(message) __builtin_unreachable();
#else
#define etl_unreachable(message) ((void)0)
#endif //__has_builtin(__builtin_unreachable)

#elif defined __GNUC__

#define etl_unreachable(message) __builtin_unreachable();

#endif //__clang__

#else

#define etl_assert(condition, message) (likely(condition) \
    ? ((void)0) \
    : ::etl::assertion::detail::assertion_failed_msg(#condition, message, \
    __PRETTY_FUNCTION__, __FILE__, __LINE__))

#if defined __clang__

#if __has_builtin(__builtin_unreachable)
#define etl_unreachable(message) etl_assert(false, message); __builtin_unreachable();
#else
#define etl_unreachable(message) etl_assert(false, message);
#endif //__has_builtin(__builtin_unreachable)

#elif defined __GNUC__

#define etl_unreachable(message) etl_assert(false, message); __builtin_unreachable();

#endif //__clang__

#endif //NDEBUG

namespace etl {
namespace assertion {
namespace detail {

    template< typename CharT >
    void assertion_failed_msg(const CharT* expr, const char* msg, const char* function, const char* file, long line){
        std::cerr
            << "***** Internal Program Error - assertion (" << expr << ") failed in "
            << function << ":\n"
            << file << '(' << line << "): " << msg << std::endl;
        std::abort();
    }

} // end of detail namespace
} // end of assertion namespace
} // end of detail namespace

#endif //ETL_ASSERT_HPP
