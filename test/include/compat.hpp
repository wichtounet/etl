//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DOCTEST
#ifndef ETL_CATCH
#ifndef ETL_FAST_CATCH
#define ETL_FAST_CATCH
#endif
#endif
#endif

#define UNIQUE_NAME_LINE2( name, line ) name##line
#define UNIQUE_NAME_LINE( name, line ) UNIQUE_NAME_LINE2( name, line )
#define UNIQUE_NAME( name ) UNIQUE_NAME_LINE( name, __LINE__ )

#ifdef ETL_DOCTEST
#include <limits>
#include "doctest/doctest.h"
#else
#include "catch.hpp"
#endif

constexpr const auto base_eps = std::numeric_limits<float>::epsilon() * 100;

#ifndef ETL_DOCTEST
#include "fast_catch.hpp"
#endif

#ifdef ETL_DOCTEST
#define ETL_TEST_CASE(name, description) TEST_CASE(name)
#define ETL_SECTION(name) SUBCASE(name)
#else
#define ETL_TEST_CASE(name, description) TEST_CASE(name, description)
#define ETL_SECTION(name) SECTION(name)
#endif

#include "template_test.hpp"

#ifdef ETL_DOCTEST

namespace doctest {
namespace detail {

inline int fast_assert_unary_value(const char* assert_name, const char* file, int line, const char* expr,
                      const bool val, bool is_false) {
    ResultBuilder rb(assert_name, doctest::detail::assertType::normal, file, line, expr);
    try {
        rb.m_res.m_passed        = val;
        rb.m_res.m_decomposition = toString(val);
        if(is_false)
            rb.m_res.m_passed = !rb.m_res.m_passed;
    } catch(...) { rb.m_threw = true; }

    int res = 0;

    if(rb.log())
        res |= fastAssertAction::dbgbreak;

    if(rb.m_failed && checkIfShouldThrow(assert_name))
        res |= fastAssertAction::shouldthrow;

    return res;
}

}
}

#define DOCTEST_FAST_ASSERTION_UNARY_VALUE(assert_name, val, is_false)                                 \
    do {                                                                                               \
        int res = doctest::detail::fast_assert_unary_value(assert_name, __FILE__, __LINE__, #val, val, \
                                                           is_false);                                  \
        if (res & doctest::detail::fastAssertAction::dbgbreak)                                         \
            DOCTEST_BREAK_INTO_DEBUGGER();                                                             \
        if (res & doctest::detail::fastAssertAction::shouldthrow)                                      \
            doctest::detail::throwException();                                                         \
    } while (doctest::detail::always_false())

#define REQUIRE_UNARY_VALUE(val) DOCTEST_FAST_ASSERTION_UNARY_VALUE("REQUIRE_UNARY", val, false)

#define REQUIRE_DIRECT(value) { REQUIRE_UNARY_VALUE(value); }
#define REQUIRE_EQUALS(lhs, rhs) REQUIRE_EQ(lhs, rhs)
#define REQUIRE_EQUALS_APPROX(lhs, rhs) REQUIRE_EQ(lhs, doctest::Approx(rhs))
#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) REQUIRE_EQ(lhs, doctest::Approx(rhs).epsilon(eps))
#endif
