//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
#define DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include <limits>
#include "doctest/doctest.h"
#else
#include "catch.hpp"
#endif

constexpr auto base_eps           = std::numeric_limits<float>::epsilon() * 100;
constexpr auto base_eps_etl       = 0.0000001f;
constexpr auto base_eps_etl_large = 0.01f;

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

#define REQUIRE_DIRECT(value) FAST_CHECK_UNARY(value)
#define REQUIRE_EQUALS(lhs, rhs) FAST_CHECK_EQ(lhs, rhs)
#define REQUIRE_EQUALS_APPROX(lhs, rhs) FAST_CHECK_EQ(lhs, doctest::Approx(rhs))
#define REQUIRE_EQUALS_APPROX_E(lhs, rhs, eps) FAST_CHECK_EQ(lhs, doctest::Approx(rhs).epsilon(eps))

#endif
