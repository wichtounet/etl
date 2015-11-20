//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("globals/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a;

    REQUIRE(a.is_square());
    REQUIRE(is_square(a));

    REQUIRE(!a.is_rectangular());
    REQUIRE(!is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_square());
    REQUIRE(is_square(a + a));

    REQUIRE(!expr.is_rectangular());
    REQUIRE(!is_rectangular(a + a));
}

TEST_CASE("globals/2", "[globals]") {
    etl::fast_matrix<double, 3, 2> a;

    REQUIRE(!a.is_square());
    REQUIRE(!is_square(a));

    REQUIRE(a.is_rectangular());
    REQUIRE(is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_square());
    REQUIRE(!is_square(a + a));

    REQUIRE(expr.is_rectangular());
    REQUIRE(is_rectangular(a + a));
}

TEST_CASE("globals/3", "[globals]") {
    etl::fast_matrix<double, 3, 2, 2> a;

    REQUIRE(a.is_sub_square());
    REQUIRE(is_sub_square(a));

    REQUIRE(is_square(a(1)));

    REQUIRE(!a.is_sub_rectangular());
    REQUIRE(!is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_sub_square());
    REQUIRE(is_sub_square(a + a));

    REQUIRE(!expr.is_sub_rectangular());
    REQUIRE(!is_sub_rectangular(a + a));
}

TEST_CASE("globals/4", "[globals]") {
    etl::fast_matrix<double, 3, 2, 3> a;

    REQUIRE(!a.is_sub_square());
    REQUIRE(!is_sub_square(a));

    REQUIRE(is_rectangular(a(1)));

    REQUIRE(a.is_sub_rectangular());
    REQUIRE(is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_sub_square());
    REQUIRE(!is_sub_square(a + a));

    REQUIRE(expr.is_sub_rectangular());
    REQUIRE(is_sub_rectangular(a + a));
}
