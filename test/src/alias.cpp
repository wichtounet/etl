//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("alias/1", "[alias]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a;
    etl::dyn_matrix<Z> b(3, 3);

    REQUIRE(a.alias(a));
    REQUIRE(b.alias(b));
    REQUIRE(!b.alias(a));
    REQUIRE(!a.alias(b));

    REQUIRE(a.alias(a + a));
    REQUIRE(a.alias(a + 1));
    REQUIRE(!a.alias(b + 1));
    REQUIRE((a + a).alias(a + a));
    REQUIRE(!a.alias(b >> b));

    REQUIRE(a.alias(a(0)));
    REQUIRE(a(0).alias(a(0)));
    REQUIRE(!a(0).alias(a(1)));
    REQUIRE(a.alias(a(0) + a(1)));
}

TEMPLATE_TEST_CASE_2("alias/traits/1", "[alias][traits]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    //Test linear operations
    REQUIRE(etl::decay_traits<decltype(a)>::is_linear);
    REQUIRE(etl::decay_traits<decltype(a * a)>::is_linear);
    REQUIRE(etl::decay_traits<decltype((a >> a) + a - a / a)>::is_linear);
    REQUIRE(etl::decay_traits<decltype(a)>::is_linear);
    REQUIRE(etl::decay_traits<decltype(a(0))>::is_linear);

    //Test non linear operations
    REQUIRE(!etl::decay_traits<decltype(transpose(a))>::is_linear);
    REQUIRE(!etl::decay_traits<decltype(fflip(a))>::is_linear);
    REQUIRE(!etl::decay_traits<decltype(a + fflip(a))>::is_linear);
}
