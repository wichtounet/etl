//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("alias/1", "[alias]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a;
    etl::dyn_matrix<Z> b(3, 3);

    REQUIRE_DIRECT(a.alias(a));
    REQUIRE_DIRECT(b.alias(b));
    REQUIRE_DIRECT(!b.alias(a));
    REQUIRE_DIRECT(!a.alias(b));

    REQUIRE_DIRECT(a.alias(a + a));
    REQUIRE_DIRECT(a.alias(a + 1));
    REQUIRE_DIRECT(!a.alias(b + 1));
    REQUIRE_DIRECT((a + a).alias(a + a));
    REQUIRE_DIRECT(!a.alias(b >> b));

    REQUIRE_DIRECT(a.alias(a(0)));
    REQUIRE_DIRECT(a(0).alias(a(0)));
    REQUIRE_DIRECT(!a(0).alias(a(1)));
    REQUIRE_DIRECT(a.alias(a(0) + a(1)));
}

TEMPLATE_TEST_CASE_2("alias/traits/1", "[alias][traits]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    //Test linear operations
    REQUIRE_DIRECT(etl::decay_traits<decltype(a)>::is_linear);
    REQUIRE_DIRECT(!etl::decay_traits<decltype(a * a)>::is_linear);
    REQUIRE_DIRECT(etl::decay_traits<decltype((a >> a) + a - a / a)>::is_linear);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a)>::is_linear);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a(0))>::is_linear);
    REQUIRE_DIRECT(etl::decay_traits<decltype(transpose(a))>::is_linear);

    //Test non linear operations
    REQUIRE_DIRECT(!etl::decay_traits<decltype(fflip(a))>::is_linear);
    REQUIRE_DIRECT(!etl::decay_traits<decltype(a + fflip(a))>::is_linear);
}

TEMPLATE_TEST_CASE_2("alias/transpose/1", "[alias][transpose]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    a = transpose(a);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 2), 8.0);
    REQUIRE_EQUALS(a(2, 0), 3.0);
    REQUIRE_EQUALS(a(2, 1), 6.0);
    REQUIRE_EQUALS(a(2, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("alias/transpose/2", "[alias][transpose]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

    a = (transpose(a) >> 2.0) + (transpose(a) >> 3.0);

    REQUIRE_EQUALS(a(0, 0), 5.0);
    REQUIRE_EQUALS(a(0, 1), 20.0);
    REQUIRE_EQUALS(a(0, 2), 35.0);
    REQUIRE_EQUALS(a(1, 0), 10.0);
    REQUIRE_EQUALS(a(1, 1), 25.0);
    REQUIRE_EQUALS(a(1, 2), 40.0);
    REQUIRE_EQUALS(a(2, 0), 15.0);
    REQUIRE_EQUALS(a(2, 1), 30.0);
    REQUIRE_EQUALS(a(2, 2), 45.0);
}
