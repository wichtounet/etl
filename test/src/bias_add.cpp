//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

// Tests for bias_add

TEMPLATE_TEST_CASE_2("bias_add/0", "[bias_add]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b{1, 2, 3};
    etl::fast_matrix<Z, 2, 3, 2, 2> c;

    c = etl::bias_add_4d(a, b);

    REQUIRE_EQUALS(c(0, 0, 0, 0), Z(a(0, 0, 0, 0) + 1));
    REQUIRE_EQUALS(c(0, 0, 0, 1), Z(a(0, 0, 0, 1) + 1));
    REQUIRE_EQUALS(c(0, 0, 1, 0), Z(a(0, 0, 1, 0) + 1));
    REQUIRE_EQUALS(c(0, 0, 1, 1), Z(a(0, 0, 1, 1) + 1));

    REQUIRE_EQUALS(c(0, 1, 0, 0), Z(a(0, 1, 0, 0) + 2));
    REQUIRE_EQUALS(c(0, 1, 0, 1), Z(a(0, 1, 0, 1) + 2));
    REQUIRE_EQUALS(c(0, 1, 1, 0), Z(a(0, 1, 1, 0) + 2));
    REQUIRE_EQUALS(c(0, 1, 1, 1), Z(a(0, 1, 1, 1) + 2));

    REQUIRE_EQUALS(c(0, 2, 0, 0), Z(a(0, 2, 0, 0) + 3));
    REQUIRE_EQUALS(c(0, 2, 0, 1), Z(a(0, 2, 0, 1) + 3));
    REQUIRE_EQUALS(c(0, 2, 1, 0), Z(a(0, 2, 1, 0) + 3));
    REQUIRE_EQUALS(c(0, 2, 1, 1), Z(a(0, 2, 1, 1) + 3));

    REQUIRE_EQUALS(c(1, 0, 0, 0), Z(a(1, 0, 0, 0) + 1));
    REQUIRE_EQUALS(c(1, 0, 0, 1), Z(a(1, 0, 0, 1) + 1));
    REQUIRE_EQUALS(c(1, 0, 1, 0), Z(a(1, 0, 1, 0) + 1));
    REQUIRE_EQUALS(c(1, 0, 1, 1), Z(a(1, 0, 1, 1) + 1));

    REQUIRE_EQUALS(c(1, 1, 0, 0), Z(a(1, 1, 0, 0) + 2));
    REQUIRE_EQUALS(c(1, 1, 0, 1), Z(a(1, 1, 0, 1) + 2));
    REQUIRE_EQUALS(c(1, 1, 1, 0), Z(a(1, 1, 1, 0) + 2));
    REQUIRE_EQUALS(c(1, 1, 1, 1), Z(a(1, 1, 1, 1) + 2));

    REQUIRE_EQUALS(c(1, 2, 0, 0), Z(a(1, 2, 0, 0) + 3));
    REQUIRE_EQUALS(c(1, 2, 0, 1), Z(a(1, 2, 0, 1) + 3));
    REQUIRE_EQUALS(c(1, 2, 1, 0), Z(a(1, 2, 1, 0) + 3));
    REQUIRE_EQUALS(c(1, 2, 1, 1), Z(a(1, 2, 1, 1) + 3));
}
