//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "bias_test.hpp"

// Tests for bias_add

BIAS_ADD_4D_TEST_CASE("bias_add/0", "[bias_add]") {
    etl::fast_matrix<T, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<T, 3> b{1, 2, 3};
    etl::fast_matrix<T, 2, 3, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0, 0, 0), T(a(0, 0, 0, 0) + 1));
    REQUIRE_EQUALS(c(0, 0, 0, 1), T(a(0, 0, 0, 1) + 1));
    REQUIRE_EQUALS(c(0, 0, 1, 0), T(a(0, 0, 1, 0) + 1));
    REQUIRE_EQUALS(c(0, 0, 1, 1), T(a(0, 0, 1, 1) + 1));

    REQUIRE_EQUALS(c(0, 1, 0, 0), T(a(0, 1, 0, 0) + 2));
    REQUIRE_EQUALS(c(0, 1, 0, 1), T(a(0, 1, 0, 1) + 2));
    REQUIRE_EQUALS(c(0, 1, 1, 0), T(a(0, 1, 1, 0) + 2));
    REQUIRE_EQUALS(c(0, 1, 1, 1), T(a(0, 1, 1, 1) + 2));

    REQUIRE_EQUALS(c(0, 2, 0, 0), T(a(0, 2, 0, 0) + 3));
    REQUIRE_EQUALS(c(0, 2, 0, 1), T(a(0, 2, 0, 1) + 3));
    REQUIRE_EQUALS(c(0, 2, 1, 0), T(a(0, 2, 1, 0) + 3));
    REQUIRE_EQUALS(c(0, 2, 1, 1), T(a(0, 2, 1, 1) + 3));

    REQUIRE_EQUALS(c(1, 0, 0, 0), T(a(1, 0, 0, 0) + 1));
    REQUIRE_EQUALS(c(1, 0, 0, 1), T(a(1, 0, 0, 1) + 1));
    REQUIRE_EQUALS(c(1, 0, 1, 0), T(a(1, 0, 1, 0) + 1));
    REQUIRE_EQUALS(c(1, 0, 1, 1), T(a(1, 0, 1, 1) + 1));

    REQUIRE_EQUALS(c(1, 1, 0, 0), T(a(1, 1, 0, 0) + 2));
    REQUIRE_EQUALS(c(1, 1, 0, 1), T(a(1, 1, 0, 1) + 2));
    REQUIRE_EQUALS(c(1, 1, 1, 0), T(a(1, 1, 1, 0) + 2));
    REQUIRE_EQUALS(c(1, 1, 1, 1), T(a(1, 1, 1, 1) + 2));

    REQUIRE_EQUALS(c(1, 2, 0, 0), T(a(1, 2, 0, 0) + 3));
    REQUIRE_EQUALS(c(1, 2, 0, 1), T(a(1, 2, 0, 1) + 3));
    REQUIRE_EQUALS(c(1, 2, 1, 0), T(a(1, 2, 1, 0) + 3));
    REQUIRE_EQUALS(c(1, 2, 1, 1), T(a(1, 2, 1, 1) + 3));
}

BIAS_ADD_2D_TEST_CASE("bias_add/1", "[bias_add]") {
    etl::fast_matrix<T, 2, 3> a({1, 2, 3, 4, 5, 6});
    etl::fast_matrix<T, 3> b{1, 2, 3};
    etl::fast_matrix<T, 2, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), T(a(0, 0) + 1));
    REQUIRE_EQUALS(c(0, 1), T(a(0, 1) + 2));
    REQUIRE_EQUALS(c(0, 2), T(a(0, 2) + 3));

    REQUIRE_EQUALS(c(1, 0), T(a(1, 0) + 1));
    REQUIRE_EQUALS(c(1, 1), T(a(1, 1) + 2));
    REQUIRE_EQUALS(c(1, 2), T(a(1, 2) + 3));
}
