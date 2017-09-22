//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("fast_matrix/max", "fast_matrix::max", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = max(a, 1.0);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 1.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/min", "fast_matrix::min", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = min(a, 1.0);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

TEMPLATE_TEST_CASE_2("pow/0", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {-1.0, 2.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 4> d;

    d = pow(a, 2);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
    REQUIRE_EQUALS(d[4], 4.0);
    REQUIRE_EQUALS(d[5], 16.0);
    REQUIRE_EQUALS(d[6], 25.0);
    REQUIRE_EQUALS(d[7], 36.0);
}

TEMPLATE_TEST_CASE_2("pow/1", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;
    d = pow((a >> a) + 1.0, 2);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], 25.0);
    REQUIRE_EQUALS(d[2], 1.0);
    REQUIRE_EQUALS(d[3], 4.0);
}

TEMPLATE_TEST_CASE_2("pow/2", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;
    d = pow(a, 2.0);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

TEMPLATE_TEST_CASE_2("pow_int/0", "[fast][pow_int]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {-1.0, 2.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 4> d;

    d = pow_int(a, 2);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
    REQUIRE_EQUALS(d[4], 4.0);
    REQUIRE_EQUALS(d[5], 16.0);
    REQUIRE_EQUALS(d[6], 25.0);
    REQUIRE_EQUALS(d[7], 36.0);
}

TEMPLATE_TEST_CASE_2("pow_int/1", "[fast][pow_int]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;
    d = pow_int((a >> a) + 1.0, 2);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], 25.0);
    REQUIRE_EQUALS(d[2], 1.0);
    REQUIRE_EQUALS(d[3], 4.0);
}
