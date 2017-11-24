//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("saxpy/0", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2.0) * x + y;

    REQUIRE_EQUALS(yy[0], Z(-1.0));
    REQUIRE_EQUALS(yy[1], Z(7.0));
    REQUIRE_EQUALS(yy[2], Z(0.5));
    REQUIRE_EQUALS(yy[3], Z(3.2));
}

TEMPLATE_TEST_CASE_2("saxpy/1", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x + Z(2.0) * y;

    REQUIRE_EQUALS(yy[0], Z(1.0));
    REQUIRE_EQUALS(yy[1], Z(8.0));
    REQUIRE_EQUALS(yy[2], Z(1.0));
    REQUIRE_EQUALS(yy[3], Z(3.4));
}

TEMPLATE_TEST_CASE_2("saxpy/2", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2) + y;

    REQUIRE_EQUALS(yy[0], Z(-1.0));
    REQUIRE_EQUALS(yy[1], Z(7.0));
    REQUIRE_EQUALS(yy[2], Z(0.5));
    REQUIRE_EQUALS(yy[3], Z(3.2));
}

TEMPLATE_TEST_CASE_2("saxpy/3", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x + y * Z(2);

    REQUIRE_EQUALS(yy[0], Z(1.0));
    REQUIRE_EQUALS(yy[1], Z(8.0));
    REQUIRE_EQUALS(yy[2], Z(1.0));
    REQUIRE_EQUALS(yy[3], Z(3.4));
}

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

TEMPLATE_TEST_CASE_2("pow_precise/0", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {-1.0, 2.0, 1.0, 1.0, 2.0, 4.0, 5.0, -6.0};
    etl::fast_matrix<Z, 2, 4> d;

    d = pow_precise(a, Z(2));

    REQUIRE_EQUALS_APPROX(d[0], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[1], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[2], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[3], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[4], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[5], Z(16.0));
    REQUIRE_EQUALS_APPROX(d[6], Z(25.0));
    REQUIRE_EQUALS_APPROX(d[7], Z(36.0));
}

TEMPLATE_TEST_CASE_2("pow/0", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {0.1, 2.0, 1.0, 1.0, 2.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 4> d;

    d = pow(a, Z(2));

    REQUIRE_EQUALS_APPROX(d[0], Z(0.01));
    REQUIRE_EQUALS_APPROX(d[1], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[2], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[3], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[4], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[5], Z(16.0));
    REQUIRE_EQUALS_APPROX(d[6], Z(25.0));
    REQUIRE_EQUALS_APPROX(d[7], Z(36.0));
}

TEMPLATE_TEST_CASE_2("pow/1", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    d = pow((a >> a) + 1.0, Z(2));

    REQUIRE_EQUALS_APPROX(d[0], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[1], Z(25.0));
    REQUIRE_EQUALS_APPROX(d[2], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[3], Z(4.0));
}

TEMPLATE_TEST_CASE_2("pow/2", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {0.1, 2.0, 0.5, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    d = pow(a, Z(2));

    REQUIRE_EQUALS_APPROX(d[0], Z(0.01));
    REQUIRE_EQUALS_APPROX(d[1], Z(4.0));
    REQUIRE_EQUALS_APPROX(d[2], Z(0.25));
    REQUIRE_EQUALS_APPROX(d[3], Z(1.0));
}

TEMPLATE_TEST_CASE_2("pow/3", "[fast][pow]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {0.01, 2.0, 9.0, 1.0, 2.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 4> d;

    d = pow(a, -2.0);

    REQUIRE_EQUALS_APPROX(d[0], Z(10000.0));
    REQUIRE_EQUALS_APPROX(d[1], Z(1.0) / Z(4.0));
    REQUIRE_EQUALS_APPROX(d[2], Z(1.0) / Z(81.0));
    REQUIRE_EQUALS_APPROX(d[3], Z(1.0));
    REQUIRE_EQUALS_APPROX(d[4], Z(1.0) / Z(4.0));
    REQUIRE_EQUALS_APPROX(d[5], Z(1.0) / Z(16.0));
    REQUIRE_EQUALS_APPROX(d[6], Z(1.0) / Z(25.0));
    REQUIRE_EQUALS_APPROX(d[7], Z(1.0) / Z(36.0));
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
