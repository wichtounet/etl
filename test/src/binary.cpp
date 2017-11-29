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

TEMPLATE_TEST_CASE_2("saxpby/0", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2.0) * x + Z(3.0) * y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(13.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(1.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(5.6));
}

TEMPLATE_TEST_CASE_2("saxpby/1", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2.0) + y * Z(3.0);

    REQUIRE_EQUALS_APPROX(yy[0], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(13.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(1.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(5.6));
}

TEMPLATE_TEST_CASE_2("saxpby/2", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2.0) * x + y * Z(3.0);

    REQUIRE_EQUALS_APPROX(yy[0], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(13.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(1.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(5.6));
}

TEMPLATE_TEST_CASE_2("saxpby/3", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2.0) + Z(3.0) * y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(13.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(1.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(5.6));
}

TEMPLATE_TEST_CASE_2("saxmy/0", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2.0) * (x >> y);

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxmy/1", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x >> y) * Z(2.0);

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxmy/2", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(2.0) * x) >> y;

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxmy/3", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x >> Z(2.0)) >> y;

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxmy/4", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x >> (y >> Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxmy/5", "[saxmy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x >> (Z(2.0) >> y);

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(12.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(2.4));
}

TEMPLATE_TEST_CASE_2("saxdy/0", "[saxdy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x / (Z(2.0) * y);

    REQUIRE_EQUALS(yy[0], Z(-0.5));
    REQUIRE_EQUALS(yy[1], Z(1.0 / 3.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("saxdy/1", "[saxdy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x / (y * Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-0.5));
    REQUIRE_EQUALS(yy[1], Z(1.0 / 3.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("saxdy/2", "[saxdy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(2.0) >> x) / y;

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(4.0 / 3.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(4.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("saxdy/3", "[saxdy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x >> Z(2.0)) / y;

    REQUIRE_EQUALS(yy[0], Z(-2.0));
    REQUIRE_EQUALS(yy[1], Z(4.0 / 3.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(4.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("saxdbpy/0", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x / (Z(2.0) + y);

    REQUIRE_EQUALS(yy[0], Z(-1.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(2.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("saxdbpy/1", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x / (y + Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-1.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(2.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("saxdbpy/2", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) * x) / (Z(2.0) + y);

    REQUIRE_EQUALS(yy[0], Z(2.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(-4.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(-2.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("saxdbpy/3", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x * Z(-2.0)) / (Z(2.0) + y);

    REQUIRE_EQUALS(yy[0], Z(2.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(-4.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(-2.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("saxdbpy/4", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) * x) / (y + Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(2.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(-4.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(-2.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("saxdbpy/5", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x * Z(-2.0)) / (y + Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(2.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(-4.0 / 5.0));
    REQUIRE_EQUALS(yy[2], Z(0.0));
    REQUIRE_EQUALS(yy[3], Z(-2.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("sapxdbpy/0", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) + x) / (Z(2.0) + y);

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 2.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("sapxdbpy/1", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / (Z(2.0) + y);

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 2.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("sapxdbpy/2", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) + x) / (y + Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 2.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("sapxdbpy/3", "[saxdbpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / (y + Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 3.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 2.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 3.2));
}

TEMPLATE_TEST_CASE_2("sapxdby/0", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) + x) / (Z(2.0) * y);

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 2.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 1.0));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("sapxdby/1", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (Z(-2.0) + x) / (y * Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 2.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 1.0));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("sapxdby/2", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / (Z(2.0) * y);

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 2.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 1.0));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("sapxdby/3", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / (y * Z(2.0));

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 2.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 1.0));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 2.4));
}

TEMPLATE_TEST_CASE_2("sapxdby/4", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / y;

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 1.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 0.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 1.2));
}

TEMPLATE_TEST_CASE_2("sapxdby/5", "[saxdby][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = (x + Z(-2.0)) / y;

    REQUIRE_EQUALS(yy[0], Z(-3.0 / 1.0));
    REQUIRE_EQUALS(yy[1], Z(0.0));
    REQUIRE_EQUALS(yy[2], Z(-2.0 / 0.5));
    REQUIRE_EQUALS(yy[3], Z(-1.0 / 1.2));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/0", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x - Z(2) * y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(-3.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-1.4));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/1", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x - y * Z(2);

    REQUIRE_EQUALS_APPROX(yy[0], Z(-3.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-1.4));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/2", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2) * x - y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(-3.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-0.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(0.8));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/3", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2) - y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(-3.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(1.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-0.5));
    REQUIRE_EQUALS_APPROX(yy[3], Z(0.8));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/4", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2) * x - Z(2) * y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-2.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-0.4));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/5", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = Z(2) * x - y * Z(2);

    REQUIRE_EQUALS_APPROX(yy[0], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-2.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-0.4));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/6", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2) - Z(2) * y;

    REQUIRE_EQUALS_APPROX(yy[0], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-2.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-0.4));
}

TEMPLATE_TEST_CASE_2("saxpy_minus/7", "[saxpy][fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> x = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> y = {1.0, 3.0, 0.5, 1.2};
    etl::fast_matrix<Z, 2, 2> yy;

    yy = x * Z(2) - y * Z(2);

    REQUIRE_EQUALS_APPROX(yy[0], Z(-4.0));
    REQUIRE_EQUALS_APPROX(yy[1], Z(-2.0));
    REQUIRE_EQUALS_APPROX(yy[2], Z(-1.0));
    REQUIRE_EQUALS_APPROX(yy[3], Z(-0.4));
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
