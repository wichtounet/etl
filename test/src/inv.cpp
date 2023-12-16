//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

ETL_TEST_CASE("inv/1", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE_EQUALS(c[0], 0.0);
    REQUIRE_EQUALS(c[1], 0.0);
    REQUIRE_EQUALS(c[2], 1.0);
    REQUIRE_EQUALS(c[3], 1.0);
    REQUIRE_EQUALS(c[4], 0.0);
    REQUIRE_EQUALS(c[5], 0.0);
    REQUIRE_EQUALS(c[6], 0.0);
    REQUIRE_EQUALS(c[7], 1.0);
    REQUIRE_EQUALS(c[8], 0.0);
}

ETL_TEST_CASE("inv/2", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE_EQUALS(c[0], 1.0);
    REQUIRE_EQUALS(c[1], 0.0);
    REQUIRE_EQUALS(c[2], 0.0);
    REQUIRE_EQUALS(c[3], 0.0);
    REQUIRE_EQUALS(c[4], 1.0);
    REQUIRE_EQUALS(c[5], 0.0);
    REQUIRE_EQUALS(c[6], 0.0);
    REQUIRE_EQUALS(c[7], 0.0);
    REQUIRE_EQUALS(c[8], 1.0);
}

ETL_TEST_CASE("inv/3", "[inv]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 2.0, 3.0};
    etl::fast_matrix<double, 2, 2> c;

    c = inv(a);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 0.0);
    REQUIRE_EQUALS_APPROX(c[2], -0.6666667);
    REQUIRE_EQUALS_APPROX(c[3], 0.3333333);
}

ETL_TEST_CASE("inv/4", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 0.0);
    REQUIRE_EQUALS_APPROX(c[2], 0.0);

    REQUIRE_EQUALS_APPROX(c[3], -0.66667);
    REQUIRE_EQUALS_APPROX(c[4], 0.333333);
    REQUIRE_EQUALS_APPROX(c[5], 0.0);

    REQUIRE_EQUALS_APPROX(c[6], -0.1111111);
    REQUIRE_EQUALS_APPROX(c[7], -0.2777778);
    REQUIRE_EQUALS_APPROX(c[8], 0.1666667);
}

ETL_TEST_CASE("inv/5", "[inv]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 0.0, 3.0};
    etl::fast_matrix<double, 2, 2> c;

    c = inv(a);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], -0.6666667);
    REQUIRE_EQUALS_APPROX(c[2], 0.0);
    REQUIRE_EQUALS_APPROX(c[3], 0.3333333);
}

ETL_TEST_CASE("inv/6", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], -0.5);
    REQUIRE_EQUALS_APPROX(c[2], -0.083333);

    REQUIRE_EQUALS_APPROX(c[3], 0.0);
    REQUIRE_EQUALS_APPROX(c[4], 0.250);
    REQUIRE_EQUALS_APPROX(c[5], -0.208333);

    REQUIRE_EQUALS_APPROX(c[6], 0.0);
    REQUIRE_EQUALS_APPROX(c[7], 0.0);
    REQUIRE_EQUALS_APPROX(c[8], 0.1666667);
}

ETL_TEST_CASE("inv/7", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE_EQUALS_APPROX(c[0], -0.33333);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], -0.666667);

    REQUIRE_EQUALS_APPROX(c[3], -1.33333);
    REQUIRE_EQUALS_APPROX(c[4], -2.0);
    REQUIRE_EQUALS_APPROX(c[5], 2.33333);

    REQUIRE_EQUALS_APPROX(c[6], 1.33333);
    REQUIRE_EQUALS_APPROX(c[7], 1.0);
    REQUIRE_EQUALS_APPROX(c[8], -1.33333);
}
