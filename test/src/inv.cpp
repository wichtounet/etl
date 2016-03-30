//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("inv/1", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE(c[0] == 0.0);
    REQUIRE(c[1] == 0.0);
    REQUIRE(c[2] == 1.0);
    REQUIRE(c[3] == 1.0);
    REQUIRE(c[4] == 0.0);
    REQUIRE(c[5] == 0.0);
    REQUIRE(c[6] == 0.0);
    REQUIRE(c[7] == 1.0);
    REQUIRE(c[8] == 0.0);
}

TEST_CASE("inv/2", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE(c[0] == 1.0);
    REQUIRE(c[1] == 0.0);
    REQUIRE(c[2] == 0.0);
    REQUIRE(c[3] == 0.0);
    REQUIRE(c[4] == 1.0);
    REQUIRE(c[5] == 0.0);
    REQUIRE(c[6] == 0.0);
    REQUIRE(c[7] == 0.0);
    REQUIRE(c[8] == 1.0);
}

TEST_CASE("inv/3", "[inv]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 2.0, 3.0};
    etl::fast_matrix<double, 2, 2> c;

    c = inv(a);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(0.0));
    REQUIRE(c[2] == Approx(-0.6666667));
    REQUIRE(c[3] == Approx(0.3333333));
}

TEST_CASE("inv/4", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(0.0));
    REQUIRE(c[2] == Approx(0.0));

    REQUIRE(c[3] == Approx(-0.66667));
    REQUIRE(c[4] == Approx(0.333333));
    REQUIRE(c[5] == Approx(0.0));

    REQUIRE(c[6] == Approx(-0.1111111));
    REQUIRE(c[7] == Approx(-0.2777778));
    REQUIRE(c[8] == Approx(0.1666667));
}

TEST_CASE("inv/5", "[inv]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 0.0, 3.0};
    etl::fast_matrix<double, 2, 2> c;

    c = inv(a);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(-0.6666667));
    REQUIRE(c[2] == Approx(0.0));
    REQUIRE(c[3] == Approx(0.3333333));
}

TEST_CASE("inv/6", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(-0.5));
    REQUIRE(c[2] == Approx(-0.083333));

    REQUIRE(c[3] == Approx(0.0));
    REQUIRE(c[4] == Approx(0.250));
    REQUIRE(c[5] == Approx(-0.208333));

    REQUIRE(c[6] == Approx(0.0));
    REQUIRE(c[7] == Approx(0.0));
    REQUIRE(c[8] == Approx(0.1666667));
}

TEST_CASE("inv/7", "[inv]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<double, 3, 3> c;

    c = inv(a);

    REQUIRE(c[0] == Approx(-0.33333));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(-0.666667));

    REQUIRE(c[3] == Approx(-1.33333));
    REQUIRE(c[4] == Approx(-2.0));
    REQUIRE(c[5] == Approx(2.33333));

    REQUIRE(c[6] == Approx(1.33333));
    REQUIRE(c[7] == Approx(1.0));
    REQUIRE(c[8] == Approx(-1.33333));
}
