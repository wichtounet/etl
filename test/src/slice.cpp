//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

// Init tests

TEMPLATE_TEST_CASE_2("slice/1", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> test_matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto s1 = slice(test_matrix, 0, 1);

    REQUIRE(etl::size(s1) == 3);

    REQUIRE(etl::dim<0>(s1) == 1);
    REQUIRE(etl::dim<1>(s1) == 3);
    REQUIRE(etl::dimensions(s1) == 2);

    REQUIRE(s1(0, 0) == 1);
    REQUIRE(s1(0, 1) == 2);
    REQUIRE(s1(0, 2) == 3);

    auto s2 = slice(test_matrix, 1, 2);

    REQUIRE(etl::size(s2) == 3);

    REQUIRE(etl::dim<0>(s2) == 1);
    REQUIRE(etl::dim<1>(s2) == 3);
    REQUIRE(etl::dimensions(s2) == 2);

    REQUIRE(s2(0, 0) == 4);
    REQUIRE(s2(0, 1) == 5);
    REQUIRE(s2(0, 2) == 6);

    auto s3 = slice(test_matrix, 0, 2);

    REQUIRE(etl::size(s3) == 6);

    REQUIRE(etl::dim<0>(s3) == 2);
    REQUIRE(etl::dim<1>(s3) == 3);
    REQUIRE(etl::dimensions(s3) == 2);

    REQUIRE(s3(0, 0) == 1);
    REQUIRE(s3(0, 1) == 2);
    REQUIRE(s3(0, 2) == 3);
    REQUIRE(s3(1, 0) == 4);
    REQUIRE(s3(1, 1) == 5);
    REQUIRE(s3(1, 2) == 6);
}
