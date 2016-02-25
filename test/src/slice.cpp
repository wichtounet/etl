//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("slice/1", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto s1 = slice(a, 0, 1);

    REQUIRE(etl::size(s1) == 3);

    REQUIRE(etl::dim<0>(s1) == 1);
    REQUIRE(etl::dim<1>(s1) == 3);
    REQUIRE(etl::dimensions(s1) == 2);

    REQUIRE(s1(0, 0) == 1);
    REQUIRE(s1(0, 1) == 2);
    REQUIRE(s1(0, 2) == 3);

    auto s2 = slice(a, 1, 2);

    REQUIRE(etl::size(s2) == 3);

    REQUIRE(etl::dim<0>(s2) == 1);
    REQUIRE(etl::dim<1>(s2) == 3);
    REQUIRE(etl::dimensions(s2) == 2);

    REQUIRE(s2(0, 0) == 4);
    REQUIRE(s2(0, 1) == 5);
    REQUIRE(s2(0, 2) == 6);

    auto s3 = slice(a, 0, 2);

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

TEMPLATE_TEST_CASE_2("slice/2", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 5, 8, 2, 3> a;

    auto s1 = slice(a, 3, 5);

    REQUIRE(etl::size(s1) == 2 * 8 * 2 * 3);

    REQUIRE(etl::dim<0>(s1) == 2);
    REQUIRE(etl::dim<1>(s1) == 8);
    REQUIRE(etl::dim<2>(s1) == 2);
    REQUIRE(etl::dim<3>(s1) == 3);
    REQUIRE(etl::dimensions(s1) == 4);

    auto s2 = slice(a, 1, 2);

    REQUIRE(etl::size(s2) == 1 * 8 * 2 * 3);

    REQUIRE(etl::dim<0>(s2) == 1);
    REQUIRE(etl::dim<1>(s2) == 8);
    REQUIRE(etl::dim<2>(s2) == 2);
    REQUIRE(etl::dim<3>(s2) == 3);
    REQUIRE(etl::dimensions(s2) == 4);

    auto s3 = slice(a, 0, 5);

    REQUIRE(etl::size(s3) == 5 * 8 * 2 * 3);

    REQUIRE(etl::dim<0>(s3) == 5);
    REQUIRE(etl::dim<1>(s3) == 8);
    REQUIRE(etl::dim<2>(s3) == 2);
    REQUIRE(etl::dim<3>(s3) == 3);
    REQUIRE(etl::dimensions(s3) == 4);
}

TEMPLATE_TEST_CASE_2("slice/3", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 1, 3> b;

    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE(b(0, 0) == 5.0);
    REQUIRE(b(0, 1) == 7.0);
    REQUIRE(b(0, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("slice/4", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 1, 1, 3> b;

    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE(b(0, 0, 0) == 5.0);
    REQUIRE(b(0, 0, 1) == 7.0);
    REQUIRE(b(0, 0, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("slice/5", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a;

    etl::slice(a, 0, 1) = 1.0;
    etl::slice(a, 1, 2) = 2.0;

    REQUIRE(a(0, 0, 0) == 1.0);
    REQUIRE(a(0, 0, 1) == 1.0);
    REQUIRE(a(0, 0, 2) == 1.0);
    REQUIRE(a(1, 0, 0) == 2.0);
    REQUIRE(a(1, 0, 1) == 2.0);
    REQUIRE(a(1, 0, 2) == 2.0);

    etl::slice(a, 0, 2) = 3.0;

    REQUIRE(a(0, 0, 0) == 3.0);
    REQUIRE(a(0, 0, 1) == 3.0);
    REQUIRE(a(0, 0, 2) == 3.0);
    REQUIRE(a(1, 0, 0) == 3.0);
    REQUIRE(a(1, 0, 1) == 3.0);
    REQUIRE(a(1, 0, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("slice/6", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a;

    a.slice(0, 1) = 1.0;
    a.slice(1, 2) = 2.0;

    REQUIRE(a(0, 0, 0) == 1.0);
    REQUIRE(a(0, 0, 1) == 1.0);
    REQUIRE(a(0, 0, 2) == 1.0);
    REQUIRE(a(1, 0, 0) == 2.0);
    REQUIRE(a(1, 0, 1) == 2.0);
    REQUIRE(a(1, 0, 2) == 2.0);

    a.slice(0, 2) = 3.0;

    REQUIRE(a(0, 0, 0) == 3.0);
    REQUIRE(a(0, 0, 1) == 3.0);
    REQUIRE(a(0, 0, 2) == 3.0);
    REQUIRE(a(1, 0, 0) == 3.0);
    REQUIRE(a(1, 0, 1) == 3.0);
    REQUIRE(a(1, 0, 2) == 3.0);
}
