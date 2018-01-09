//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("slice/1", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto s1 = slice(a, 0, 1);

    REQUIRE_EQUALS(etl::size(s1), 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s1), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s1), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s1), 2UL);

    REQUIRE_EQUALS(s1(0, 0), 1);
    REQUIRE_EQUALS(s1(0, 1), 2);
    REQUIRE_EQUALS(s1(0, 2), 3);

    auto s2 = slice(a, 1, 2);

    REQUIRE_EQUALS(etl::size(s2), 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s2), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s2), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s2), 2UL);

    REQUIRE_EQUALS(s2(0, 0), 4);
    REQUIRE_EQUALS(s2(0, 1), 5);
    REQUIRE_EQUALS(s2(0, 2), 6);

    auto s3 = slice(a, 0, 2);

    REQUIRE_EQUALS(etl::size(s3), 6UL);

    REQUIRE_EQUALS(etl::dim<0>(s3), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(s3), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s3), 2UL);

    REQUIRE_EQUALS(s3(0, 0), 1);
    REQUIRE_EQUALS(s3(0, 1), 2);
    REQUIRE_EQUALS(s3(0, 2), 3);
    REQUIRE_EQUALS(s3(1, 0), 4);
    REQUIRE_EQUALS(s3(1, 1), 5);
    REQUIRE_EQUALS(s3(1, 2), 6);
}

TEMPLATE_TEST_CASE_2("slice/2", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 5, 8, 2, 3> a;

    auto s1 = slice(a, 3, 5);

    REQUIRE_EQUALS(etl::size(s1), 2UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s1), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(s1), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s1), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s1), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s1), 4UL);

    auto s2 = slice(a, 1, 2);

    REQUIRE_EQUALS(etl::size(s2), 1UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s2), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s2), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s2), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s2), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s2), 4UL);

    auto s3 = slice(a, 0, 5);

    REQUIRE_EQUALS(etl::size(s3), 5UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s3), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(s3), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s3), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s3), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s3), 4UL);
}

TEMPLATE_TEST_CASE_2("slice/3", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 1, 3> b;

    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE_EQUALS(b(0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("slice/4", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 1, 1, 3> b;

    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE_EQUALS(b(0, 0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 0, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("slice/5", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a;

    etl::slice(a, 0, 1) = 1.0;
    etl::slice(a, 1, 2) = 2.0;

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 1.0);
    REQUIRE_EQUALS(a(0, 0, 2), 1.0);
    REQUIRE_EQUALS(a(1, 0, 0), 2.0);
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
    REQUIRE_EQUALS(a(1, 0, 2), 2.0);

    etl::slice(a, 0, 2) = 3.0;

    REQUIRE_EQUALS(a(0, 0, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 1), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2), 3.0);
    REQUIRE_EQUALS(a(1, 0, 0), 3.0);
    REQUIRE_EQUALS(a(1, 0, 1), 3.0);
    REQUIRE_EQUALS(a(1, 0, 2), 3.0);
}

TEMPLATE_TEST_CASE_2("slice/6", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a;

    a.slice(0, 1) = 1.0;
    a.slice(1, 2) = 2.0;

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 1.0);
    REQUIRE_EQUALS(a(0, 0, 2), 1.0);
    REQUIRE_EQUALS(a(1, 0, 0), 2.0);
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
    REQUIRE_EQUALS(a(1, 0, 2), 2.0);

    a.slice(0, 2) = 3.0;

    REQUIRE_EQUALS(a(0, 0, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 1), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2), 3.0);
    REQUIRE_EQUALS(a(1, 0, 0), 3.0);
    REQUIRE_EQUALS(a(1, 0, 1), 3.0);
    REQUIRE_EQUALS(a(1, 0, 2), 3.0);
}

TEMPLATE_TEST_CASE_2("slice/7", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 4> a;

    a.slice(0, 2) = 1.0;
    a.slice(2, 4) = 2.0;

    REQUIRE_EQUALS(a(0), 1.0);
    REQUIRE_EQUALS(a(1), 1.0);
    REQUIRE_EQUALS(a(2), 2.0);
    REQUIRE_EQUALS(a(3), 2.0);

    a.slice(0, 3) = 3.0;

    REQUIRE_EQUALS(a(0), 3.0);
    REQUIRE_EQUALS(a(1), 3.0);
    REQUIRE_EQUALS(a(2), 3.0);
    REQUIRE_EQUALS(a(3), 2.0);
}

TEMPLATE_TEST_CASE_2("slice/8", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 4> a(1.0);
    etl::fast_matrix<Z, 4> b(2.0);
    etl::fast_matrix<Z, 2> c;

    c = etl::slice(a + b, 0, 2);

    REQUIRE_EQUALS(c(0), 3.0);
    REQUIRE_EQUALS(c(1), 3.0);
}

TEMPLATE_TEST_CASE_2("slice/9", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 4> a(1.0);

    a.slice(0, 2) += 1.0;
    a.slice(2, 4) += 2.0;

    REQUIRE_EQUALS(a(0), 2.0);
    REQUIRE_EQUALS(a(1), 2.0);
    REQUIRE_EQUALS(a(2), 3.0);
    REQUIRE_EQUALS(a(3), 3.0);
}

TEMPLATE_TEST_CASE_2("slice/cm/1", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a_rm = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    etl::fast_matrix_cm<Z, 2, 3> a;
    a = a_rm;

    auto s1 = slice(a, 0, 1);

    REQUIRE_EQUALS(etl::size(s1), 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s1), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s1), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s1), 2UL);

    REQUIRE_EQUALS(s1(0, 0), 1);
    REQUIRE_EQUALS(s1(0, 1), 2);
    REQUIRE_EQUALS(s1(0, 2), 3);

    auto s2 = slice(a, 1, 2);

    REQUIRE_EQUALS(etl::size(s2), 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s2), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s2), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s2), 2UL);

    REQUIRE_EQUALS(s2(0, 0), 4);
    REQUIRE_EQUALS(s2(0, 1), 5);
    REQUIRE_EQUALS(s2(0, 2), 6);

    auto s3 = slice(a, 0, 2);

    REQUIRE_EQUALS(etl::size(s3), 6UL);

    REQUIRE_EQUALS(etl::dim<0>(s3), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(s3), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s3), 2UL);

    REQUIRE_EQUALS(s3(0, 0), 1);
    REQUIRE_EQUALS(s3(0, 1), 2);
    REQUIRE_EQUALS(s3(0, 2), 3);
    REQUIRE_EQUALS(s3(1, 0), 4);
    REQUIRE_EQUALS(s3(1, 1), 5);
    REQUIRE_EQUALS(s3(1, 2), 6);
}

TEMPLATE_TEST_CASE_2("slice/cm/2", "[slice]", Z, float, double) {
    etl::fast_matrix_cm<Z, 5, 8, 2, 3> a;

    auto s1 = slice(a, 3, 5);

    REQUIRE_EQUALS(etl::size(s1), 2UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s1), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(s1), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s1), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s1), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s1), 4UL);

    auto s2 = slice(a, 1, 2);

    REQUIRE_EQUALS(etl::size(s2), 1UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s2), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(s2), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s2), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s2), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s2), 4UL);

    auto s3 = slice(a, 0, 5);

    REQUIRE_EQUALS(etl::size(s3), 5UL * 8UL * 2UL * 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s3), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(s3), 8UL);
    REQUIRE_EQUALS(etl::dim<2>(s3), 2UL);
    REQUIRE_EQUALS(etl::dim<3>(s3), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s3), 4UL);
}

TEMPLATE_TEST_CASE_2("slice/cm/3", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a_rm = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix_cm<Z, 2, 3> a;
    etl::fast_matrix_cm<Z, 1, 3> b;

    a = a_rm;
    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE_EQUALS(b(0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("slice/cm/4", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 3> a_rm = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    etl::fast_matrix_cm<Z, 2, 1, 3> a;
    a(0) = a_rm(0);
    a(1) = a_rm(1);

    etl::fast_matrix_cm<Z, 1, 1, 3> b;

    b = etl::slice(a, 0, 1) + etl::slice(a, 1, 2);

    REQUIRE_EQUALS(b(0, 0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 0, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("slice/cm/5", "[slice]", Z, float, double) {
    etl::fast_matrix_cm<Z, 2, 1, 3> a;

    etl::slice(a, 0, 1) = 1.0;
    etl::slice(a, 1, 2) = 2.0;

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 1.0);
    REQUIRE_EQUALS(a(0, 0, 2), 1.0);
    REQUIRE_EQUALS(a(1, 0, 0), 2.0);
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
    REQUIRE_EQUALS(a(1, 0, 2), 2.0);

    etl::slice(a, 0, 2) = 3.0;

    REQUIRE_EQUALS(a(0, 0, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 1), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2), 3.0);
    REQUIRE_EQUALS(a(1, 0, 0), 3.0);
    REQUIRE_EQUALS(a(1, 0, 1), 3.0);
    REQUIRE_EQUALS(a(1, 0, 2), 3.0);
}

TEMPLATE_TEST_CASE_2("slice/cm/6", "[slice]", Z, float, double) {
    etl::fast_matrix_cm<Z, 2, 1, 3> a;

    a.slice(0, 1) = 1.0;
    a.slice(1, 2) = 2.0;

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 1.0);
    REQUIRE_EQUALS(a(0, 0, 2), 1.0);
    REQUIRE_EQUALS(a(1, 0, 0), 2.0);
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
    REQUIRE_EQUALS(a(1, 0, 2), 2.0);

    a.slice(0, 2) = 3.0;

    REQUIRE_EQUALS(a(0, 0, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 1), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2), 3.0);
    REQUIRE_EQUALS(a(1, 0, 0), 3.0);
    REQUIRE_EQUALS(a(1, 0, 1), 3.0);
    REQUIRE_EQUALS(a(1, 0, 2), 3.0);
}
