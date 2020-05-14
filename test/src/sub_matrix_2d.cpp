//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("sub_matrix_2d/0", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto a_0 = sub(a, 0, 0, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 1), 2);
    REQUIRE_EQUALS(a_0(1, 0), 4);
    REQUIRE_EQUALS(a_0(1, 1), 5);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 2);
    REQUIRE_EQUALS(a_0[2], 4);
    REQUIRE_EQUALS(a_0[3], 5);

    a_0(1, 1) = 99;

    REQUIRE_EQUALS(a_0(1, 1), 99);
    REQUIRE_EQUALS(a(1, 1), 99);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/1", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto a_0 = sub(a, 1, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);

    REQUIRE_EQUALS(a_0(0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1), 6);
    REQUIRE_EQUALS(a_0(1, 0), 8);
    REQUIRE_EQUALS(a_0(1, 1), 9);

    REQUIRE_EQUALS(a_0[0], 5);
    REQUIRE_EQUALS(a_0[1], 6);
    REQUIRE_EQUALS(a_0[2], 8);
    REQUIRE_EQUALS(a_0[3], 9);

    a_0(1, 1) = 99;

    REQUIRE_EQUALS(a_0(1, 1), 99);
    REQUIRE_EQUALS(a(2, 2), 99);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/2", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto a_0 = sub(a, 1, 1, 2, 2);

    a_0 = 0;

    REQUIRE_EQUALS(a_0(0, 0), 0);
    REQUIRE_EQUALS(a_0(0, 1), 0);
    REQUIRE_EQUALS(a_0(1, 0), 0);
    REQUIRE_EQUALS(a_0(1, 1), 0);

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 2);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(1, 0), 4);
    REQUIRE_EQUALS(a(1, 1), 0);
    REQUIRE_EQUALS(a(1, 2), 0);
    REQUIRE_EQUALS(a(2, 0), 7);
    REQUIRE_EQUALS(a(2, 1), 0);
    REQUIRE_EQUALS(a(2, 2), 0);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/3", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 4, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto a_0 = sub(a, 1, 1, 3, 2);

    REQUIRE_EQUALS(a_0(0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1), 6);
    REQUIRE_EQUALS(a_0(1, 0), 8);
    REQUIRE_EQUALS(a_0(1, 1), 9);
    REQUIRE_EQUALS(a_0(2, 0), 11);
    REQUIRE_EQUALS(a_0(2, 1), 12);

    REQUIRE_EQUALS(a_0[0], 5);
    REQUIRE_EQUALS(a_0[1], 6);
    REQUIRE_EQUALS(a_0[2], 8);
    REQUIRE_EQUALS(a_0[3], 9);
    REQUIRE_EQUALS(a_0[4], 11);
    REQUIRE_EQUALS(a_0[5], 12);

    a_0 = 0;

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 2);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(1, 0), 4);
    REQUIRE_EQUALS(a(1, 1), 0);
    REQUIRE_EQUALS(a(1, 2), 0);
    REQUIRE_EQUALS(a(2, 0), 7);
    REQUIRE_EQUALS(a(2, 1), 0);
    REQUIRE_EQUALS(a(2, 2), 0);
    REQUIRE_EQUALS(a(3, 0), 10);
    REQUIRE_EQUALS(a(3, 1), 0);
    REQUIRE_EQUALS(a(3, 2), 0);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/4", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    sub(a, 0, 0, 2, 2) = sub(a, 2, 2, 2, 2) + sub(a, 2, 0, 2, 2) + sub(a, 0, 2, 2, 2);

    REQUIRE_EQUALS(a(0, 0), 23);
    REQUIRE_EQUALS(a(0, 1), 26);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(0, 3), 4);
    REQUIRE_EQUALS(a(1, 0), 35);
    REQUIRE_EQUALS(a(1, 1), 38);
    REQUIRE_EQUALS(a(1, 2), 7);
    REQUIRE_EQUALS(a(1, 3), 8);
    REQUIRE_EQUALS(a(2, 0), 9);
    REQUIRE_EQUALS(a(2, 1), 10);
    REQUIRE_EQUALS(a(2, 2), 11);
    REQUIRE_EQUALS(a(2, 3), 12);
    REQUIRE_EQUALS(a(3, 0), 13);
    REQUIRE_EQUALS(a(3, 1), 14);
    REQUIRE_EQUALS(a(3, 2), 15);
    REQUIRE_EQUALS(a(3, 3), 16);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/cm/0", "[sub]", Z, double, float) {
    etl::fast_matrix_cm<Z, 3, 3> a = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    auto a_0 = sub(a, 0, 0, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 1), 2);
    REQUIRE_EQUALS(a_0(1, 0), 4);
    REQUIRE_EQUALS(a_0(1, 1), 5);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 4);
    REQUIRE_EQUALS(a_0[2], 2);
    REQUIRE_EQUALS(a_0[3], 5);

    a_0(1, 1) = 99;

    REQUIRE_EQUALS(a_0(1, 1), 99);
    REQUIRE_EQUALS(a(1, 1), 99);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/cm/1", "[sub]", Z, double, float) {
    etl::fast_matrix_cm<Z, 3, 3> a = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    auto a_0 = sub(a, 1, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);

    REQUIRE_EQUALS(a_0(0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1), 6);
    REQUIRE_EQUALS(a_0(1, 0), 8);
    REQUIRE_EQUALS(a_0(1, 1), 9);

    REQUIRE_EQUALS(a_0[0], 5);
    REQUIRE_EQUALS(a_0[1], 8);
    REQUIRE_EQUALS(a_0[2], 6);
    REQUIRE_EQUALS(a_0[3], 9);

    a_0(1, 1) = 99;

    REQUIRE_EQUALS(a_0(1, 1), 99);
    REQUIRE_EQUALS(a(2, 2), 99);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/cm/2", "[sub]", Z, double, float) {
    etl::fast_matrix_cm<Z, 3, 3> a = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    auto a_0 = sub(a, 1, 1, 2, 2);

    a_0 = 0;

    REQUIRE_EQUALS(a_0(0, 0), 0);
    REQUIRE_EQUALS(a_0(0, 1), 0);
    REQUIRE_EQUALS(a_0(1, 0), 0);
    REQUIRE_EQUALS(a_0(1, 1), 0);

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 2);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(1, 0), 4);
    REQUIRE_EQUALS(a(1, 1), 0);
    REQUIRE_EQUALS(a(1, 2), 0);
    REQUIRE_EQUALS(a(2, 0), 7);
    REQUIRE_EQUALS(a(2, 1), 0);
    REQUIRE_EQUALS(a(2, 2), 0);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/cm/3", "[sub]", Z, double, float) {
    etl::fast_matrix_cm<Z, 4, 3> a = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

    auto a_0 = sub(a, 1, 1, 3, 2);

    REQUIRE_EQUALS(a_0(0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1), 6);
    REQUIRE_EQUALS(a_0(1, 0), 8);
    REQUIRE_EQUALS(a_0(1, 1), 9);
    REQUIRE_EQUALS(a_0(2, 0), 11);
    REQUIRE_EQUALS(a_0(2, 1), 12);

    REQUIRE_EQUALS(a_0[0], 5);
    REQUIRE_EQUALS(a_0[1], 8);
    REQUIRE_EQUALS(a_0[2], 11);
    REQUIRE_EQUALS(a_0[3], 6);
    REQUIRE_EQUALS(a_0[4], 9);
    REQUIRE_EQUALS(a_0[5], 12);

    a_0 = 0;

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 2);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(1, 0), 4);
    REQUIRE_EQUALS(a(1, 1), 0);
    REQUIRE_EQUALS(a(1, 2), 0);
    REQUIRE_EQUALS(a(2, 0), 7);
    REQUIRE_EQUALS(a(2, 1), 0);
    REQUIRE_EQUALS(a(2, 2), 0);
    REQUIRE_EQUALS(a(3, 0), 10);
    REQUIRE_EQUALS(a(3, 1), 0);
    REQUIRE_EQUALS(a(3, 2), 0);
}

TEMPLATE_TEST_CASE_2("sub_matrix_2d/cm/4", "[sub]", Z, double, float) {
    etl::fast_matrix_cm<Z, 4, 4> a = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};

    sub(a, 0, 0, 2, 2) = sub(a, 2, 2, 2, 2) + sub(a, 2, 0, 2, 2) + sub(a, 0, 2, 2, 2);

    REQUIRE_EQUALS(a(0, 0), 23);
    REQUIRE_EQUALS(a(0, 1), 26);
    REQUIRE_EQUALS(a(0, 2), 3);
    REQUIRE_EQUALS(a(0, 3), 4);
    REQUIRE_EQUALS(a(1, 0), 35);
    REQUIRE_EQUALS(a(1, 1), 38);
    REQUIRE_EQUALS(a(1, 2), 7);
    REQUIRE_EQUALS(a(1, 3), 8);
    REQUIRE_EQUALS(a(2, 0), 9);
    REQUIRE_EQUALS(a(2, 1), 10);
    REQUIRE_EQUALS(a(2, 2), 11);
    REQUIRE_EQUALS(a(2, 3), 12);
    REQUIRE_EQUALS(a(3, 0), 13);
    REQUIRE_EQUALS(a(3, 1), 14);
    REQUIRE_EQUALS(a(3, 2), 15);
    REQUIRE_EQUALS(a(3, 3), 16);
}
