//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("sub_matrix_4d/0", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    auto a_0 = sub(a, 0, 0, 0, 0, 1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 6);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 2);
    REQUIRE_EQUALS(a_0[2], 5);
    REQUIRE_EQUALS(a_0[3], 6);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 0, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_4d/1", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    auto a_0 = sub(a,  0, 0, 1, 0,  1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 3);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 4);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 7);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 8);

    REQUIRE_EQUALS(a_0[0], 3);
    REQUIRE_EQUALS(a_0[1], 4);
    REQUIRE_EQUALS(a_0[2], 7);
    REQUIRE_EQUALS(a_0[3], 8);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_4d/2", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    auto a_0 = sub(a,  1, 0, 1, 0,  1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 15);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 16);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 19);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 20);

    REQUIRE_EQUALS(a_0[0], 15);
    REQUIRE_EQUALS(a_0[1], 16);
    REQUIRE_EQUALS(a_0[2], 19);
    REQUIRE_EQUALS(a_0[3], 20);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(1, 1, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_4d/cm/0", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    etl::fast_matrix_cm<Z, 2, 3, 2, 2> a;
    a(0)(0) = aa(0)(0);
    a(0)(1) = aa(0)(1);
    a(0)(2) = aa(0)(2);
    a(1)(0) = aa(1)(0);
    a(1)(1) = aa(1)(1);
    a(1)(2) = aa(1)(2);

    auto a_0 = sub(a, 0, 0, 0, 0, 1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 5);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 6);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 5);
    REQUIRE_EQUALS(a_0[2], 2);
    REQUIRE_EQUALS(a_0[3], 6);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 0, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_4d/cm/1", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    etl::fast_matrix_cm<Z, 2, 3, 2, 2> a;
    a(0)(0) = aa(0)(0);
    a(0)(1) = aa(0)(1);
    a(0)(2) = aa(0)(2);
    a(1)(0) = aa(1)(0);
    a(1)(1) = aa(1)(1);
    a(1)(2) = aa(1)(2);

    auto a_0 = sub(a,  0, 0, 1, 0,  1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 3);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 4);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 7);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 8);

    REQUIRE_EQUALS(a_0[0], 3);
    REQUIRE_EQUALS(a_0[1], 7);
    REQUIRE_EQUALS(a_0[2], 4);
    REQUIRE_EQUALS(a_0[3], 8);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_4d/cm/2", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 2, 2> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    etl::fast_matrix_cm<Z, 2, 3, 2, 2> a;
    a(0)(0) = aa(0)(0);
    a(0)(1) = aa(0)(1);
    a(0)(2) = aa(0)(2);
    a(1)(0) = aa(1)(0);
    a(1)(1) = aa(1)(1);
    a(1)(2) = aa(1)(2);

    auto a_0 = sub(a,  1, 0, 1, 0,  1, 2, 1, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<3>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0, 0), 15);
    REQUIRE_EQUALS(a_0(0, 0, 0, 1), 16);
    REQUIRE_EQUALS(a_0(0, 1, 0, 0), 19);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 20);

    REQUIRE_EQUALS(a_0[0], 15);
    REQUIRE_EQUALS(a_0[1], 19);
    REQUIRE_EQUALS(a_0[2], 16);
    REQUIRE_EQUALS(a_0[3], 20);

    a_0(0, 1, 0, 1) = 42;

    REQUIRE_EQUALS(a_0[3], 42);
    REQUIRE_EQUALS(a_0(0, 1, 0, 1), 42);
    REQUIRE_EQUALS(a(1, 1, 1, 1), 42);
}
