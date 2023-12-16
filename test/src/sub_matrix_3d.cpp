//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("sub_matrix_3d/0", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    auto a_0 = sub(a, 0, 0, 0, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0), 4);
    REQUIRE_EQUALS(a_0(0, 1, 1), 5);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 2);
    REQUIRE_EQUALS(a_0[2], 4);
    REQUIRE_EQUALS(a_0[3], 5);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_3d/1", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    auto a_0 = sub(a, 0, 0, 0, 2, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 8UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0), 4);
    REQUIRE_EQUALS(a_0(0, 1, 1), 5);

    REQUIRE_EQUALS(a_0(1, 0, 0), 10);
    REQUIRE_EQUALS(a_0(1, 0, 1), 11);
    REQUIRE_EQUALS(a_0(1, 1, 0), 13);
    REQUIRE_EQUALS(a_0(1, 1, 1), 14);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 2);
    REQUIRE_EQUALS(a_0[2], 4);
    REQUIRE_EQUALS(a_0[3], 5);

    REQUIRE_EQUALS(a_0[4], 10);
    REQUIRE_EQUALS(a_0[5], 11);
    REQUIRE_EQUALS(a_0[6], 13);
    REQUIRE_EQUALS(a_0[7], 14);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_3d/2", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    auto a_0 = sub(a, 1, 1, 0, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::RowMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 13);
    REQUIRE_EQUALS(a_0(0, 0, 1), 14);
    REQUIRE_EQUALS(a_0(0, 1, 0), 16);
    REQUIRE_EQUALS(a_0(0, 1, 1), 17);

    REQUIRE_EQUALS(a_0[0], 13);
    REQUIRE_EQUALS(a_0[1], 14);
    REQUIRE_EQUALS(a_0[2], 16);
    REQUIRE_EQUALS(a_0[3], 17);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(1, 2, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_3d/cm/0", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    etl::fast_matrix_cm<Z, 2, 3, 3> a;
    a(0) = aa(0);
    a(1) = aa(1);

    auto a_0 = sub(a, 0, 0, 0, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0), 4);
    REQUIRE_EQUALS(a_0(0, 1, 1), 5);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 4);
    REQUIRE_EQUALS(a_0[2], 2);
    REQUIRE_EQUALS(a_0[3], 5);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_3d/cm/1", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    etl::fast_matrix_cm<Z, 2, 3, 3> a;
    a(0) = aa(0);
    a(1) = aa(1);

    auto a_0 = sub(a, 0, 0, 0, 2, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 8UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 1);
    REQUIRE_EQUALS(a_0(0, 0, 1), 2);
    REQUIRE_EQUALS(a_0(0, 1, 0), 4);
    REQUIRE_EQUALS(a_0(0, 1, 1), 5);

    REQUIRE_EQUALS(a_0(1, 0, 0), 10);
    REQUIRE_EQUALS(a_0(1, 0, 1), 11);
    REQUIRE_EQUALS(a_0(1, 1, 0), 13);
    REQUIRE_EQUALS(a_0(1, 1, 1), 14);

    REQUIRE_EQUALS(a_0[0], 1);
    REQUIRE_EQUALS(a_0[1], 10);
    REQUIRE_EQUALS(a_0[2], 4);
    REQUIRE_EQUALS(a_0[3], 13);

    REQUIRE_EQUALS(a_0[4], 2);
    REQUIRE_EQUALS(a_0[5], 11);
    REQUIRE_EQUALS(a_0[6], 5);
    REQUIRE_EQUALS(a_0[7], 14);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(0, 1, 1), 42);
}

TEMPLATE_TEST_CASE_2("sub_matrix_3d/cm/2", "[sub]", Z, double, float) {
    etl::fast_matrix<Z, 2, 3, 3> aa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    etl::fast_matrix_cm<Z, 2, 3, 3> a;
    a(0) = aa(0);
    a(1) = aa(1);

    auto a_0 = sub(a, 1, 1, 0, 1, 2, 2);

    REQUIRE_EQUALS(etl::size(a_0), 4UL);
    REQUIRE_EQUALS(etl::dimensions(a_0), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a_0), 1UL);
    REQUIRE_EQUALS(etl::dim<1>(a_0), 2UL);
    REQUIRE_EQUALS(etl::dim<2>(a_0), 2UL);
    REQUIRE_DIRECT(etl::decay_traits<decltype(a_0)>::storage_order == etl::order::ColumnMajor);

    REQUIRE_EQUALS(a_0(0, 0, 0), 13);
    REQUIRE_EQUALS(a_0(0, 0, 1), 14);
    REQUIRE_EQUALS(a_0(0, 1, 0), 16);
    REQUIRE_EQUALS(a_0(0, 1, 1), 17);

    REQUIRE_EQUALS(a_0[0], 13);
    REQUIRE_EQUALS(a_0[1], 16);
    REQUIRE_EQUALS(a_0[2], 14);
    REQUIRE_EQUALS(a_0[3], 17);

    a_0(0, 1, 1) = 42;

    REQUIRE_EQUALS(a_0(0, 1, 1), 42);
    REQUIRE_EQUALS(a(1, 2, 1), 42);
}
