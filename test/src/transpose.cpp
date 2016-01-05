//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("transpose/fast_matrix_1", "transpose", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 2, 3> b(transpose(a));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 3.0);
    REQUIRE(b(0, 2) == 0.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == 0.5);
    REQUIRE(b(1, 2) == -1);
}

TEMPLATE_TEST_CASE_2("transpose/fast_matrix_2", "transpose", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b(transpose(a));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 0.5);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == 0.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == -1);
}

TEMPLATE_TEST_CASE_2("transpose/fast_matrix_3", "transpose", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9});

    a.transpose_inplace();

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 4.0);
    REQUIRE(a(0, 2) == 7.0);
    REQUIRE(a(1, 0) == 2.0);
    REQUIRE(a(1, 1) == 5.0);
    REQUIRE(a(1, 2) == 8.0);
    REQUIRE(a(2, 0) == 3.0);
    REQUIRE(a(2, 1) == 6.0);
    REQUIRE(a(2, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("transpose/dyn_matrix_1", "transpose", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(transpose(a));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 3.0);
    REQUIRE(b(0, 2) == 0.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == 0.5);
    REQUIRE(b(1, 2) == -1);
}

TEMPLATE_TEST_CASE_2("transpose/dyn_matrix_2", "transpose", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(transpose(a));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 0.5);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == 0.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == -1);
}

TEMPLATE_TEST_CASE_2("transpose/dyn_matrix_3", "transpose", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9}));

    a.transpose_inplace();

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 4.0);
    REQUIRE(a(0, 2) == 7.0);
    REQUIRE(a(1, 0) == 2.0);
    REQUIRE(a(1, 1) == 5.0);
    REQUIRE(a(1, 2) == 8.0);
    REQUIRE(a(2, 0) == 3.0);
    REQUIRE(a(2, 1) == 6.0);
    REQUIRE(a(2, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("transpose/dyn_matrix_4", "transpose", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    a.transpose_inplace();

    REQUIRE(etl::dim<0>(a) == 2);
    REQUIRE(etl::dim<1>(a) == 3);

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 3.0);
    REQUIRE(a(0, 2) == 0.0);

    REQUIRE(a(1, 0) == -2.0);
    REQUIRE(a(1, 1) == 0.5);
    REQUIRE(a(1, 2) == -1.0);

    a.transpose_inplace();

    REQUIRE(etl::dim<0>(a) == 3);
    REQUIRE(etl::dim<1>(a) == 2);

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == -2.0);

    REQUIRE(a(1, 0) == 3.0);
    REQUIRE(a(1, 1) == 0.5);

    REQUIRE(a(2, 0) == 0.0);
    REQUIRE(a(2, 1) == -1.0);
}

TEMPLATE_TEST_CASE_2("transpose/dyn_matrix_5", "transpose", Z, float, double) {
    etl::dyn_matrix<Z> a(5, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));

    a.transpose_inplace();

    REQUIRE(etl::dim<0>(a) == 3);
    REQUIRE(etl::dim<1>(a) == 5);

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 4.0);
    REQUIRE(a(0, 2) == 7.0);
    REQUIRE(a(0, 3) == 10.0);
    REQUIRE(a(0, 4) == 13.0);

    REQUIRE(a(1, 0) == 2.0);
    REQUIRE(a(1, 1) == 5.0);
    REQUIRE(a(1, 2) == 8.0);
    REQUIRE(a(1, 3) == 11.0);
    REQUIRE(a(1, 4) == 14.0);

    REQUIRE(a(2, 0) == 3.0);
    REQUIRE(a(2, 1) == 6.0);
    REQUIRE(a(2, 2) == 9.0);
    REQUIRE(a(2, 3) == 12.0);
    REQUIRE(a(2, 4) == 15.0);

    a.transpose_inplace();

    REQUIRE(etl::dim<0>(a) == 5);
    REQUIRE(etl::dim<1>(a) == 3);

    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(0, 1) == 2.0);
    REQUIRE(a(0, 2) == 3.0);

    REQUIRE(a(1, 0) == 4.0);
    REQUIRE(a(1, 1) == 5.0);
    REQUIRE(a(1, 2) == 6.0);

    REQUIRE(a(2, 0) == 7.0);
    REQUIRE(a(2, 1) == 8.0);
    REQUIRE(a(2, 2) == 9.0);

    REQUIRE(a(3, 0) == 10.0);
    REQUIRE(a(3, 1) == 11.0);
    REQUIRE(a(3, 2) == 12.0);

    REQUIRE(a(4, 0) == 13.0);
    REQUIRE(a(4, 1) == 14.0);
    REQUIRE(a(4, 2) == 15.0);
}

TEMPLATE_TEST_CASE_2("transpose/expr_1", "transpose", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(3, 3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    a(1).transpose_inplace();

    REQUIRE(a(0, 0, 0) == 1.0);
    REQUIRE(a(0, 0, 1) == 2.0);
    REQUIRE(a(0, 0, 2) == 3.0);
    REQUIRE(a(0, 1, 0) == 4.0);
    REQUIRE(a(0, 1, 1) == 5.0);
    REQUIRE(a(0, 1, 2) == 6.0);
    REQUIRE(a(0, 2, 0) == 7.0);
    REQUIRE(a(0, 2, 1) == 8.0);
    REQUIRE(a(0, 2, 2) == 9.0);

    REQUIRE(a(1, 0, 0) == 1.0);
    REQUIRE(a(1, 0, 1) == 4.0);
    REQUIRE(a(1, 0, 2) == 7.0);
    REQUIRE(a(1, 1, 0) == 2.0);
    REQUIRE(a(1, 1, 1) == 5.0);
    REQUIRE(a(1, 1, 2) == 8.0);
    REQUIRE(a(1, 2, 0) == 3.0);
    REQUIRE(a(1, 2, 1) == 6.0);
    REQUIRE(a(1, 2, 2) == 9.0);

    REQUIRE(a(2, 0, 0) == 1.0);
    REQUIRE(a(2, 0, 1) == 2.0);
    REQUIRE(a(2, 0, 2) == 3.0);
    REQUIRE(a(2, 1, 0) == 4.0);
    REQUIRE(a(2, 1, 1) == 5.0);
    REQUIRE(a(2, 1, 2) == 6.0);
    REQUIRE(a(2, 2, 0) == 7.0);
    REQUIRE(a(2, 2, 1) == 8.0);
    REQUIRE(a(2, 2, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/1", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 3, 3, etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9));

    a.deep_transpose_inplace();

    REQUIRE(a(0, 0, 0) == 1.0);
    REQUIRE(a(0, 0, 1) == 4.0);
    REQUIRE(a(0, 0, 2) == 7.0);
    REQUIRE(a(0, 1, 0) == 2.0);
    REQUIRE(a(0, 1, 1) == 5.0);
    REQUIRE(a(0, 1, 2) == 8.0);
    REQUIRE(a(0, 2, 0) == 3.0);
    REQUIRE(a(0, 2, 1) == 6.0);
    REQUIRE(a(0, 2, 2) == 9.0);

    REQUIRE(a(1, 0, 0) == 1.0);
    REQUIRE(a(1, 0, 1) == 4.0);
    REQUIRE(a(1, 0, 2) == 7.0);
    REQUIRE(a(1, 1, 0) == 2.0);
    REQUIRE(a(1, 1, 1) == 5.0);
    REQUIRE(a(1, 1, 2) == 8.0);
    REQUIRE(a(1, 2, 0) == 3.0);
    REQUIRE(a(1, 2, 1) == 6.0);
    REQUIRE(a(1, 2, 2) == 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/2", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 4> a(2, 2, 3, 3, etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9));

    a.deep_transpose_inplace();

    REQUIRE(a(0, 0, 0, 0) == 1.0);
    REQUIRE(a(0, 0, 0, 1) == 4.0);
    REQUIRE(a(0, 0, 0, 2) == 7.0);
    REQUIRE(a(0, 0, 1, 0) == 2.0);
    REQUIRE(a(0, 0, 1, 1) == 5.0);
    REQUIRE(a(0, 0, 1, 2) == 8.0);
    REQUIRE(a(0, 0, 2, 0) == 3.0);
    REQUIRE(a(0, 0, 2, 1) == 6.0);
    REQUIRE(a(0, 0, 2, 2) == 9.0);

    REQUIRE(a(1, 1, 0, 0) == 1.0);
    REQUIRE(a(1, 1, 0, 1) == 4.0);
    REQUIRE(a(1, 1, 0, 2) == 7.0);
    REQUIRE(a(1, 1, 1, 0) == 2.0);
    REQUIRE(a(1, 1, 1, 1) == 5.0);
    REQUIRE(a(1, 1, 1, 2) == 8.0);
    REQUIRE(a(1, 1, 2, 0) == 3.0);
    REQUIRE(a(1, 1, 2, 1) == 6.0);
    REQUIRE(a(1, 1, 2, 2) == 9.0);
}
