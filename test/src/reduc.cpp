//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// Tests for sum_r

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_1", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b(etl::sum_r(a));

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_2", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::sum_r(a);

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_3", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::sum_r(a) * 2.5;

    REQUIRE(b(0) == 25.0);
    REQUIRE(b(1) == 5.0);
    REQUIRE(b(2) == 55.0);
}

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_4", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = (etl::sum_r(a) - etl::sum_r(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

// Tests for mean_r

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_1", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b(etl::mean_r(a));

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_2", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_3", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a) * 2.5;

    REQUIRE(b(0) == 6.25);
    REQUIRE(b(1) == 1.25);
    REQUIRE(b(2) == 13.75);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_4", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = (etl::mean_r(a) - etl::mean_r(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_5", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 1, 1> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_6", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/dyn_matrix_1", "mean_r", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(etl::mean_r(a));

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/dyn_matrix_2", "mean_r", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(3);

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

// Tests for mean_l

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_1", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b(etl::mean_l(a));

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_2", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_l(a);

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_3", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_l(a) * 2.5;

    REQUIRE(b(0) == Approx(4.1666666));
    REQUIRE(b(1) == Approx(5.8333333));
    REQUIRE(b(2) == Approx(8.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_4", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = (etl::mean_l(a) - etl::mean_l(a)) + 2.5;

    REQUIRE(b(0) == Approx(Z(2.5)));
    REQUIRE(b(1) == Approx(Z(2.5)));
    REQUIRE(b(2) == Approx(Z(2.5)));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_5", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 1> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4, 1> b;

    b = etl::mean_l(a);

    REQUIRE(b(0, 0) == Approx(1.666666));
    REQUIRE(b(1, 0) == Approx(2.333333));
    REQUIRE(b(2, 0) == Approx(3.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_6", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4, 2> b;

    b = etl::mean_l(a);

    REQUIRE(b(0, 0) == Approx(9.0));
    REQUIRE(b(0, 1) == Approx(10.0));
    REQUIRE(b(1, 0) == Approx(11.0));
    REQUIRE(b(1, 1) == Approx(12.0));
    REQUIRE(b(2, 0) == Approx(13.0));
    REQUIRE(b(2, 1) == Approx(14.0));
    REQUIRE(b(3, 0) == Approx(15.0));
    REQUIRE(b(3, 1) == Approx(16.0));
}

TEMPLATE_TEST_CASE_2("mean_l/dyn_matrix_1", "mean_l", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(etl::mean_l(a));

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/dyn_matrix_2", "mean_l", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(4);

    b = etl::mean_l(a);

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_7", "mean_l/mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_r(etl::mean_l(a));

    REQUIRE(b(0) == Approx(9.5));
    REQUIRE(b(1) == Approx(11.5));
    REQUIRE(b(2) == Approx(13.5));
    REQUIRE(b(3) == Approx(15.5));
}

// Tests for sum_l

TEMPLATE_TEST_CASE_2("sum_l/fast_matrix_1", "sum_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b(etl::sum_l(a));

    REQUIRE(b(0) == Approx(5.0));
    REQUIRE(b(1) == Approx(7.0));
    REQUIRE(b(2) == Approx(10.0));
}

TEMPLATE_TEST_CASE_2("sum_l/fast_matrix_6", "sum_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4, 2> b;

    b = etl::sum_l(a);

    REQUIRE(b(0, 0) == Approx(27.0));
    REQUIRE(b(0, 1) == Approx(30.0));
    REQUIRE(b(1, 0) == Approx(33.0));
    REQUIRE(b(1, 1) == Approx(36.0));
    REQUIRE(b(2, 0) == Approx(39.0));
    REQUIRE(b(2, 1) == Approx(42.0));
    REQUIRE(b(3, 0) == Approx(45.0));
    REQUIRE(b(3, 1) == Approx(48.0));
}
