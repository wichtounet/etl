//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

// Tests for sum_r

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_2", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::sum_r(a);

    REQUIRE_EQUALS(b(0), 10);
    REQUIRE_EQUALS(b(1), 2);
    REQUIRE_EQUALS(b(2), 22);
}

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_3", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::sum_r(a) * 2.5;

    REQUIRE_EQUALS(b(0), 25.0);
    REQUIRE_EQUALS(b(1), 5.0);
    REQUIRE_EQUALS(b(2), 55.0);
}

TEMPLATE_TEST_CASE_2("sum_r/fast_matrix_4", "sum_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = (etl::sum_r(a) - etl::sum_r(a)) + 2.5;

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 2.5);
    REQUIRE_EQUALS(b(2), 2.5);
}

// Tests for mean_r

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_2", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 0.5);
    REQUIRE_EQUALS(b(2), 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_3", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a) * 2.5;

    REQUIRE_EQUALS(b(0), 6.25);
    REQUIRE_EQUALS(b(1), 1.25);
    REQUIRE_EQUALS(b(2), 13.75);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_4", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = (etl::mean_r(a) - etl::mean_r(a)) + 2.5;

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 2.5);
    REQUIRE_EQUALS(b(2), 2.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_5", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 1, 1> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 0.5);
    REQUIRE_EQUALS(b(2), 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/fast_matrix_6", "mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 3> b;

    b = etl::mean_r(a);

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 0.5);
    REQUIRE_EQUALS(b(2), 5.5);
}

TEMPLATE_TEST_CASE_2("mean_r/dyn_matrix_2", "mean_r", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(3);

    b = etl::mean_r(a);

    REQUIRE_EQUALS(b(0), 2.5);
    REQUIRE_EQUALS(b(1), 0.5);
    REQUIRE_EQUALS(b(2), 5.5);
}

// Tests for mean_l

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_2", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_l(a);

    REQUIRE_EQUALS_APPROX(b(0), 1.666666);
    REQUIRE_EQUALS_APPROX(b(1), 2.333333);
    REQUIRE_EQUALS_APPROX(b(2), 3.333333);
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_3", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_l(a) * 2.5;

    REQUIRE_EQUALS_APPROX(b(0), 4.1666666);
    REQUIRE_EQUALS_APPROX(b(1), 5.8333333);
    REQUIRE_EQUALS_APPROX(b(2), 8.333333);
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_4", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;

    b = (etl::mean_l(a) - etl::mean_l(a)) + 2.5;

    REQUIRE_EQUALS_APPROX(b(0), Z(2.5));
    REQUIRE_EQUALS_APPROX(b(1), Z(2.5));
    REQUIRE_EQUALS_APPROX(b(2), Z(2.5));
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_5", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 1> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4, 1> b;

    b = etl::mean_l(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 1.666666);
    REQUIRE_EQUALS_APPROX(b(1, 0), 2.333333);
    REQUIRE_EQUALS_APPROX(b(2, 0), 3.333333);
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_6", "mean_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4, 2> b;

    b = etl::mean_l(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 9.0);
    REQUIRE_EQUALS_APPROX(b(0, 1), 10.0);
    REQUIRE_EQUALS_APPROX(b(1, 0), 11.0);
    REQUIRE_EQUALS_APPROX(b(1, 1), 12.0);
    REQUIRE_EQUALS_APPROX(b(2, 0), 13.0);
    REQUIRE_EQUALS_APPROX(b(2, 1), 14.0);
    REQUIRE_EQUALS_APPROX(b(3, 0), 15.0);
    REQUIRE_EQUALS_APPROX(b(3, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("mean_l/dyn_matrix_2", "mean_l", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 4, etl::values(1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7));
    etl::dyn_vector<Z> b(4);

    b = etl::mean_l(a);

    REQUIRE_EQUALS_APPROX(b(0), 1.666666);
    REQUIRE_EQUALS_APPROX(b(1), 2.333333);
    REQUIRE_EQUALS_APPROX(b(2), 3.333333);
}

TEMPLATE_TEST_CASE_2("mean_l/fast_matrix_7", "mean_l/mean_r", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4> b;

    b = etl::mean_r(etl::mean_l(a));

    REQUIRE_EQUALS_APPROX(b(0), 9.5);
    REQUIRE_EQUALS_APPROX(b(1), 11.5);
    REQUIRE_EQUALS_APPROX(b(2), 13.5);
    REQUIRE_EQUALS_APPROX(b(3), 15.5);
}

// Tests for sum_l

TEMPLATE_TEST_CASE_2("sum_l/fast_matrix_1", "sum_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4> a({1, 2, 3, 4, 0, 0, 1, 1, 4, 5, 6, 7});
    etl::fast_matrix<Z, 4> b;
    b = etl::sum_l(a);

    REQUIRE_EQUALS_APPROX(b(0), 5.0);
    REQUIRE_EQUALS_APPROX(b(1), 7.0);
    REQUIRE_EQUALS_APPROX(b(2), 10.0);
}

TEMPLATE_TEST_CASE_2("sum_l/fast_matrix_6", "sum_l", Z, float, double) {
    etl::fast_matrix<Z, 3, 4, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 4, 2> b;

    b = etl::sum_l(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 27.0);
    REQUIRE_EQUALS_APPROX(b(0, 1), 30.0);
    REQUIRE_EQUALS_APPROX(b(1, 0), 33.0);
    REQUIRE_EQUALS_APPROX(b(1, 1), 36.0);
    REQUIRE_EQUALS_APPROX(b(2, 0), 39.0);
    REQUIRE_EQUALS_APPROX(b(2, 1), 42.0);
    REQUIRE_EQUALS_APPROX(b(3, 0), 45.0);
    REQUIRE_EQUALS_APPROX(b(3, 1), 48.0);
}

// Tests for bias_batch_mean_2d

TEMPLATE_TEST_CASE_2("bias_batch_mean_2d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 4, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    etl::fast_matrix<Z, 3> b;

    b = etl::bias_batch_mean_2d(a);

    REQUIRE_EQUALS(b(0), Z(5.5));
    REQUIRE_EQUALS(b(1), Z(6.5));
    REQUIRE_EQUALS(b(2), Z(7.5));
}

// Tests for bias_batch_var_2d

TEMPLATE_TEST_CASE_2("bias_batch_var_2d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 4, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    etl::fast_matrix<Z, 3> b({0.0, 0.0, 0.0});
    etl::fast_matrix<Z, 3> c;

    c = etl::bias_batch_var_2d(a, b);

    REQUIRE_EQUALS_APPROX(c(0), Z(41.5));
    REQUIRE_EQUALS_APPROX(c(1), Z(53.5));
    REQUIRE_EQUALS_APPROX(c(2), Z(67.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_var_2d/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 4, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    etl::fast_matrix<Z, 3> b({0.1, -1.0, 2.0});
    etl::fast_matrix<Z, 3> c;

    c = etl::bias_batch_var_2d(a, b);

    REQUIRE_EQUALS_APPROX(c(0), Z(40.41));
    REQUIRE_EQUALS_APPROX(c(1), Z(67.5));
    REQUIRE_EQUALS_APPROX(c(2), Z(41.5));
}

// Tests for bias_batch_sum_2d

TEMPLATE_TEST_CASE_2("bias_batch_sum_2d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 4, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    etl::fast_matrix<Z, 3> b;

    b = etl::bias_batch_sum_2d(a);

    REQUIRE_EQUALS(b(0), Z(4 * 5.5));
    REQUIRE_EQUALS(b(1), Z(4 * 6.5));
    REQUIRE_EQUALS(b(2), Z(4 * 7.5));
}

// Tests for bias_batch_mean_4d

TEMPLATE_TEST_CASE_2("bias_batch_mean_4d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b;

    b = etl::bias_batch_mean_4d(a);

    REQUIRE_EQUALS(b(0), Z(8.5));
    REQUIRE_EQUALS(b(1), Z(12.5));
    REQUIRE_EQUALS(b(2), Z(16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_mean_4d/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(1.0);

    b += etl::bias_batch_mean_4d(a);

    REQUIRE_EQUALS(b(0), Z(1.0 + 8.5));
    REQUIRE_EQUALS(b(1), Z(1.0 + 12.5));
    REQUIRE_EQUALS(b(2), Z(1.0 + 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_mean_4d/2", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b -= etl::bias_batch_mean_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 - 8.5));
    REQUIRE_EQUALS(b(1), Z(2.0 - 12.5));
    REQUIRE_EQUALS(b(2), Z(2.0 - 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_mean_4d/3", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b *= etl::bias_batch_mean_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 * 8.5));
    REQUIRE_EQUALS(b(1), Z(2.0 * 12.5));
    REQUIRE_EQUALS(b(2), Z(2.0 * 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_mean_4d/4", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b /= etl::bias_batch_mean_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 / 8.5));
    REQUIRE_EQUALS(b(1), Z(2.0 / 12.5));
    REQUIRE_EQUALS(b(2), Z(2.0 / 16.5));
}

// Tests for bias_batch_sum_4d

TEMPLATE_TEST_CASE_2("bias_batch_sum_4d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b;

    b = etl::bias_batch_sum_4d(a);

    REQUIRE_EQUALS(b(0), Z(8 * 8.5));
    REQUIRE_EQUALS(b(1), Z(8 * 12.5));
    REQUIRE_EQUALS(b(2), Z(8 * 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_sum_4d/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(1.0);

    b += etl::bias_batch_sum_4d(a);

    REQUIRE_EQUALS(b(0), Z(1.0 + 8 * 8.5));
    REQUIRE_EQUALS(b(1), Z(1.0 + 8 * 12.5));
    REQUIRE_EQUALS(b(2), Z(1.0 + 8 * 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_sum_4d/2", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b -= etl::bias_batch_sum_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 - 8 * 8.5));
    REQUIRE_EQUALS(b(1), Z(2.0 - 8 * 12.5));
    REQUIRE_EQUALS(b(2), Z(2.0 - 8 * 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_sum_4d/3", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b *= etl::bias_batch_sum_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 * 8 * 8.5));
    REQUIRE_EQUALS(b(1), Z(2.0 * 8 * 12.5));
    REQUIRE_EQUALS(b(2), Z(2.0 * 8 * 16.5));
}

TEMPLATE_TEST_CASE_2("bias_batch_sum_4d/4", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    etl::fast_matrix<Z, 3> b(2.0);

    b /= etl::bias_batch_sum_4d(a);

    REQUIRE_EQUALS(b(0), Z(2.0 / (8 * 8.5)));
    REQUIRE_EQUALS(b(1), Z(2.0 / (8 * 12.5)));
    REQUIRE_EQUALS(b(2), Z(2.0 / (8 * 16.5)));
}

// Tests for bias_batch_var_4d

TEMPLATE_TEST_CASE_2("bias_batch_var_4d/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 8, 3, 4, 5, 6, 7, 8, 9, 2, 11, 3, 13, 3, 15, 7, 17, 4, 19, 9, 21, 0, 11, 24});
    etl::fast_matrix<Z, 3> mean;
    etl::fast_matrix<Z, 3> var;

    mean = etl::bias_batch_mean_4d(a);
    var  = etl::bias_batch_var_4d(a, mean);

    REQUIRE_EQUALS(mean(0), Z(6.75));
    REQUIRE_EQUALS(mean(1), Z(9.375));
    REQUIRE_EQUALS(mean(2), Z(10.125));

    REQUIRE_EQUALS_APPROX(var(0), Z(22.1875));
    REQUIRE_EQUALS_APPROX(var(1), Z(27.23438));
    REQUIRE_EQUALS_APPROX(var(2), Z(66.60938));
}

TEMPLATE_TEST_CASE_2("bias_batch_var_4d/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 8, 3, 4, 5, 6, 7, 8, 9, 2, 11, 3, 13, 3, 15, 7, 17, 4, 19, 9, 21, 0, 11, 24});
    etl::fast_matrix<Z, 3> mean;
    etl::fast_matrix<Z, 3> var;

    var = Z(10);
    mean = etl::bias_batch_mean_4d(a);
    var  += etl::bias_batch_var_4d(a, mean);

    REQUIRE_EQUALS(mean(0), Z(6.75));
    REQUIRE_EQUALS(mean(1), Z(9.375));
    REQUIRE_EQUALS(mean(2), Z(10.125));

    REQUIRE_EQUALS_APPROX(var(0), Z(32.1875));
    REQUIRE_EQUALS_APPROX(var(1), Z(37.23438));
    REQUIRE_EQUALS_APPROX(var(2), Z(76.60938));
}

TEMPLATE_TEST_CASE_2("bias_batch_var_4d/2", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 8, 3, 4, 5, 6, 7, 8, 9, 2, 11, 3, 13, 3, 15, 7, 17, 4, 19, 9, 21, 0, 11, 24});
    etl::fast_matrix<Z, 3> mean;
    etl::fast_matrix<Z, 3> var;

    var = Z(10);
    mean = etl::bias_batch_mean_4d(a);
    var  -= etl::bias_batch_var_4d(a, mean);

    REQUIRE_EQUALS(mean(0), Z(6.75));
    REQUIRE_EQUALS(mean(1), Z(9.375));
    REQUIRE_EQUALS(mean(2), Z(10.125));

    REQUIRE_EQUALS_APPROX(var(0), Z(10 - 22.1875));
    REQUIRE_EQUALS_APPROX(var(1), Z(10 - 27.23438));
    REQUIRE_EQUALS_APPROX(var(2), Z(10 - 66.60938));
}

TEMPLATE_TEST_CASE_2("bias_batch_var_4d/3", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 8, 3, 4, 5, 6, 7, 8, 9, 2, 11, 3, 13, 3, 15, 7, 17, 4, 19, 9, 21, 0, 11, 24});
    etl::fast_matrix<Z, 3> mean;
    etl::fast_matrix<Z, 3> var;

    var = Z(10);
    mean = etl::bias_batch_mean_4d(a);
    var  *= etl::bias_batch_var_4d(a, mean);

    REQUIRE_EQUALS(mean(0), Z(6.75));
    REQUIRE_EQUALS(mean(1), Z(9.375));
    REQUIRE_EQUALS(mean(2), Z(10.125));

    REQUIRE_EQUALS_APPROX(var(0), Z(10 * 22.1875));
    REQUIRE_EQUALS_APPROX(var(1), Z(10 * 27.23438));
    REQUIRE_EQUALS_APPROX(var(2), Z(10 * 66.60938));
}

TEMPLATE_TEST_CASE_2("bias_batch_var_4d/4", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2, 2> a({1, 8, 3, 4, 5, 6, 7, 8, 9, 2, 11, 3, 13, 3, 15, 7, 17, 4, 19, 9, 21, 0, 11, 24});
    etl::fast_matrix<Z, 3> mean;
    etl::fast_matrix<Z, 3> var;

    var = Z(10);
    mean = etl::bias_batch_mean_4d(a);
    var  /= etl::bias_batch_var_4d(a, mean);

    REQUIRE_EQUALS(mean(0), Z(6.75));
    REQUIRE_EQUALS(mean(1), Z(9.375));
    REQUIRE_EQUALS(mean(2), Z(10.125));

    REQUIRE_EQUALS_APPROX(var(0), Z(10.0 / 22.1875));
    REQUIRE_EQUALS_APPROX(var(1), Z(10.0 / 27.23438));
    REQUIRE_EQUALS_APPROX(var(2), Z(10.0 / 66.60938));
}

// Tests for argmax

TEMPLATE_TEST_CASE_2("argmax/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9});
    etl::fast_matrix<Z, 3> b;

    b = etl::argmax(a);

    REQUIRE_EQUALS(b(0), Z(2));
    REQUIRE_EQUALS(b(1), Z(2));
    REQUIRE_EQUALS(b(2), Z(2));
}

TEMPLATE_TEST_CASE_2("argmax/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({0, 1, 0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2});
    etl::fast_matrix<Z, 3> b;

    b = etl::argmax(a);

    REQUIRE_EQUALS(b(0), Z(1));
    REQUIRE_EQUALS(b(1), Z(1));
    REQUIRE_EQUALS(b(2), Z(2));
}

TEMPLATE_TEST_CASE_2("argmax/2", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 9> a({0, 1, 0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2});

    REQUIRE_EQUALS(etl::argmax(a), 1UL);
}

// Tests for argmax

TEMPLATE_TEST_CASE_2("argmin/0", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9});
    etl::fast_matrix<Z, 3> b;

    b = etl::argmin(a);

    REQUIRE_EQUALS(b(0), Z(0));
    REQUIRE_EQUALS(b(1), Z(0));
    REQUIRE_EQUALS(b(2), Z(0));
}

TEMPLATE_TEST_CASE_2("argmin/1", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({0, 1, 0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.0});
    etl::fast_matrix<Z, 3> b;

    b = etl::argmin(a);

    REQUIRE_EQUALS(b(0), Z(0));
    REQUIRE_EQUALS(b(1), Z(0));
    REQUIRE_EQUALS(b(2), Z(2));
}

TEMPLATE_TEST_CASE_2("argmin/2", "[mean]", Z, float, double) {
    etl::fast_matrix<Z, 9> a({0, 1, 0, 0.1, 0.2, 0.1, 0.1, 0.1, -0.2});

    REQUIRE_EQUALS(etl::argmin(a), 8UL);
}
