//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

// convolution_1d_full

CONV1_FULL_TEST_CASE("convolution_1d/full_1", "convolution_1d_full") {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_2", "convolution_1d_full") {
    etl::fast_vector<T, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<T, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<T, 7> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0);
    REQUIRE_EQUALS_APPROX(c[3], 8.0);
    REQUIRE_EQUALS_APPROX(c[4], 11.0);
    REQUIRE_EQUALS_APPROX(c[5], 11.0);
    REQUIRE_EQUALS_APPROX(c[6], 7.5);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_3", "convolution_1d_full") {
    etl::fast_vector<T, 9> a;
    etl::fast_vector<T, 4> b;
    etl::fast_vector<T, 12> c;

    a = etl::magic<T>(3);
    b = etl::magic<T>(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 8);
    REQUIRE_EQUALS_APPROX(c[1], 25);
    REQUIRE_EQUALS_APPROX(c[2], 41);
    REQUIRE_EQUALS_APPROX(c[3], 41);
    REQUIRE_EQUALS_APPROX(c[4], 40);
    REQUIRE_EQUALS_APPROX(c[5], 46);
    REQUIRE_EQUALS_APPROX(c[6], 51);
    REQUIRE_EQUALS_APPROX(c[7], 59);
    REQUIRE_EQUALS_APPROX(c[8], 59);
    REQUIRE_EQUALS_APPROX(c[9], 50);
    REQUIRE_EQUALS_APPROX(c[10], 26);
    REQUIRE_EQUALS_APPROX(c[11], 4);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_4", "convolution_1d_full") {
    etl::fast_vector<T, 25> a;
    etl::fast_vector<T, 9> b;
    etl::fast_vector<T, 33> c;

    a = etl::magic<T>(5);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 136);
    REQUIRE_EQUALS_APPROX(c[1], 209);
    REQUIRE_EQUALS_APPROX(c[2], 134);
    REQUIRE_EQUALS_APPROX(c[3], 260);
    REQUIRE_EQUALS_APPROX(c[4], 291);
    REQUIRE_EQUALS_APPROX(c[5], 489);
    REQUIRE_EQUALS_APPROX(c[6], 418);
    REQUIRE_EQUALS_APPROX(c[7], 540);
    REQUIRE_EQUALS_APPROX(c[8], 603);
    REQUIRE_EQUALS_APPROX(c[9], 508);
    REQUIRE_EQUALS_APPROX(c[10], 473);
    REQUIRE_EQUALS_APPROX(c[11], 503);
    REQUIRE_EQUALS_APPROX(c[12], 558);
    REQUIRE_EQUALS_APPROX(c[13], 518);
    REQUIRE_EQUALS_APPROX(c[14], 553);
    REQUIRE_EQUALS_APPROX(c[15], 523);
    REQUIRE_EQUALS_APPROX(c[16], 593);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_5", "convolution_1d_full") {
    etl::fast_vector<T, 17 * 17> a;
    etl::fast_vector<T, 3 * 3> b;
    etl::fast_vector<T, 17 * 17 + 9 - 1> c;

    a = etl::magic<T>(17);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX_E(c[0], 1240, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[1], 1547, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[2], 2648, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[3], 3398, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[4], 4515, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[5], 6037, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[6], 7227, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[7], 9268, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[8], 7947, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[9], 8496, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[10], 7515, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[11], 7452, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[12], 6777, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[13], 5490, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[14], 5121, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[15], 3222, base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c[16], 3465, base_eps * 10);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_6", "reduc_conv1_full") {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_7", "convolution_1d_full") {
    etl::dyn_vector<T> a({1.0, 2.0, 3.0});
    etl::dyn_vector<T> b({0.0, 1.0, 0.5});
    etl::dyn_vector<T> c(5);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

CONV1_FULL_TEST_CASE("convolution_1d/full_8", "convolution_1d_full") {
    etl::dyn_vector<T> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::dyn_vector<T> b({0.5, 1.0, 1.5});
    etl::dyn_vector<T> c(7);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0);
    REQUIRE_EQUALS_APPROX(c[3], 8.0);
    REQUIRE_EQUALS_APPROX(c[4], 11.0);
    REQUIRE_EQUALS_APPROX(c[5], 11.0);
    REQUIRE_EQUALS_APPROX(c[6], 7.5);
}

// convolution_1d_same

CONV1_SAME_TEST_CASE("convolution_1d/same_0", "convolution_1d_same") {
    etl::dyn_vector<T> a = {1.0, 2.0, 3.0};
    etl::dyn_vector<T> b = {0.0, 1.0, 0.5};
    etl::dyn_vector<T> c(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 2.5);
    REQUIRE_EQUALS_APPROX(c[2], 4.0);
}

CONV1_SAME_TEST_CASE("convolution_1d/same_1", "convolution_1d_same") {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 2.5);
    REQUIRE_EQUALS_APPROX(c[2], 4.0);
}

CONV1_SAME_TEST_CASE("convolution_1d/same_2", "convolution_1d_same") {
    etl::fast_vector<T, 6> a = {1.0, 2.0, 3.0, 0.0, 0.5, 2.0};
    etl::fast_vector<T, 4> b = {0.0, 0.5, 1.0, 0.0};
    etl::fast_vector<T, 6> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 2.0);
    REQUIRE_EQUALS_APPROX(c[1], 3.5);
    REQUIRE_EQUALS_APPROX(c[2], 3.0);
    REQUIRE_EQUALS_APPROX(c[3], 0.25);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
    REQUIRE_EQUALS_APPROX(c[5], 2.0);
}

CONV1_SAME_TEST_CASE("convolution_1d/same_3", "convolution_1d_same") {
    etl::fast_vector<T, 9> a;
    etl::fast_vector<T, 4> b;
    etl::fast_vector<T, 9> c;

    a = etl::magic<T>(3);
    b = etl::magic<T>(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 41);
    REQUIRE_EQUALS(c[1], 41);
    REQUIRE_EQUALS(c[2], 40);
    REQUIRE_EQUALS(c[3], 46);
    REQUIRE_EQUALS(c[4], 51);
    REQUIRE_EQUALS(c[5], 59);
    REQUIRE_EQUALS(c[6], 59);
    REQUIRE_EQUALS(c[7], 50);
    REQUIRE_EQUALS(c[8], 26);
}

CONV1_SAME_TEST_CASE("convolution_1d/same_4", "convolution_1d_same") {
    etl::fast_vector<T, 25> a;
    etl::fast_vector<T, 9> b;
    etl::fast_vector<T, 25> c;

    a = etl::magic<T>(5);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 291);
    REQUIRE_EQUALS(c[1], 489);
    REQUIRE_EQUALS(c[2], 418);
    REQUIRE_EQUALS(c[3], 540);
    REQUIRE_EQUALS(c[4], 603);
    REQUIRE_EQUALS(c[5], 508);
    REQUIRE_EQUALS(c[6], 473);
    REQUIRE_EQUALS(c[7], 503);
    REQUIRE_EQUALS(c[8], 558);
    REQUIRE_EQUALS(c[9], 518);
    REQUIRE_EQUALS(c[10], 553);
    REQUIRE_EQUALS(c[11], 523);
    REQUIRE_EQUALS(c[12], 593);
    REQUIRE_EQUALS(c[13], 573);
    REQUIRE_EQUALS(c[14], 653);
}

CONV1_SAME_TEST_CASE("convolution_1d/same_5", "convolution_1d_same") {
    etl::fast_vector<T, 17 * 17> a;
    etl::fast_vector<T, 3 * 3> b;
    etl::fast_vector<T, 17 * 17> c;

    a = etl::magic<T>(17);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[4 - 4], 4515);
    REQUIRE_EQUALS(c[5 - 4], 6037);
    REQUIRE_EQUALS(c[6 - 4], 7227);
    REQUIRE_EQUALS(c[7 - 4], 9268);
    REQUIRE_EQUALS(c[8 - 4], 7947);
    REQUIRE_EQUALS(c[9 - 4], 8496);
    REQUIRE_EQUALS(c[10 - 4], 7515);
    REQUIRE_EQUALS(c[11 - 4], 7452);
    REQUIRE_EQUALS(c[12 - 4], 6777);
    REQUIRE_EQUALS(c[13 - 4], 5490);
    REQUIRE_EQUALS(c[14 - 4], 5121);
    REQUIRE_EQUALS(c[15 - 4], 3222);
    REQUIRE_EQUALS(c[16 - 4], 3465);
}

// convolution_1d_valid

CONV1_VALID_TEST_CASE("convolution_1d/valid_1", "convolution_1d_valid") {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 1> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 2.5);
}

CONV1_VALID_TEST_CASE("convolution_1d/valid_2", "convolution_1d_valid") {
    etl::fast_vector<T, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<T, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<T, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 5.0);
    REQUIRE_EQUALS(c[1], 8.0);
    REQUIRE_EQUALS(c[2], 11);
}

CONV1_VALID_TEST_CASE("convolution_1d/valid_3", "convolution_1d_valid") {
    etl::fast_vector<T, 9> a;
    etl::fast_vector<T, 4> b;
    etl::fast_vector<T, 6> c;

    a = etl::magic<T>(3);
    b = etl::magic<T>(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 41);
    REQUIRE_EQUALS(c[1], 40);
    REQUIRE_EQUALS(c[2], 46);
    REQUIRE_EQUALS(c[3], 51);
    REQUIRE_EQUALS(c[4], 59);
    REQUIRE_EQUALS(c[5], 59);
}

CONV1_VALID_TEST_CASE("convolution_1d/valid_4", "convolution_1d_valid") {
    etl::fast_vector<T, 25> a;
    etl::fast_vector<T, 9> b;
    etl::fast_vector<T, 17> c;

    a = etl::magic<T>(5);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 603);
    REQUIRE_EQUALS(c[1], 508);
    REQUIRE_EQUALS(c[2], 473);
    REQUIRE_EQUALS(c[3], 503);
    REQUIRE_EQUALS(c[4], 558);
    REQUIRE_EQUALS(c[5], 518);
    REQUIRE_EQUALS(c[6], 553);
    REQUIRE_EQUALS(c[7], 523);
    REQUIRE_EQUALS(c[8], 593);
    REQUIRE_EQUALS(c[9], 573);
    REQUIRE_EQUALS(c[10], 653);
    REQUIRE_EQUALS(c[11], 608);
    REQUIRE_EQUALS(c[12], 698);
    REQUIRE_EQUALS(c[13], 693);
    REQUIRE_EQUALS(c[14], 713);
    REQUIRE_EQUALS(c[15], 548);
    REQUIRE_EQUALS(c[16], 633);
}

CONV1_VALID_TEST_CASE("convolution_1d/valid_5", "convolution_1d_valid") {
    etl::fast_vector<T, 17 * 17> a;
    etl::fast_vector<T, 3 * 3> b;
    etl::fast_vector<T, 17 * 17 - 9 + 1> c;

    a = etl::magic<T>(17);
    b = etl::magic<T>(3);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c[0], 7947);
    REQUIRE_EQUALS(c[1], 8496);
    REQUIRE_EQUALS(c[2], 7515);
    REQUIRE_EQUALS(c[3], 7452);
    REQUIRE_EQUALS(c[4], 6777);
    REQUIRE_EQUALS(c[5], 5490);
    REQUIRE_EQUALS(c[6], 5121);
    REQUIRE_EQUALS(c[7], 3222);
    REQUIRE_EQUALS(c[8], 3465);
    REQUIRE_EQUALS(c[9], 4328);
    REQUIRE_EQUALS(c[10], 5184);
    REQUIRE_EQUALS(c[11], 6045);
    REQUIRE_EQUALS(c[12], 6903);
    REQUIRE_EQUALS(c[13], 7763);
    REQUIRE_EQUALS(c[14], 8625);
    REQUIRE_EQUALS(c[15], 9484);
    REQUIRE_EQUALS(c[16], 8036);
}

// convolution_subs

TEMPLATE_TEST_CASE_2("convolution_1d/sub_1", "convolution_1d_full_sub", Z, float, double) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 5> c;

    etl::conv_1d_full(a(0), b(0), c(0));

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

TEMPLATE_TEST_CASE_2("convolution_1d/sub_2", "convolution_1d_same", Z, float, double) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 3> c;

    etl::conv_1d_same(a(0), b(0), c(0));

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 2.5);
    REQUIRE_EQUALS_APPROX(c[2], 4.0);
}

TEMPLATE_TEST_CASE_2("convolution_1d/sub_3", "convolution_1d_valid", Z, float, double) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 1> c;

    etl::conv_1d_valid(a(0), b(0), c(0));

    REQUIRE_EQUALS(c[0], 2.5);
}
// convolution_1d_full_expr

TEMPLATE_TEST_CASE_2("convolution_1d/expr_full_1", "convolution_1d_full", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, 2.0, 3.0});
    etl::dyn_vector<Z> b({0.0, 1.0, 0.5});
    etl::dyn_vector<Z> c(5);

    etl::conv_1d_full(a + b - b, abs(b), c);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

TEMPLATE_TEST_CASE_2("convolution_1d/expr_full_2", "convolution_1d_full", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::dyn_vector<Z> b({0.5, 1.0, 1.5});
    etl::dyn_vector<Z> c(7);

    etl::conv_1d_full(a + a - a, b + b - b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0);
    REQUIRE_EQUALS_APPROX(c[3], 8.0);
    REQUIRE_EQUALS_APPROX(c[4], 11.0);
    REQUIRE_EQUALS_APPROX(c[5], 11.0);
    REQUIRE_EQUALS_APPROX(c[6], 7.5);
}

// Mixed tests

ETL_TEST_CASE("conv1/full/mixed/0", "[conv1][conv]") {
    etl::fast_vector<float, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<double, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<float, 5> c;

    c = conv_1d_full(a, b);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

ETL_TEST_CASE("conv1/valid/mixed/0", "[conv1][conv]") {
    etl::fast_vector<float, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<double, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<float, 3> c;

    c = conv_1d_valid(a, b, c);

    REQUIRE_EQUALS(c[0], 5.0);
    REQUIRE_EQUALS(c[1], 8.0);
    REQUIRE_EQUALS(c[2], 11);
}

ETL_TEST_CASE("conv1/same/mixed/0", "[conv1][conv]") {
    etl::fast_vector<float, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<double, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<float, 3> c;

    c = conv_1d_same(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 2.5);
    REQUIRE_EQUALS_APPROX(c[2], 4.0);
}

ETL_TEST_CASE("conv1/full/mixed/1", "[conv1][conv]") {
    etl::fast_vector<float, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector_cm<float, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<float, 5> c;

    c = conv_1d_full(a, b);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

ETL_TEST_CASE("conv1/valid/mixed/1", "[conv1][conv]") {
    etl::fast_vector<float, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector_cm<float, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<float, 3> c;

    c = conv_1d_valid(a, b, c);

    REQUIRE_EQUALS(c[0], 5.0);
    REQUIRE_EQUALS(c[1], 8.0);
    REQUIRE_EQUALS(c[2], 11);
}

ETL_TEST_CASE("conv1/same/mixed/1", "[conv1][conv]") {
    etl::fast_vector<float, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector_cm<float, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<float, 3> c;

    c = conv_1d_same(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 1.0);
    REQUIRE_EQUALS_APPROX(c[1], 2.5);
    REQUIRE_EQUALS_APPROX(c[2], 4.0);
}
