//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

//{{{ convolution_1d_full

CONV1_FULL_TEST_CASE( "convolution_1d/full_1", "convolution_1d_full" ) {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_2", "convolution_1d_full" ) {
    etl::fast_vector<T, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<T, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<T, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.5));
    REQUIRE(c[1] == Approx(2.0));
    REQUIRE(c[2] == Approx(5.0));
    REQUIRE(c[3] == Approx(8.0));
    REQUIRE(c[4] == Approx(11.0));
    REQUIRE(c[5] == Approx(11.0));
    REQUIRE(c[6] == Approx(7.5));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_3", "convolution_1d_full" ) {
    etl::fast_vector<T, 9> a(etl::magic<T>(3));
    etl::fast_vector<T, 4> b(etl::magic<T>(2));
    etl::fast_vector<T, 12> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(8));
    REQUIRE(c[1] == Approx(25));
    REQUIRE(c[2] == Approx(41));
    REQUIRE(c[3] == Approx(41));
    REQUIRE(c[4] == Approx(40));
    REQUIRE(c[5] == Approx(46));
    REQUIRE(c[6] == Approx(51));
    REQUIRE(c[7] == Approx(59));
    REQUIRE(c[8] == Approx(59));
    REQUIRE(c[9] == Approx(50));
    REQUIRE(c[10] == Approx(26));
    REQUIRE(c[11] == Approx(4));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_4", "convolution_1d_full" ) {
    etl::fast_vector<T, 25> a(etl::magic(5));
    etl::fast_vector<T, 9> b(etl::magic(3));
    etl::fast_vector<T, 33> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(136));
    REQUIRE(c[1] == Approx(209));
    REQUIRE(c[2] == Approx(134));
    REQUIRE(c[3] == Approx(260));
    REQUIRE(c[4] == Approx(291));
    REQUIRE(c[5] == Approx(489));
    REQUIRE(c[6] == Approx(418));
    REQUIRE(c[7] == Approx(540));
    REQUIRE(c[8] == Approx(603));
    REQUIRE(c[9] == Approx(508));
    REQUIRE(c[10] == Approx(473));
    REQUIRE(c[11] == Approx(503));
    REQUIRE(c[12] == Approx(558));
    REQUIRE(c[13] == Approx(518));
    REQUIRE(c[14] == Approx(553));
    REQUIRE(c[15] == Approx(523));
    REQUIRE(c[16] == Approx(593));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_5", "convolution_1d_full" ) {
    etl::fast_vector<T, 17 * 17> a(etl::magic(17));
    etl::fast_vector<T, 3 * 3> b(etl::magic(3));
    etl::fast_vector<T, 17 * 17 + 9 - 1> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(1240));
    REQUIRE(c[1] == Approx(1547));
    REQUIRE(c[2] == Approx(2648));
    REQUIRE(c[3] == Approx(3398));
    REQUIRE(c[4] == Approx(4515));
    REQUIRE(c[5] == Approx(6037));
    REQUIRE(c[6] == Approx(7227));
    REQUIRE(c[7] == Approx(9268));
    REQUIRE(c[8] == Approx(7947));
    REQUIRE(c[9] == Approx(8496));
    REQUIRE(c[10] == Approx(7515));
    REQUIRE(c[11] == Approx(7452));
    REQUIRE(c[12] == Approx(6777));
    REQUIRE(c[13] == Approx(5490));
    REQUIRE(c[14] == Approx(5121));
    REQUIRE(c[15] == Approx(3222));
    REQUIRE(c[16] == Approx(3465));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_6", "reduc_conv1_full" ) {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_7", "convolution_1d_full" ) {
    etl::dyn_vector<T> a({1.0, 2.0, 3.0});
    etl::dyn_vector<T> b({0.0, 1.0, 0.5});
    etl::dyn_vector<T> c(5);

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_8", "convolution_1d_full" ) {
    etl::dyn_vector<T> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::dyn_vector<T> b({0.5, 1.0, 1.5});
    etl::dyn_vector<T> c(7);

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.5));
    REQUIRE(c[1] == Approx(2.0));
    REQUIRE(c[2] == Approx(5.0));
    REQUIRE(c[3] == Approx(8.0));
    REQUIRE(c[4] == Approx(11.0));
    REQUIRE(c[5] == Approx(11.0));
    REQUIRE(c[6] == Approx(7.5));
}

//}}}

//{{{ convolution_1d_same

CONV1_SAME_TEST_CASE( "convolution_1d/same_1", "convolution_1d_same" ) {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 3> c;

    *etl::conv_1d_same(a, b, c);

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(2.5));
    REQUIRE(c[2] == Approx(4.0));
}

CONV1_SAME_TEST_CASE( "convolution_1d/same_2", "convolution_1d_same" ) {
    etl::fast_vector<T, 6> a = {1.0, 2.0, 3.0, 0.0, 0.5, 2.0};
    etl::fast_vector<T, 4> b = {0.0, 0.5, 1.0, 0.0};
    etl::fast_vector<T, 6> c;

    *etl::conv_1d_same(a, b, c);

    REQUIRE(c[0] == Approx(2.0));
    REQUIRE(c[1] == Approx(3.5));
    REQUIRE(c[2] == Approx(3.0));
    REQUIRE(c[3] == Approx(0.25));
    REQUIRE(c[4] == Approx(1.5));
    REQUIRE(c[5] == Approx(2.0));
}

CONV1_SAME_TEST_CASE( "convolution_1d/same_3", "convolution_1d_same" ) {
    etl::fast_vector<T, 9> a(etl::magic(3));
    etl::fast_vector<T, 4> b(etl::magic(2));
    etl::fast_vector<T, 9> c;

    *etl::conv_1d_same(a, b, c);

    REQUIRE(c[0] == 41);
    REQUIRE(c[1] == 41);
    REQUIRE(c[2] == 40);
    REQUIRE(c[3] == 46);
    REQUIRE(c[4] == 51);
    REQUIRE(c[5] == 59);
    REQUIRE(c[6] == 59);
    REQUIRE(c[7] == 50);
    REQUIRE(c[8] == 26);
}

CONV1_SAME_TEST_CASE( "convolution_1d/same_4", "convolution_1d_same" ) {
    etl::fast_vector<T, 25> a(etl::magic(5));
    etl::fast_vector<T, 9> b(etl::magic(3));
    etl::fast_vector<T, 25> c;

    *etl::conv_1d_same(a, b, c);

    REQUIRE(c[0] == 291);
    REQUIRE(c[1] == 489);
    REQUIRE(c[2] == 418);
    REQUIRE(c[3] == 540);
    REQUIRE(c[4] == 603);
    REQUIRE(c[5] == 508);
    REQUIRE(c[6] == 473);
    REQUIRE(c[7] == 503);
    REQUIRE(c[8] == 558);
    REQUIRE(c[9] == 518);
    REQUIRE(c[10] == 553);
    REQUIRE(c[11] == 523);
    REQUIRE(c[12] == 593);
    REQUIRE(c[13] == 573);
    REQUIRE(c[14] == 653);
}

CONV1_SAME_TEST_CASE( "convolution_1d/same_5", "convolution_1d_same" ) {
    etl::fast_vector<T, 17 * 17> a(etl::magic(17));
    etl::fast_vector<T, 3 * 3> b(etl::magic(3));
    etl::fast_vector<T, 17 * 17> c;

    *etl::conv_1d_same(a, b, c);

    REQUIRE(c[4-4] == 4515);
    REQUIRE(c[5-4] == 6037);
    REQUIRE(c[6-4] == 7227);
    REQUIRE(c[7-4] == 9268);
    REQUIRE(c[8-4] == 7947);
    REQUIRE(c[9-4] == 8496);
    REQUIRE(c[10-4] == 7515);
    REQUIRE(c[11-4] == 7452);
    REQUIRE(c[12-4] == 6777);
    REQUIRE(c[13-4] == 5490);
    REQUIRE(c[14-4] == 5121);
    REQUIRE(c[15-4] == 3222);
    REQUIRE(c[16-4] == 3465);
}

//}}}

//{{{ convolution_1d_valid

CONV1_VALID_TEST_CASE( "convolution_1d/valid_1", "convolution_1d_valid" ) {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector<T, 1> c;

    *etl::conv_1d_valid(a, b, c);

    REQUIRE(c[0] == 2.5);
}

CONV1_VALID_TEST_CASE( "convolution_1d/valid_2", "convolution_1d_valid" ) {
    etl::fast_vector<T, 5> a = {1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_vector<T, 3> b = {0.5, 1.0, 1.5};
    etl::fast_vector<T, 3> c;

    *etl::conv_1d_valid(a, b, c);

    REQUIRE(c[0] == 5.0);
    REQUIRE(c[1] == 8.0);
    REQUIRE(c[2] == 11);
}

CONV1_VALID_TEST_CASE( "convolution_1d/valid_3", "convolution_1d_valid" ) {
    etl::fast_vector<T, 9> a(etl::magic(3));
    etl::fast_vector<T, 4> b(etl::magic(2));
    etl::fast_vector<T, 6> c;

    *etl::conv_1d_valid(a, b, c);

    REQUIRE(c[0] == 41);
    REQUIRE(c[1] == 40);
    REQUIRE(c[2] == 46);
    REQUIRE(c[3] == 51);
    REQUIRE(c[4] == 59);
    REQUIRE(c[5] == 59);
}

CONV1_VALID_TEST_CASE( "convolution_1d/valid_4", "convolution_1d_valid" ) {
    etl::fast_vector<T, 25> a(etl::magic(5));
    etl::fast_vector<T, 9> b(etl::magic(3));
    etl::fast_vector<T, 17> c;

    *etl::conv_1d_valid(a, b, c);

    REQUIRE(c[0] == 603);
    REQUIRE(c[1] == 508);
    REQUIRE(c[2] == 473);
    REQUIRE(c[3] == 503);
    REQUIRE(c[4] == 558);
    REQUIRE(c[5] == 518);
    REQUIRE(c[6] == 553);
    REQUIRE(c[7] == 523);
    REQUIRE(c[8] == 593);
    REQUIRE(c[9] == 573);
    REQUIRE(c[10] == 653);
    REQUIRE(c[11] == 608);
    REQUIRE(c[12] == 698);
    REQUIRE(c[13] == 693);
    REQUIRE(c[14] == 713);
    REQUIRE(c[15] == 548);
    REQUIRE(c[16] == 633);
}

CONV1_VALID_TEST_CASE( "convolution_1d/valid_5", "convolution_1d_valid" ) {
    etl::fast_vector<T, 17 * 17> a(etl::magic(17));
    etl::fast_vector<T, 3 * 3> b(etl::magic(3));
    etl::fast_vector<T, 17 * 17 - 9 + 1> c;

    *etl::conv_1d_valid(a, b, c);

    REQUIRE(c[0] == 7947);
    REQUIRE(c[1] == 8496);
    REQUIRE(c[2] == 7515);
    REQUIRE(c[3] == 7452);
    REQUIRE(c[4] == 6777);
    REQUIRE(c[5] == 5490);
    REQUIRE(c[6] == 5121);
    REQUIRE(c[7] == 3222);
    REQUIRE(c[8] == 3465);
    REQUIRE(c[9] == 4328);
    REQUIRE(c[10] == 5184);
    REQUIRE(c[11] == 6045);
    REQUIRE(c[12] == 6903);
    REQUIRE(c[13] == 7763);
    REQUIRE(c[14] == 8625);
    REQUIRE(c[15] == 9484);
    REQUIRE(c[16] == 8036);
}

//}}}

//{{{ convolution_subs

TEMPLATE_TEST_CASE_2( "convolution_1d/sub_1", "convolution_1d_full_sub", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 5> c;

    *etl::conv_1d_full(a(0), b(0), c(0));

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

TEMPLATE_TEST_CASE_2( "convolution_1d/sub_2", "convolution_1d_same", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 3> c;

    *etl::conv_1d_same(a(0), b(0), c(0));

    REQUIRE(c[0] == Approx(1.0));
    REQUIRE(c[1] == Approx(2.5));
    REQUIRE(c[2] == Approx(4.0));
}

TEMPLATE_TEST_CASE_2( "convolution_1d/sub_3", "convolution_1d_valid", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 1, 3> b = {0.0, 1.0, 0.5};
    etl::fast_matrix<Z, 1, 1> c;

    *etl::conv_1d_valid(a(0), b(0), c(0));

    REQUIRE(c[0] == 2.5);
}
//{{{ convolution_1d_full_expr

TEMPLATE_TEST_CASE_2( "convolution_1d/expr_full_1", "convolution_1d_full", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, 2.0, 3.0});
    etl::dyn_vector<Z> b({0.0, 1.0, 0.5});
    etl::dyn_vector<Z> c(5);

    *etl::conv_1d_full(a + b - b, abs(b), c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

TEMPLATE_TEST_CASE_2( "convolution_1d/expr_full_2", "convolution_1d_full", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::dyn_vector<Z> b({0.5, 1.0, 1.5});
    etl::dyn_vector<Z> c(7);

    *etl::conv_1d_full(a + a - a, b + b - b, c);

    REQUIRE(c[0] == Approx(0.5));
    REQUIRE(c[1] == Approx(2.0));
    REQUIRE(c[2] == Approx(5.0));
    REQUIRE(c[3] == Approx(8.0));
    REQUIRE(c[4] == Approx(11.0));
    REQUIRE(c[5] == Approx(11.0));
    REQUIRE(c[6] == Approx(7.5));
}

//}}}
