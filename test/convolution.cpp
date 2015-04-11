//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

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

    REQUIRE(c[0] == 8);
    REQUIRE(c[1] == 25);
    REQUIRE(c[2] == 41);
    REQUIRE(c[3] == 41);
    REQUIRE(c[4] == 40);
    REQUIRE(c[5] == 46);
    REQUIRE(c[6] == 51);
    REQUIRE(c[7] == 59);
    REQUIRE(c[8] == 59);
    REQUIRE(c[9] == 50);
    REQUIRE(c[10] == 26);
    REQUIRE(c[11] == 4);
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_4", "convolution_1d_full" ) {
    etl::fast_vector<T, 25> a(etl::magic(5));
    etl::fast_vector<T, 9> b(etl::magic(3));
    etl::fast_vector<T, 33> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == 136);
    REQUIRE(c[1] == 209);
    REQUIRE(c[2] == 134);
    REQUIRE(c[3] == 260);
    REQUIRE(c[4] == 291);
    REQUIRE(c[5] == 489);
    REQUIRE(c[6] == 418);
    REQUIRE(c[7] == 540);
    REQUIRE(c[8] == 603);
    REQUIRE(c[9] == 508);
    REQUIRE(c[10] == 473);
    REQUIRE(c[11] == 503);
    REQUIRE(c[12] == 558);
    REQUIRE(c[13] == 518);
    REQUIRE(c[14] == 553);
    REQUIRE(c[15] == 523);
    REQUIRE(c[16] == 593);
}

CONV1_FULL_TEST_CASE( "convolution_1d/full_5", "convolution_1d_full" ) {
    etl::fast_vector<T, 17 * 17> a(etl::magic(17));
    etl::fast_vector<T, 3 * 3> b(etl::magic(3));
    etl::fast_vector<T, 17 * 17 + 9 - 1> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == 1240);
    REQUIRE(c[1] == 1547);
    REQUIRE(c[2] == 2648);
    REQUIRE(c[3] == 3398);
    REQUIRE(c[4] == 4515);
    REQUIRE(c[5] == 6037);
    REQUIRE(c[6] == 7227);
    REQUIRE(c[7] == 9268);
    REQUIRE(c[8] == 7947);
    REQUIRE(c[9] == 8496);
    REQUIRE(c[10] == 7515);
    REQUIRE(c[11] == 7452);
    REQUIRE(c[12] == 6777);
    REQUIRE(c[13] == 5490);
    REQUIRE(c[14] == 5121);
    REQUIRE(c[15] == 3222);
    REQUIRE(c[16] == 3465);
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

//{{{ convolution_2d_full

CONV2_FULL_TEST_CASE( "convolution_2d/full_1", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 4.0);
    REQUIRE(c(0,2) == 6.0);
    REQUIRE(c(0,3) == 0.0);

    REQUIRE(c(1,0) == 0.5);
    REQUIRE(c(1,1) == 3.5);
    REQUIRE(c(1,2) == 4.5);
    REQUIRE(c(1,3) == 1.5);

    REQUIRE(c(2,0) == 6.0);
    REQUIRE(c(2,1) == 4.5);
    REQUIRE(c(2,2) == 3.0);
    REQUIRE(c(2,3) == 0.5);

    REQUIRE(c(3,0) == 1.5);
    REQUIRE(c(3,1) == 2.5);
    REQUIRE(c(3,2) == 1.5);
    REQUIRE(c(3,3) == 0.5);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_2", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 3> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 4.0);
    REQUIRE(c(0,2) == 0.0);

    REQUIRE(c(1,0) == 0.5);
    REQUIRE(c(1,1) == 3.5);
    REQUIRE(c(1,2) == 1.0);

    REQUIRE(c(2,0) == 6.0);
    REQUIRE(c(2,1) == 4.5);
    REQUIRE(c(2,2) == 0.5);

    REQUIRE(c(3,0) == 1.5);
    REQUIRE(c(3,1) == 2.5);
    REQUIRE(c(3,2) == 1.0);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_3", "convolution_2d_full" ) {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 2.0);
    REQUIRE(c(0,1) == 5.0);
    REQUIRE(c(0,2) == 2.0);

    REQUIRE(c(1,0) == 6.5);
    REQUIRE(c(1,1) == 8.5);
    REQUIRE(c(1,2) == 3.0);

    REQUIRE(c(2,0) == 1.5);
    REQUIRE(c(2,1) == 2.5);
    REQUIRE(c(2,2) == 1.0);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_4", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == 8);
    REQUIRE(c(0, 1) == 25);
    REQUIRE(c(0, 2) == 9);
    REQUIRE(c(0, 3) == 18);

    REQUIRE(c(1, 0) == 35);
    REQUIRE(c(1, 1) == 34);
    REQUIRE(c(1, 2) == 48);
    REQUIRE(c(1, 3) == 33);

    REQUIRE(c(2, 0) == 16);
    REQUIRE(c(2, 1) == 47);
    REQUIRE(c(2, 2) == 67);
    REQUIRE(c(2, 3) == 20);

    REQUIRE(c(3, 0) == 16);
    REQUIRE(c(3, 1) == 44);
    REQUIRE(c(3, 2) == 26);
    REQUIRE(c(3, 3) == 4);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_5", "convolution_2d_full" ) {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 7, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == 136);
    REQUIRE(c(0, 1) == 209);
    REQUIRE(c(0, 2) == 134);
    REQUIRE(c(0, 3) == 209);

    REQUIRE(c(1, 0) == 235);
    REQUIRE(c(1, 1) == 220);
    REQUIRE(c(1, 2) == 441);
    REQUIRE(c(1, 3) == 346);

    REQUIRE(c(2, 0) == 169);
    REQUIRE(c(2, 1) == 431);
    REQUIRE(c(2, 2) == 595);
    REQUIRE(c(2, 3) == 410);

    REQUIRE(c(3, 0) == 184);
    REQUIRE(c(3, 1) == 371);
    REQUIRE(c(3, 2) == 440);
    REQUIRE(c(3, 3) == 555);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_6", "convolution_2d_full" ) {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 19, 19> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == 1240);
    REQUIRE(c(0, 1) == 1547);
    REQUIRE(c(0, 2) == 2648);
    REQUIRE(c(0, 3) == 2933);

    REQUIRE(c(1, 0) == 1849);
    REQUIRE(c(1, 1) == 3006);
    REQUIRE(c(1, 2) == 5452);
    REQUIRE(c(1, 3) == 6022);

    REQUIRE(c(2, 0) == 2667);
    REQUIRE(c(2, 1) == 5403);
    REQUIRE(c(2, 2) == 8640);
    REQUIRE(c(2, 3) == 9495);

    REQUIRE(c(3, 0) == 2937);
    REQUIRE(c(3, 1) == 5943);
    REQUIRE(c(3, 2) == 9450);
    REQUIRE(c(3, 3) == 10305);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_7", "convolution_2d_full" ) {
    etl::fast_matrix<T, 2, 6> a = {1,2,3,4,5,6,7,8,9,10,11,12};
    etl::fast_matrix<T, 2, 2> b = {1,2,3,4};
    etl::fast_matrix<T, 3, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 1) == 4);
    REQUIRE(c(0, 2) == 7);
    REQUIRE(c(0, 3) == 10);
    REQUIRE(c(0, 4) == 13);
    REQUIRE(c(0, 5) == 16);
    REQUIRE(c(0, 6) == 12);

    REQUIRE(c(1, 0) == 10);
    REQUIRE(c(1, 1) == 32);
    REQUIRE(c(1, 2) == 42);
    REQUIRE(c(1, 3) == 52);
    REQUIRE(c(1, 4) == 62);
    REQUIRE(c(1, 5) == 72);
    REQUIRE(c(1, 6) == 48);

    REQUIRE(c(2, 0) == 21);
    REQUIRE(c(2, 1) == 52);
    REQUIRE(c(2, 2) == 59);
    REQUIRE(c(2, 3) == 66);
    REQUIRE(c(2, 4) == 73);
    REQUIRE(c(2, 5) == 80);
    REQUIRE(c(2, 6) == 48);
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_8", "convolution_2d_full" ) {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 41, 41> c;

    Impl::apply(a, b, c);

    CHECK(c(0, 0) == 26461);
    CHECK(c(0, 1) == 60760);
    CHECK(c(0, 2) == 103282);
    CHECK(c(0, 3) == 154412);

    CHECK(c(1, 0) == 60150);
    CHECK(c(1, 1) == 136700);
    CHECK(c(1, 2) == 230420);
    CHECK(c(1, 3) == 296477);

    CHECK(c(2, 0) == 101407);
    CHECK(c(2, 1) == 228500);
    CHECK(c(2, 2) == 336831);
    CHECK(c(2, 3) == 416899);

    CHECK(c(3, 0) == 150572);
    CHECK(c(3, 1) == 291237);
    CHECK(c(3, 2) == 417946);
    CHECK(c(3, 3) == 516210);
}

//}}}

//{{{ convolution_2d_same

CONV2_SAME_TEST_CASE( "convolution_2d/same_1", "convolution_2d_same" ) {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 3> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 4.5);
    REQUIRE(c(0,2) == 1.5);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 3.0);
    REQUIRE(c(1,2) == 0.5);

    REQUIRE(c(2,0) == 2.5);
    REQUIRE(c(2,1) == 1.5);
    REQUIRE(c(2,2) == 0.5);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_2", "convolution_2d_same" ) {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 2> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 1.0);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 0.5);

    REQUIRE(c(2,0) == 2.5);
    REQUIRE(c(2,1) == 1.0);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_3", "convolution_2d_same" ) {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0,0) == 8.5);
    REQUIRE(c(0,1) == 3.0);

    REQUIRE(c(1,0) == 2.5);
    REQUIRE(c(1,1) == 1.0);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_4", "convolution_2d_same" ) {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 3, 3> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0, 0) == 34);
    REQUIRE(c(0, 1) == 48);
    REQUIRE(c(0, 2) == 33);

    REQUIRE(c(1, 0) == 47);
    REQUIRE(c(1, 1) == 67);
    REQUIRE(c(1, 2) == 20);

    REQUIRE(c(2, 0) == 44);
    REQUIRE(c(2, 1) == 26);
    REQUIRE(c(2, 2) == 4);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_5", "convolution_2d_same" ) {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 5, 5> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0, 0) == 220);
    REQUIRE(c(0, 1) == 441);
    REQUIRE(c(0, 2) == 346);
    REQUIRE(c(0, 3) == 276);

    REQUIRE(c(1, 0) == 431);
    REQUIRE(c(1, 1) == 595);
    REQUIRE(c(1, 2) == 410);
    REQUIRE(c(1, 3) == 575);

    REQUIRE(c(2, 0) == 371);
    REQUIRE(c(2, 1) == 440);
    REQUIRE(c(2, 2) == 555);
    REQUIRE(c(2, 3) == 620);

    REQUIRE(c(3, 0) == 301);
    REQUIRE(c(3, 1) == 585);
    REQUIRE(c(3, 2) == 600);
    REQUIRE(c(3, 3) == 765);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_6", "convolution_2d_same" ) {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 17, 17> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0, 0) == 3006);
    REQUIRE(c(0, 1) == 5452);
    REQUIRE(c(0, 2) == 6022);
    REQUIRE(c(0, 3) == 6592);

    REQUIRE(c(1, 0) == 5403);
    REQUIRE(c(1, 1) == 8640);
    REQUIRE(c(1, 2) == 9495);
    REQUIRE(c(1, 3) == 10350);

    REQUIRE(c(2, 0) == 5943);
    REQUIRE(c(2, 1) == 9450);
    REQUIRE(c(2, 2) == 10305);
    REQUIRE(c(2, 3) == 11160);

    REQUIRE(c(3, 0) == 6483);
    REQUIRE(c(3, 1) == 10260);
    REQUIRE(c(3, 2) == 11115);
    REQUIRE(c(3, 3) == 9658);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_7", "convolution_2d_same" ) {
    etl::fast_matrix<T, 2, 6> a = {1,2,3,4,5,6,7,8,9,10,11,12};
    etl::fast_matrix<T, 2, 2> b = {1,2,3,4};
    etl::fast_matrix<T, 2, 6> c;

    *etl::conv_2d_same(a, b, c);

    REQUIRE(c(0, 0) == 32);
    REQUIRE(c(0, 1) == 42);
    REQUIRE(c(0, 2) == 52);
    REQUIRE(c(0, 3) == 62);
    REQUIRE(c(0, 4) == 72);
    REQUIRE(c(0, 5) == 48);

    REQUIRE(c(1, 0) == 52);
    REQUIRE(c(1, 1) == 59);
    REQUIRE(c(1, 2) == 66);
    REQUIRE(c(1, 3) == 73);
    REQUIRE(c(1, 4) == 80);
    REQUIRE(c(1, 5) == 48);
}

CONV2_SAME_TEST_CASE( "convolution_2d/same_8", "convolution_2d_same" ) {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 33, 33> c;

    *etl::conv_2d_same(a, b, c);

    CHECK(c(0, 0) == 676494);
    CHECK(c(0, 1) == 806569);
    CHECK(c(0, 2) == 976949);
    CHECK(c(0, 3) == 1179119);

    CHECK(c(1, 0) == 808354);
    CHECK(c(1, 1) == 984480);
    CHECK(c(1, 2) == 1206077);
    CHECK(c(1, 3) == 1469155);

    CHECK(c(2, 0) == 971394);
    CHECK(c(2, 1) == 1202744);
    CHECK(c(2, 2) == 1485149);
    CHECK(c(2, 3) == 1773847);

    CHECK(c(3, 0) == 1173020);
    CHECK(c(3, 1) == 1464355);
    CHECK(c(3, 2) == 1771896);
    CHECK(c(3, 3) == 2091280);
}


//}}}

//{{{ convolution_2d_valid

CONV2_VALID_TEST_CASE( "convolution_2d/valid_1", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 4.5);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 3.0);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_2", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 1> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(1,0) == 4.5);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_3", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 1, 1> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 8.5);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_4", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 2, 2> c;

    etl::force(etl::conv_2d_valid(a, b, c));

    REQUIRE(c(0, 0) == 34);
    REQUIRE(c(0, 1) == 48);

    REQUIRE(c(1, 0) == 47);
    REQUIRE(c(1, 1) == 67);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_5", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 3, 3> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0, 0) == 595);
    REQUIRE(c(0, 1) == 410);
    REQUIRE(c(0, 2) == 575);

    REQUIRE(c(1, 0) == 440);
    REQUIRE(c(1, 1) == 555);
    REQUIRE(c(1, 2) == 620);

    REQUIRE(c(2, 0) == 585);
    REQUIRE(c(2, 1) == 600);
    REQUIRE(c(2, 2) == 765);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_6", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 15, 15> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0, 0) == 8640);
    REQUIRE(c(0, 1) == 9495);
    REQUIRE(c(0, 2) == 10350);
    REQUIRE(c(0, 3) == 11205);

    REQUIRE(c(1, 0) == 9450);
    REQUIRE(c(1, 1) == 10305);
    REQUIRE(c(1, 2) == 11160);
    REQUIRE(c(1, 3) == 9703);

    REQUIRE(c(2, 0) == 10260);
    REQUIRE(c(2, 1) == 11115);
    REQUIRE(c(2, 2) == 9658);
    REQUIRE(c(2, 3) == 9357);

    REQUIRE(c(3, 0) == 11070);
    REQUIRE(c(3, 1) == 9613);
    REQUIRE(c(3, 2) == 9312);
    REQUIRE(c(3, 3) == 5832);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_7", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 2, 6> a = {1,2,3,4,5,6,7,8,9,10,11,12};
    etl::fast_matrix<T, 2, 2> b = {1,2,3,4};
    etl::fast_matrix<T, 1, 5> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0, 0) == 32);
    REQUIRE(c(0, 1) == 42);
    REQUIRE(c(0, 2) == 52);
    REQUIRE(c(0, 3) == 62);
    REQUIRE(c(0, 4) == 72);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_8", "convolution_2d_valid" ) {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 25, 25> c;

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0, 0) == 2735136);
    REQUIRE(c(0, 1) == 2726136);
    REQUIRE(c(0, 2) == 2620215);
    REQUIRE(c(0, 3) == 2394504);

    REQUIRE(c(1, 0) == 2722815);
    REQUIRE(c(1, 1) == 2616894);
    REQUIRE(c(1, 2) == 2391183);
    REQUIRE(c(1, 3) == 2473659);

    REQUIRE(c(2, 0) == 2613573);
    REQUIRE(c(2, 1) == 2387862);
    REQUIRE(c(2, 2) == 2470338);
    REQUIRE(c(2, 3) == 2493546);

    REQUIRE(c(3, 0) == 2384541);
    REQUIRE(c(3, 1) == 2467017);
    REQUIRE(c(3, 2) == 2491776);
    REQUIRE(c(3, 3) == 2432517);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_9", "convolution_2d_valid" ) {
    etl::dyn_matrix<T> a(3,3, std::initializer_list<T>({1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0}));
    etl::dyn_matrix<T> b(2,2, std::initializer_list<T>({2.0, 0.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(2,2);

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(0,1) == 4.5);

    REQUIRE(c(1,0) == 4.5);
    REQUIRE(c(1,1) == 3.0);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_10", "convolution_2d_valid" ) {
    etl::dyn_matrix<T> a(3,2,std::initializer_list<T>({1.0, 2.0, 0.0, 1.0, 3.0, 2.0}));
    etl::dyn_matrix<T> b(2,2,std::initializer_list<T>({2.0, 0.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(2,1);

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 3.5);
    REQUIRE(c(1,0) == 4.5);
}

CONV2_VALID_TEST_CASE( "convolution_2d/valid_11", "convolution_2d_valid" ) {
    etl::dyn_matrix<T> a(2,2,std::initializer_list<T>({1.0, 2.0, 3.0, 2.0}));
    etl::dyn_matrix<T> b(2,2,std::initializer_list<T>({2.0, 1.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(1,1);

    *etl::conv_2d_valid(a, b, c);

    REQUIRE(c(0,0) == 8.5);
}

//}}}

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

TEMPLATE_TEST_CASE_2( "convolution_2d/sub_1", "convolution_2d_full", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<Z, 1, 4, 4> c;

    *etl::conv_2d_full(a(0), b(0), c(0));

    REQUIRE(c(0,0,0) == 2.0);
    REQUIRE(c(0,0,1) == 4.0);
    REQUIRE(c(0,0,2) == 6.0);
    REQUIRE(c(0,0,3) == 0.0);

    REQUIRE(c(0,1,0) == 0.5);
    REQUIRE(c(0,1,1) == 3.5);
    REQUIRE(c(0,1,2) == 4.5);
    REQUIRE(c(0,1,3) == 1.5);

    REQUIRE(c(0,2,0) == 6.0);
    REQUIRE(c(0,2,1) == 4.5);
    REQUIRE(c(0,2,2) == 3.0);
    REQUIRE(c(0,2,3) == 0.5);

    REQUIRE(c(0,3,0) == 1.5);
    REQUIRE(c(0,3,1) == 2.5);
    REQUIRE(c(0,3,2) == 1.5);
    REQUIRE(c(0,3,3) == 0.5);
}

TEMPLATE_TEST_CASE_2( "convolution_2d/sub_2", "convolution_2d_same", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<Z, 1, 3, 3> c;

    *etl::conv_2d_same(a(0), b(0), c(0));

    REQUIRE(c(0,0,0) == 3.5);
    REQUIRE(c(0,0,1) == 4.5);
    REQUIRE(c(0,0,2) == 1.5);

    REQUIRE(c(0,1,0) == 4.5);
    REQUIRE(c(0,1,1) == 3.0);
    REQUIRE(c(0,1,2) == 0.5);

    REQUIRE(c(0,2,0) == 2.5);
    REQUIRE(c(0,2,1) == 1.5);
    REQUIRE(c(0,2,2) == 0.5);
}

TEMPLATE_TEST_CASE_2( "convolution_2d/sub_3", "convolution_2d_valid", Z, float, double ) {
    etl::fast_matrix<Z, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<Z, 1, 2, 2> c;

    *etl::conv_2d_valid(a(0), b(0), c(0));

    REQUIRE(c(0,0,0) == 3.5);
    REQUIRE(c(0,0,1) == 4.5);

    REQUIRE(c(0,1,0) == 4.5);
    REQUIRE(c(0,1,1) == 3.0);
}

//}}}

//{{{ convolution_deep_full

TEMPLATE_TEST_CASE_2( "convolution_3d/full_1", "convolution_deep_full", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0};
    etl::fast_matrix<Z, 2, 3, 3> c;

    *etl::conv_deep_full(a, b, c);

    REQUIRE(c(0,0,0) == 2.0);
    REQUIRE(c(0,0,1) == 5.0);
    REQUIRE(c(0,0,2) == 2.0);

    REQUIRE(c(0,1,0) == 6.5);
    REQUIRE(c(0,1,1) == 8.5);
    REQUIRE(c(0,1,2) == 3.0);

    REQUIRE(c(0,2,0) == 1.5);
    REQUIRE(c(0,2,1) == 2.5);
    REQUIRE(c(0,2,2) == 1.0);

    REQUIRE(c(1,0,0) == 5.0);
    REQUIRE(c(1,0,1) == 16.0);
    REQUIRE(c(1,0,2) == 12.0);

    REQUIRE(c(1,1,0) == 12.0);
    REQUIRE(c(1,1,1) == 38.0);
    REQUIRE(c(1,1,2) == 28.0);

    REQUIRE(c(1,2,0) == 7.0);
    REQUIRE(c(1,2,1) == 22.0);
    REQUIRE(c(1,2,2) == 16.0);
}

TEMPLATE_TEST_CASE_2( "convolution_4d/full_1", "convolution_deep_full", Z, float, double ) {
    etl::fast_matrix<Z, 1, 2, 2, 2> a = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    etl::fast_matrix<Z, 1, 2, 2, 2> b = {2.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0};
    etl::fast_matrix<Z, 1, 2, 3, 3> c;

    c = etl::conv_deep_full(a, b);

    REQUIRE(c(0,0,0,0) == 2.0);
    REQUIRE(c(0,0,0,1) == 5.0);
    REQUIRE(c(0,0,0,2) == 2.0);

    REQUIRE(c(0,0,1,0) == 6.5);
    REQUIRE(c(0,0,1,1) == 8.5);
    REQUIRE(c(0,0,1,2) == 3.0);

    REQUIRE(c(0,0,2,0) == 1.5);
    REQUIRE(c(0,0,2,1) == 2.5);
    REQUIRE(c(0,0,2,2) == 1.0);

    REQUIRE(c(0,1,0,0) == 5.0);
    REQUIRE(c(0,1,0,1) == 16.0);
    REQUIRE(c(0,1,0,2) == 12.0);

    REQUIRE(c(0,1,1,0) == 12.0);
    REQUIRE(c(0,1,1,1) == 38.0);
    REQUIRE(c(0,1,1,2) == 28.0);

    REQUIRE(c(0,1,2,0) == 7.0);
    REQUIRE(c(0,1,2,1) == 22.0);
    REQUIRE(c(0,1,2,2) == 16.0);
}

//}}}


TEMPLATE_TEST_CASE_2( "convmtx/convmtx_1", "convmtx", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};

    auto c = convmtx(a, 4);

    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 1) == 2);
    REQUIRE(c(0, 2) == 3);
    REQUIRE(c(0, 3) == 0);
    REQUIRE(c(0, 4) == 0);
    REQUIRE(c(0, 5) == 0);

    REQUIRE(c(1, 0) == 0);
    REQUIRE(c(1, 1) == 1);
    REQUIRE(c(1, 2) == 2);
    REQUIRE(c(1, 3) == 3);
    REQUIRE(c(1, 4) == 0);
    REQUIRE(c(1, 5) == 0);

    REQUIRE(c(2, 0) == 0);
    REQUIRE(c(2, 1) == 0);
    REQUIRE(c(2, 2) == 1);
    REQUIRE(c(2, 3) == 2);
    REQUIRE(c(2, 4) == 3);
    REQUIRE(c(2, 5) == 0);

    REQUIRE(c(3, 0) == 0);
    REQUIRE(c(3, 1) == 0);
    REQUIRE(c(3, 2) == 0);
    REQUIRE(c(3, 3) == 1);
    REQUIRE(c(3, 4) == 2);
    REQUIRE(c(3, 5) == 3);
}

TEMPLATE_TEST_CASE_2( "convmtx/convmtx_2", "convmtx", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 4, 6> c;

    c = convmtx(a, 4);

    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 1) == 2);
    REQUIRE(c(0, 2) == 3);
    REQUIRE(c(0, 3) == 0);
    REQUIRE(c(0, 4) == 0);
    REQUIRE(c(0, 5) == 0);

    REQUIRE(c(1, 0) == 0);
    REQUIRE(c(1, 1) == 1);
    REQUIRE(c(1, 2) == 2);
    REQUIRE(c(1, 3) == 3);
    REQUIRE(c(1, 4) == 0);
    REQUIRE(c(1, 5) == 0);

    REQUIRE(c(2, 0) == 0);
    REQUIRE(c(2, 1) == 0);
    REQUIRE(c(2, 2) == 1);
    REQUIRE(c(2, 3) == 2);
    REQUIRE(c(2, 4) == 3);
    REQUIRE(c(2, 5) == 0);

    REQUIRE(c(3, 0) == 0);
    REQUIRE(c(3, 1) == 0);
    REQUIRE(c(3, 2) == 0);
    REQUIRE(c(3, 3) == 1);
    REQUIRE(c(3, 4) == 2);
    REQUIRE(c(3, 5) == 3);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_1", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 1, 1> K(etl::magic<Z>(1));
    etl::fast_matrix<Z, 3*3, 1> C;

    C = convmtx2(I, 1, 1);

    REQUIRE(C(0, 0) == 8);
    REQUIRE(C(1, 0) == 3);
    REQUIRE(C(2, 0) == 4);
    REQUIRE(C(3, 0) == 1);
    REQUIRE(C(4, 0) == 5);
    REQUIRE(C(5, 0) == 9);
    REQUIRE(C(6, 0) == 6);
    REQUIRE(C(7, 0) == 7);
    REQUIRE(C(8, 0) == 2);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_2", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 2, 2> K(etl::magic<Z>(2));
    etl::fast_matrix<Z, 4*4, 4> C;

    C = convmtx2(I, 2, 2);

    REQUIRE(C(0, 0) == 8);
    REQUIRE(C(0, 1) == 0);
    REQUIRE(C(0, 2) == 0);
    REQUIRE(C(0, 3) == 0);

    REQUIRE(C(1, 0) == 3);
    REQUIRE(C(1, 1) == 8);
    REQUIRE(C(1, 2) == 0);
    REQUIRE(C(1, 3) == 0);

    REQUIRE(C(2, 0) == 4);
    REQUIRE(C(2, 1) == 3);
    REQUIRE(C(2, 2) == 0);
    REQUIRE(C(2, 3) == 0);

    REQUIRE(C(3, 0) == 0);
    REQUIRE(C(3, 1) == 4);
    REQUIRE(C(3, 2) == 0);
    REQUIRE(C(3, 3) == 0);

    REQUIRE(C(10, 0) == 2);
    REQUIRE(C(10, 1) == 7);
    REQUIRE(C(10, 2) == 9);
    REQUIRE(C(10, 3) == 5);

    REQUIRE(C(15, 0) == 0);
    REQUIRE(C(15, 1) == 0);
    REQUIRE(C(15, 2) == 0);
    REQUIRE(C(15, 3) == 2);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_3", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 2, 2> K(etl::magic<Z>(2));

    etl::fast_matrix<Z, 4*4, 4> C(convmtx2(I, 2, 2));

    etl::fast_matrix<Z, 4, 4> c1(etl::conv_2d_full(I, K));
    etl::fast_matrix<Z, 16, 1> c2(etl::mul(C, etl::reshape<4, 1>(etl::transpose(K))));
    etl::fast_matrix<Z, 4, 4> c3(etl::transpose(etl::reshape<4, 4>(c2)));

    for(std::size_t i = 0; i < c1.size(); ++i){
        REQUIRE(c1[i] == c3[i]);
    }
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_4", "convmtx conv", Z, double, float ) {
    etl::dyn_matrix<Z> I(etl::magic<Z>(7));
    etl::dyn_matrix<Z> K(etl::magic<Z>(5));

    etl::dyn_matrix<Z> C(convmtx2(I, 5, 5));

    etl::dyn_matrix<Z> c1(etl::conv_2d_full(I, K));
    etl::dyn_matrix<Z> c2(etl::mul(C, etl::reshape<25, 1>(etl::transpose(K))));
    etl::dyn_matrix<Z> c3(etl::transpose(etl::reshape(c2, 11, 11)));

    for(std::size_t i = 0; i < c1.size(); ++i){
        REQUIRE(c1[i] == c3[i]);
    }
}

TEMPLATE_TEST_CASE_2( "im2col/im2col_1", "im2col", Z, double, float ) {
    etl::dyn_matrix<Z> I(4, 4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<Z> C(4, 9);;

    etl::im2col_direct(C, I, 2, 2);

    REQUIRE(C(0,0) == 1);
    REQUIRE(C(1,0) == 5);
    REQUIRE(C(2,0) == 2);
    REQUIRE(C(3,0) == 6);

    REQUIRE(C(0,2) == 9);
    REQUIRE(C(1,2) == 13);
    REQUIRE(C(2,2) == 10);
    REQUIRE(C(3,2) == 14);

    REQUIRE(C(0,5) == 10);
    REQUIRE(C(1,5) == 14);
    REQUIRE(C(2,5) == 11);
    REQUIRE(C(3,5) == 15);

    REQUIRE(C(0,8) == 11);
    REQUIRE(C(1,8) == 15);
    REQUIRE(C(2,8) == 12);
    REQUIRE(C(3,8) == 16);
}

TEMPLATE_TEST_CASE_2( "im2col/im2col_2", "im2col", Z, double, float ) {
    etl::dyn_matrix<Z> I(3, 3, etl::values(1,2,3,4,5,6,7,8,9));
    etl::dyn_matrix<Z> C(4, 4);;

    etl::im2col_direct(C, I, 2, 2);

    REQUIRE(C(0,0) == 1);
    REQUIRE(C(1,0) == 4);
    REQUIRE(C(2,0) == 2);
    REQUIRE(C(3,0) == 5);

    REQUIRE(C(0,1) == 4);
    REQUIRE(C(1,1) == 7);
    REQUIRE(C(2,1) == 5);
    REQUIRE(C(3,1) == 8);

    REQUIRE(C(0,2) == 2);
    REQUIRE(C(1,2) == 5);
    REQUIRE(C(2,2) == 3);
    REQUIRE(C(3,2) == 6);

    REQUIRE(C(0,3) == 5);
    REQUIRE(C(1,3) == 8);
    REQUIRE(C(2,3) == 6);
    REQUIRE(C(3,3) == 9);
}

TEMPLATE_TEST_CASE_2( "im2col/im2col_3", "im2col", Z, double, float ) {
    etl::dyn_matrix<Z> I(3, 3, etl::values(1,2,3,4,5,6,7,8,9));
    etl::dyn_matrix<Z> C(2, 6);;

    etl::im2col_direct(C, I, 2, 1);

    REQUIRE(C(0,0) == 1);
    REQUIRE(C(1,0) == 4);

    REQUIRE(C(0,1) == 4);
    REQUIRE(C(1,1) == 7);

    REQUIRE(C(0,2) == 2);
    REQUIRE(C(1,2) == 5);

    REQUIRE(C(0,3) == 5);
    REQUIRE(C(1,3) == 8);

    REQUIRE(C(0,4) == 3);
    REQUIRE(C(1,4) == 6);

    REQUIRE(C(0,5) == 6);
    REQUIRE(C(1,5) == 9);
}

TEMPLATE_TEST_CASE_2( "im2col/im2col_4", "im2col", Z, double, float ) {
    etl::dyn_matrix<Z> I(3, 3, etl::values(1,2,3,4,5,6,7,8,9));
    etl::dyn_matrix<Z> C(2, 6);;

    etl::im2col_direct(C, I, 1, 2);

    REQUIRE(C(0,0) == 1);
    REQUIRE(C(1,0) == 2);

    REQUIRE(C(0,1) == 4);
    REQUIRE(C(1,1) == 5);

    REQUIRE(C(0,2) == 7);
    REQUIRE(C(1,2) == 8);

    REQUIRE(C(0,3) == 2);
    REQUIRE(C(1,3) == 3);

    REQUIRE(C(0,4) == 5);
    REQUIRE(C(1,4) == 6);

    REQUIRE(C(0,5) == 8);
    REQUIRE(C(1,5) == 9);
}
