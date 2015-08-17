//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

// convolution_2d_full

CONV2_FULL_TEST_CASE( "convolution_2d/full_1", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == Approx(T(2.0)));
    REQUIRE(c(0,1) == Approx(T(4.0)));
    REQUIRE(c(0,2) == Approx(T(6.0)));
    REQUIRE(c(0,3) == Approx(T(0.0)));

    REQUIRE(c(1,0) == Approx(T(0.5)));
    REQUIRE(c(1,1) == Approx(T(3.5)));
    REQUIRE(c(1,2) == Approx(T(4.5)));
    REQUIRE(c(1,3) == Approx(T(1.5)));

    REQUIRE(c(2,0) == Approx(T(6.0)));
    REQUIRE(c(2,1) == Approx(T(4.5)));
    REQUIRE(c(2,2) == Approx(T(3.0)));
    REQUIRE(c(2,3) == Approx(T(0.5)));

    REQUIRE(c(3,0) == Approx(T(1.5)));
    REQUIRE(c(3,1) == Approx(T(2.5)));
    REQUIRE(c(3,2) == Approx(T(1.5)));
    REQUIRE(c(3,3) == Approx(T(0.5)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_2", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 3> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == Approx(T(2.0)));
    REQUIRE(c(0,1) == Approx(T(4.0)));
    REQUIRE(c(0,2) == Approx(T(0.0)));

    REQUIRE(c(1,0) == Approx(T(0.5)));
    REQUIRE(c(1,1) == Approx(T(3.5)));
    REQUIRE(c(1,2) == Approx(T(1.0)));

    REQUIRE(c(2,0) == Approx(T(6.0)));
    REQUIRE(c(2,1) == Approx(T(4.5)));
    REQUIRE(c(2,2) == Approx(T(0.5)));

    REQUIRE(c(3,0) == Approx(T(1.5)));
    REQUIRE(c(3,1) == Approx(T(2.5)));
    REQUIRE(c(3,2) == Approx(T(1.0)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_3", "convolution_2d_full" ) {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == Approx(T(2.0)));
    REQUIRE(c(0,1) == Approx(T(5.0)));
    REQUIRE(c(0,2) == Approx(T(2.0)));

    REQUIRE(c(1,0) == Approx(T(6.5)));
    REQUIRE(c(1,1) == Approx(T(8.5)));
    REQUIRE(c(1,2) == Approx(T(3.0)));

    REQUIRE(c(2,0) == Approx(T(1.5)));
    REQUIRE(c(2,1) == Approx(T(2.5)));
    REQUIRE(c(2,2) == Approx(T(1.0)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_4", "convolution_2d_full" ) {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(8)));
    REQUIRE(c(0, 1) == Approx(T(25)));
    REQUIRE(c(0, 2) == Approx(T(9)));
    REQUIRE(c(0, 3) == Approx(T(18)));

    REQUIRE(c(1, 0) == Approx(T(35)));
    REQUIRE(c(1, 1) == Approx(T(34)));
    REQUIRE(c(1, 2) == Approx(T(48)));
    REQUIRE(c(1, 3) == Approx(T(33)));

    REQUIRE(c(2, 0) == Approx(T(16)));
    REQUIRE(c(2, 1) == Approx(T(47)));
    REQUIRE(c(2, 2) == Approx(T(67)));
    REQUIRE(c(2, 3) == Approx(T(20)));

    REQUIRE(c(3, 0) == Approx(T(16)));
    REQUIRE(c(3, 1) == Approx(T(44)));
    REQUIRE(c(3, 2) == Approx(T(26)));
    REQUIRE(c(3, 3) == Approx(T(4)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_5", "convolution_2d_full" ) {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 7, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(136)));
    REQUIRE(c(0, 1) == Approx(T(209)));
    REQUIRE(c(0, 2) == Approx(T(134)));
    REQUIRE(c(0, 3) == Approx(T(209)));

    REQUIRE(c(1, 0) == Approx(T(235)));
    REQUIRE(c(1, 1) == Approx(T(220)));
    REQUIRE(c(1, 2) == Approx(T(441)));
    REQUIRE(c(1, 3) == Approx(T(346)));

    REQUIRE(c(2, 0) == Approx(T(169)));
    REQUIRE(c(2, 1) == Approx(T(431)));
    REQUIRE(c(2, 2) == Approx(T(595)));
    REQUIRE(c(2, 3) == Approx(T(410)));

    REQUIRE(c(3, 0) == Approx(T(184)));
    REQUIRE(c(3, 1) == Approx(T(371)));
    REQUIRE(c(3, 2) == Approx(T(440)));
    REQUIRE(c(3, 3) == Approx(T(555)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_6", "convolution_2d_full" ) {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 19, 19> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(1240)));
    REQUIRE(c(0, 1) == Approx(T(1547)));
    REQUIRE(c(0, 2) == Approx(T(2648)));
    REQUIRE(c(0, 3) == Approx(T(2933)));

    REQUIRE(c(1, 0) == Approx(T(1849)));
    REQUIRE(c(1, 1) == Approx(T(3006)));
    REQUIRE(c(1, 2) == Approx(T(5452)));
    REQUIRE(c(1, 3) == Approx(T(6022)));

    REQUIRE(c(2, 0) == Approx(T(2667)));
    REQUIRE(c(2, 1) == Approx(T(5403)));
    REQUIRE(c(2, 2) == Approx(T(8640)));
    REQUIRE(c(2, 3) == Approx(T(9495)));

    REQUIRE(c(3, 0) == Approx(T(2937)));
    REQUIRE(c(3, 1) == Approx(T(5943)));
    REQUIRE(c(3, 2) == Approx(T(9450)));
    REQUIRE(c(3, 3) == Approx(T(10305)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_7", "convolution_2d_full" ) {
    etl::fast_matrix<T, 2, 6> a = {1,2,3,4,5,6,7,8,9,10,11,12};
    etl::fast_matrix<T, 2, 2> b = {1,2,3,4};
    etl::fast_matrix<T, 3, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(1)));
    REQUIRE(c(0, 1) == Approx(T(4)));
    REQUIRE(c(0, 2) == Approx(T(7)));
    REQUIRE(c(0, 3) == Approx(T(10)));
    REQUIRE(c(0, 4) == Approx(T(13)));
    REQUIRE(c(0, 5) == Approx(T(16)));
    REQUIRE(c(0, 6) == Approx(T(12)));

    REQUIRE(c(1, 0) == Approx(T(10)));
    REQUIRE(c(1, 1) == Approx(T(32)));
    REQUIRE(c(1, 2) == Approx(T(42)));
    REQUIRE(c(1, 3) == Approx(T(52)));
    REQUIRE(c(1, 4) == Approx(T(62)));
    REQUIRE(c(1, 5) == Approx(T(72)));
    REQUIRE(c(1, 6) == Approx(T(48)));

    REQUIRE(c(2, 0) == Approx(T(21)));
    REQUIRE(c(2, 1) == Approx(T(52)));
    REQUIRE(c(2, 2) == Approx(T(59)));
    REQUIRE(c(2, 3) == Approx(T(66)));
    REQUIRE(c(2, 4) == Approx(T(73)));
    REQUIRE(c(2, 5) == Approx(T(80)));
    REQUIRE(c(2, 6) == Approx(T(48)));
}

CONV2_FULL_TEST_CASE( "convolution_2d/full_8", "convolution_2d_full" ) {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 41, 41> c;

    Impl::apply(a, b, c);

    CHECK(c(0, 0) == Approx(T(26461)).epsilon(5.0));
    CHECK(c(0, 1) == Approx(T(60760)).epsilon(5.0));
    CHECK(c(0, 2) == Approx(T(103282)).epsilon(5.0));
    CHECK(c(0, 3) == Approx(T(154412)).epsilon(5.0));

    CHECK(c(1, 0) == Approx(T(60150)).epsilon(5.0));
    CHECK(c(1, 1) == Approx(T(136700)).epsilon(5.0));
    CHECK(c(1, 2) == Approx(T(230420)).epsilon(5.0));
    CHECK(c(1, 3) == Approx(T(296477)).epsilon(5.0));

    CHECK(c(2, 0) == Approx(T(101407)).epsilon(5.0));
    CHECK(c(2, 1) == Approx(T(228500)).epsilon(5.0));
    CHECK(c(2, 2) == Approx(T(336831)).epsilon(5.0));
    CHECK(c(2, 3) == Approx(T(416899)).epsilon(5.0));

    CHECK(c(3, 0) == Approx(T(150572)).epsilon(5.0));
    CHECK(c(3, 1) == Approx(T(291237)).epsilon(5.0));
    CHECK(c(3, 2) == Approx(T(417946)).epsilon(5.0));
    CHECK(c(3, 3) == Approx(T(516210)).epsilon(5.0));
}

// convolution_2d_same

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


// convolution_2d_valid

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

// convolution_subs

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

TEMPLATE_TEST_CASE_2( "convolution/1", "[conv]", Z, float, double) {
    //Test for bugfix: conv_2d_expr dimensions were not working for more than 1 dimensions

    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<Z, 4, 4> c;

    c = conv_2d_full(a, b) + conv_2d_full(a,b) * 2;

    REQUIRE(c(0,0) == 6.0);
    REQUIRE(c(0,1) == 12.0);
    REQUIRE(c(0,2) == 18.0);
    REQUIRE(c(0,3) == 0.0);

    REQUIRE(c(1,0) == 1.5);
    REQUIRE(c(1,1) == 10.5);
    REQUIRE(c(1,2) == 13.5);
    REQUIRE(c(1,3) == 4.5);

    REQUIRE(c(2,0) == 18.0);
    REQUIRE(c(2,1) == 13.5);
    REQUIRE(c(2,2) == 9.0);
    REQUIRE(c(2,3) == 1.5);

    REQUIRE(c(3,0) == 4.5);
    REQUIRE(c(3,1) == 7.5);
    REQUIRE(c(3,2) == 4.5);
    REQUIRE(c(3,3) == 1.5);
}

TEMPLATE_TEST_CASE_2( "convolution/2", "[conv][dyn]", Z, float, double) {
    //Test for bugfix: conv_2d_expr dimensions were not working for more than 1 dimensions

    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.0, 0.0, 0.5, 0.5));
    etl::dyn_matrix<Z> c(4,4);

    c = conv_2d_full(a, b) + conv_2d_full(a,b) * 2;

    REQUIRE(c(0,0) == 6.0);
    REQUIRE(c(0,1) == 12.0);
    REQUIRE(c(0,2) == 18.0);
    REQUIRE(c(0,3) == 0.0);

    REQUIRE(c(1,0) == 1.5);
    REQUIRE(c(1,1) == 10.5);
    REQUIRE(c(1,2) == 13.5);
    REQUIRE(c(1,3) == 4.5);

    REQUIRE(c(2,0) == 18.0);
    REQUIRE(c(2,1) == 13.5);
    REQUIRE(c(2,2) == 9.0);
    REQUIRE(c(2,3) == 1.5);

    REQUIRE(c(3,0) == 4.5);
    REQUIRE(c(3,1) == 7.5);
    REQUIRE(c(3,2) == 4.5);
    REQUIRE(c(3,3) == 1.5);
}
