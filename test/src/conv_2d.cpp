//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

// convolution_2d_full

CONV2_FULL_TEST_CASE("convolution_2d/full_1", "convolution_2d_full") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(0.5));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_2", "convolution_2d_full") {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(1.0));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.0));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_3", "convolution_2d_full") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(5.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(2.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(6.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(8.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(3.0));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(1.0));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_4", "convolution_2d_full") {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(8));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(25));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(9));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(18));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(35));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(34));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(48));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(33));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(16));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(47));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(67));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(20));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(16));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(44));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(26));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_5", "convolution_2d_full") {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 7, 7> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(136));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(209));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(134));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(209));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(235));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(220));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(441));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(346));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(169));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(431));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(595));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(410));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(184));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(371));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(440));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(555));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_6", "convolution_2d_full") {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 19, 19> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1240));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(1547));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(2648));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(2933));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(1849));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3006));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(5452));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(6022));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(2667));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(5403));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(8640));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(9495));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(2937));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(5943));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(9450));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(10305));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_7", "convolution_2d_full") {
    etl::fast_matrix<T, 2, 6> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 2> b = {1, 2, 3, 4};
    etl::fast_matrix<T, 3, 7> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(7));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(10));
    REQUIRE_EQUALS_APPROX(c(0, 4), T(13));
    REQUIRE_EQUALS_APPROX(c(0, 5), T(16));
    REQUIRE_EQUALS_APPROX(c(0, 6), T(12));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(10));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(32));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(42));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(52));
    REQUIRE_EQUALS_APPROX(c(1, 4), T(62));
    REQUIRE_EQUALS_APPROX(c(1, 5), T(72));
    REQUIRE_EQUALS_APPROX(c(1, 6), T(48));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(21));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(52));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(59));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(66));
    REQUIRE_EQUALS_APPROX(c(2, 4), T(73));
    REQUIRE_EQUALS_APPROX(c(2, 5), T(80));
    REQUIRE_EQUALS_APPROX(c(2, 6), T(48));
}

CONV2_FULL_TEST_CASE("convolution_2d/full_8", "convolution_2d_full") {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 41, 41> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX_E(c(0, 0), T(26461), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1), T(60760), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2), T(103282), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 3), T(154412), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0), T(60150), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1), T(136700), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2), T(230420), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 3), T(296477), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(2, 0), T(101407), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 1), T(228500), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 2), T(336831), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 3), T(416899), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(3, 0), T(150572), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 1), T(291237), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 2), T(417946), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 3), T(516210), base_eps * 10);
}

// convolution_2d_same

CONV2_SAME_TEST_CASE("convolution_2d/same_1", "convolution_2d_same") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);
    REQUIRE_EQUALS(c(0, 2), 1.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
    REQUIRE_EQUALS(c(1, 2), 0.5);

    REQUIRE_EQUALS(c(2, 0), 2.5);
    REQUIRE_EQUALS(c(2, 1), 1.5);
    REQUIRE_EQUALS(c(2, 2), 0.5);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_2", "convolution_2d_same") {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 3, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 1.0);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 0.5);

    REQUIRE_EQUALS(c(2, 0), 2.5);
    REQUIRE_EQUALS(c(2, 1), 1.0);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_3", "convolution_2d_same") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 8.5);
    REQUIRE_EQUALS(c(0, 1), 3.0);

    REQUIRE_EQUALS(c(1, 0), 2.5);
    REQUIRE_EQUALS(c(1, 1), 1.0);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_4", "convolution_2d_same") {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 34);
    REQUIRE_EQUALS(c(0, 1), 48);
    REQUIRE_EQUALS(c(0, 2), 33);

    REQUIRE_EQUALS(c(1, 0), 47);
    REQUIRE_EQUALS(c(1, 1), 67);
    REQUIRE_EQUALS(c(1, 2), 20);

    REQUIRE_EQUALS(c(2, 0), 44);
    REQUIRE_EQUALS(c(2, 1), 26);
    REQUIRE_EQUALS(c(2, 2), 4);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_5", "convolution_2d_same") {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 5, 5> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 220);
    REQUIRE_EQUALS(c(0, 1), 441);
    REQUIRE_EQUALS(c(0, 2), 346);
    REQUIRE_EQUALS(c(0, 3), 276);

    REQUIRE_EQUALS(c(1, 0), 431);
    REQUIRE_EQUALS(c(1, 1), 595);
    REQUIRE_EQUALS(c(1, 2), 410);
    REQUIRE_EQUALS(c(1, 3), 575);

    REQUIRE_EQUALS(c(2, 0), 371);
    REQUIRE_EQUALS(c(2, 1), 440);
    REQUIRE_EQUALS(c(2, 2), 555);
    REQUIRE_EQUALS(c(2, 3), 620);

    REQUIRE_EQUALS(c(3, 0), 301);
    REQUIRE_EQUALS(c(3, 1), 585);
    REQUIRE_EQUALS(c(3, 2), 600);
    REQUIRE_EQUALS(c(3, 3), 765);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_6", "convolution_2d_same") {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 17, 17> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3006);
    REQUIRE_EQUALS(c(0, 1), 5452);
    REQUIRE_EQUALS(c(0, 2), 6022);
    REQUIRE_EQUALS(c(0, 3), 6592);

    REQUIRE_EQUALS(c(1, 0), 5403);
    REQUIRE_EQUALS(c(1, 1), 8640);
    REQUIRE_EQUALS(c(1, 2), 9495);
    REQUIRE_EQUALS(c(1, 3), 10350);

    REQUIRE_EQUALS(c(2, 0), 5943);
    REQUIRE_EQUALS(c(2, 1), 9450);
    REQUIRE_EQUALS(c(2, 2), 10305);
    REQUIRE_EQUALS(c(2, 3), 11160);

    REQUIRE_EQUALS(c(3, 0), 6483);
    REQUIRE_EQUALS(c(3, 1), 10260);
    REQUIRE_EQUALS(c(3, 2), 11115);
    REQUIRE_EQUALS(c(3, 3), 9658);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_7", "convolution_2d_same") {
    etl::fast_matrix<T, 2, 6> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 2> b = {1, 2, 3, 4};
    etl::fast_matrix<T, 2, 6> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 32);
    REQUIRE_EQUALS(c(0, 1), 42);
    REQUIRE_EQUALS(c(0, 2), 52);
    REQUIRE_EQUALS(c(0, 3), 62);
    REQUIRE_EQUALS(c(0, 4), 72);
    REQUIRE_EQUALS(c(0, 5), 48);

    REQUIRE_EQUALS(c(1, 0), 52);
    REQUIRE_EQUALS(c(1, 1), 59);
    REQUIRE_EQUALS(c(1, 2), 66);
    REQUIRE_EQUALS(c(1, 3), 73);
    REQUIRE_EQUALS(c(1, 4), 80);
    REQUIRE_EQUALS(c(1, 5), 48);
}

CONV2_SAME_TEST_CASE("convolution_2d/same_8", "convolution_2d_same") {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 33, 33> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 676494);
    REQUIRE_EQUALS(c(0, 1), 806569);
    REQUIRE_EQUALS(c(0, 2), 976949);
    REQUIRE_EQUALS(c(0, 3), 1179119);

    REQUIRE_EQUALS(c(1, 0), 808354);
    REQUIRE_EQUALS(c(1, 1), 984480);
    REQUIRE_EQUALS(c(1, 2), 1206077);
    REQUIRE_EQUALS(c(1, 3), 1469155);

    REQUIRE_EQUALS(c(2, 0), 971394);
    REQUIRE_EQUALS(c(2, 1), 1202744);
    REQUIRE_EQUALS(c(2, 2), 1485149);
    REQUIRE_EQUALS(c(2, 3), 1773847);

    REQUIRE_EQUALS(c(3, 0), 1173020);
    REQUIRE_EQUALS(c(3, 1), 1464355);
    REQUIRE_EQUALS(c(3, 2), 1771896);
    REQUIRE_EQUALS(c(3, 3), 2091280);
}

// convolution_2d_same_flipped

CONV2_SAME_FLIPPED_TEST_CASE("convolution_2d_flipped/same_1", "convolution_2d_same") {
    etl::fast_matrix<T, 2, 6> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 2> b = {4, 3, 2, 1};
    etl::fast_matrix<T, 2, 6> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 32);
    REQUIRE_EQUALS(c(0, 1), 42);
    REQUIRE_EQUALS(c(0, 2), 52);
    REQUIRE_EQUALS(c(0, 3), 62);
    REQUIRE_EQUALS(c(0, 4), 72);
    REQUIRE_EQUALS(c(0, 5), 48);

    REQUIRE_EQUALS(c(1, 0), 52);
    REQUIRE_EQUALS(c(1, 1), 59);
    REQUIRE_EQUALS(c(1, 2), 66);
    REQUIRE_EQUALS(c(1, 3), 73);
    REQUIRE_EQUALS(c(1, 4), 80);
    REQUIRE_EQUALS(c(1, 5), 48);
}

CONV2_SAME_FLIPPED_TEST_CASE("convolution_2d_flipped/same_2", "convolution_2d_same") {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(fflip(etl::magic(9)));
    etl::fast_matrix<T, 33, 33> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 676494);
    REQUIRE_EQUALS(c(0, 1), 806569);
    REQUIRE_EQUALS(c(0, 2), 976949);
    REQUIRE_EQUALS(c(0, 3), 1179119);

    REQUIRE_EQUALS(c(1, 0), 808354);
    REQUIRE_EQUALS(c(1, 1), 984480);
    REQUIRE_EQUALS(c(1, 2), 1206077);
    REQUIRE_EQUALS(c(1, 3), 1469155);

    REQUIRE_EQUALS(c(2, 0), 971394);
    REQUIRE_EQUALS(c(2, 1), 1202744);
    REQUIRE_EQUALS(c(2, 2), 1485149);
    REQUIRE_EQUALS(c(2, 3), 1773847);

    REQUIRE_EQUALS(c(3, 0), 1173020);
    REQUIRE_EQUALS(c(3, 1), 1464355);
    REQUIRE_EQUALS(c(3, 2), 1771896);
    REQUIRE_EQUALS(c(3, 3), 2091280);
}

// convolution_2d_valid

CONV2_VALID_TEST_CASE("convolution_2d/valid_1", "convolution_2d_valid") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_2", "convolution_2d_valid") {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 1> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(1, 0), 4.5);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_3", "convolution_2d_valid") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 1.0, 0.5, 0.5};
    etl::fast_matrix<T, 1, 1> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 8.5);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_4", "convolution_2d_valid") {
    etl::fast_matrix<T, 3, 3> a(etl::magic(3));
    etl::fast_matrix<T, 2, 2> b(etl::magic(2));
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 34);
    REQUIRE_EQUALS(c(0, 1), 48);

    REQUIRE_EQUALS(c(1, 0), 47);
    REQUIRE_EQUALS(c(1, 1), 67);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_5", "convolution_2d_valid") {
    etl::fast_matrix<T, 5, 5> a(etl::magic(5));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 595);
    REQUIRE_EQUALS(c(0, 1), 410);
    REQUIRE_EQUALS(c(0, 2), 575);

    REQUIRE_EQUALS(c(1, 0), 440);
    REQUIRE_EQUALS(c(1, 1), 555);
    REQUIRE_EQUALS(c(1, 2), 620);

    REQUIRE_EQUALS(c(2, 0), 585);
    REQUIRE_EQUALS(c(2, 1), 600);
    REQUIRE_EQUALS(c(2, 2), 765);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_6", "convolution_2d_valid") {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(etl::magic(3));
    etl::fast_matrix<T, 15, 15> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 8640);
    REQUIRE_EQUALS(c(0, 1), 9495);
    REQUIRE_EQUALS(c(0, 2), 10350);
    REQUIRE_EQUALS(c(0, 3), 11205);

    REQUIRE_EQUALS(c(1, 0), 9450);
    REQUIRE_EQUALS(c(1, 1), 10305);
    REQUIRE_EQUALS(c(1, 2), 11160);
    REQUIRE_EQUALS(c(1, 3), 9703);

    REQUIRE_EQUALS(c(2, 0), 10260);
    REQUIRE_EQUALS(c(2, 1), 11115);
    REQUIRE_EQUALS(c(2, 2), 9658);
    REQUIRE_EQUALS(c(2, 3), 9357);

    REQUIRE_EQUALS(c(3, 0), 11070);
    REQUIRE_EQUALS(c(3, 1), 9613);
    REQUIRE_EQUALS(c(3, 2), 9312);
    REQUIRE_EQUALS(c(3, 3), 5832);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_7", "convolution_2d_valid") {
    etl::fast_matrix<T, 2, 6> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 2> b = {1, 2, 3, 4};
    etl::fast_matrix<T, 1, 5> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 32);
    REQUIRE_EQUALS(c(0, 1), 42);
    REQUIRE_EQUALS(c(0, 2), 52);
    REQUIRE_EQUALS(c(0, 3), 62);
    REQUIRE_EQUALS(c(0, 4), 72);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_8", "convolution_2d_valid") {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(etl::magic(9));
    etl::fast_matrix<T, 25, 25> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 2735136);
    REQUIRE_EQUALS(c(0, 1), 2726136);
    REQUIRE_EQUALS(c(0, 2), 2620215);
    REQUIRE_EQUALS(c(0, 3), 2394504);

    REQUIRE_EQUALS(c(1, 0), 2722815);
    REQUIRE_EQUALS(c(1, 1), 2616894);
    REQUIRE_EQUALS(c(1, 2), 2391183);
    REQUIRE_EQUALS(c(1, 3), 2473659);

    REQUIRE_EQUALS(c(2, 0), 2613573);
    REQUIRE_EQUALS(c(2, 1), 2387862);
    REQUIRE_EQUALS(c(2, 2), 2470338);
    REQUIRE_EQUALS(c(2, 3), 2493546);

    REQUIRE_EQUALS(c(3, 0), 2384541);
    REQUIRE_EQUALS(c(3, 1), 2467017);
    REQUIRE_EQUALS(c(3, 2), 2491776);
    REQUIRE_EQUALS(c(3, 3), 2432517);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_9", "convolution_2d_valid") {
    etl::dyn_matrix<T> a(3, 3, std::initializer_list<T>({1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0}));
    etl::dyn_matrix<T> b(2, 2, std::initializer_list<T>({2.0, 0.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(2, 2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_10", "convolution_2d_valid") {
    etl::dyn_matrix<T> a(3, 2, std::initializer_list<T>({1.0, 2.0, 0.0, 1.0, 3.0, 2.0}));
    etl::dyn_matrix<T> b(2, 2, std::initializer_list<T>({2.0, 0.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(2, 1);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(1, 0), 4.5);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_11", "convolution_2d_valid") {
    etl::dyn_matrix<T> a(2, 2, std::initializer_list<T>({1.0, 2.0, 3.0, 2.0}));
    etl::dyn_matrix<T> b(2, 2, std::initializer_list<T>({2.0, 1.0, 0.5, 0.5}));
    etl::dyn_matrix<T> c(1, 1);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 8.5);
}

// convolution_subs

CONV2_FULL_TEST_CASE("convolution_2d/sub_1", "convolution_2d_full") {
    etl::fast_matrix<T, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 1, 4, 4> c;

    Impl::apply(a(0), b(0), c(0));

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 2), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 2), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0, 2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 3), T(0.5));

    REQUIRE_EQUALS_APPROX(c(0, 3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 2), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 3), T(0.5));
}

CONV2_SAME_TEST_CASE("convolution_2d/sub_2", "convolution_2d_same") {
    etl::fast_matrix<T, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 1, 3, 3> c;

    Impl::apply(a(0), b(0), c(0));

    REQUIRE_EQUALS(c(0, 0, 0), T(3.5));
    REQUIRE_EQUALS(c(0, 0, 1), T(4.5));
    REQUIRE_EQUALS(c(0, 0, 2), T(1.5));

    REQUIRE_EQUALS(c(0, 1, 0), T(4.5));
    REQUIRE_EQUALS(c(0, 1, 1), T(3.0));
    REQUIRE_EQUALS(c(0, 1, 2), T(0.5));

    REQUIRE_EQUALS(c(0, 2, 0), T(2.5));
    REQUIRE_EQUALS(c(0, 2, 1), T(1.5));
    REQUIRE_EQUALS(c(0, 2, 2), T(0.5));
}

CONV2_VALID_TEST_CASE("convolution_2d/sub_3", "convolution_2d_valid") {
    etl::fast_matrix<T, 1, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 1, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 1, 2, 2> c;

    Impl::apply(a(0), b(0), c(0));

    REQUIRE_EQUALS(c(0, 0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 0, 1), 4.5);

    REQUIRE_EQUALS(c(0, 1, 0), 4.5);
    REQUIRE_EQUALS(c(0, 1, 1), 3.0);
}

TEMPLATE_TEST_CASE_2("convolution/1", "[conv]", Z, float, double) {
    //Test for bugfix: conv_2d_expr dimensions were not working for more than 1 dimensions

    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<Z, 4, 4> c;

    c = conv_2d_full(a, b) + conv_2d_full(a, b) * 2;

    REQUIRE_EQUALS(c(0, 0), 6.0);
    REQUIRE_EQUALS(c(0, 1), 12.0);
    REQUIRE_EQUALS(c(0, 2), 18.0);
    REQUIRE_EQUALS(c(0, 3), 0.0);

    REQUIRE_EQUALS(c(1, 0), 1.5);
    REQUIRE_EQUALS(c(1, 1), 10.5);
    REQUIRE_EQUALS(c(1, 2), 13.5);
    REQUIRE_EQUALS(c(1, 3), 4.5);

    REQUIRE_EQUALS(c(2, 0), 18.0);
    REQUIRE_EQUALS(c(2, 1), 13.5);
    REQUIRE_EQUALS(c(2, 2), 9.0);
    REQUIRE_EQUALS(c(2, 3), 1.5);

    REQUIRE_EQUALS(c(3, 0), 4.5);
    REQUIRE_EQUALS(c(3, 1), 7.5);
    REQUIRE_EQUALS(c(3, 2), 4.5);
    REQUIRE_EQUALS(c(3, 3), 1.5);
}

TEMPLATE_TEST_CASE_2("convolution/2", "[conv][dyn]", Z, float, double) {
    //Test for bugfix: conv_2d_expr dimensions were not working for more than 1 dimensions

    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.0, 0.0, 0.5, 0.5));
    etl::dyn_matrix<Z> c(4, 4);

    c = conv_2d_full(a, b) + conv_2d_full(a, b) * 2;

    REQUIRE_EQUALS(c(0, 0), 6.0);
    REQUIRE_EQUALS(c(0, 1), 12.0);
    REQUIRE_EQUALS(c(0, 2), 18.0);
    REQUIRE_EQUALS(c(0, 3), 0.0);

    REQUIRE_EQUALS(c(1, 0), 1.5);
    REQUIRE_EQUALS(c(1, 1), 10.5);
    REQUIRE_EQUALS(c(1, 2), 13.5);
    REQUIRE_EQUALS(c(1, 3), 4.5);

    REQUIRE_EQUALS(c(2, 0), 18.0);
    REQUIRE_EQUALS(c(2, 1), 13.5);
    REQUIRE_EQUALS(c(2, 2), 9.0);
    REQUIRE_EQUALS(c(2, 3), 1.5);

    REQUIRE_EQUALS(c(3, 0), 4.5);
    REQUIRE_EQUALS(c(3, 1), 7.5);
    REQUIRE_EQUALS(c(3, 2), 4.5);
    REQUIRE_EQUALS(c(3, 3), 1.5);
}

// conv_2d_valid_flipped

CONV2_VALID_FLIPPED_TEST_CASE("conv/2d/flipped/valid/1", "[conv][conv2][valid]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 2.0};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
}

CONV2_VALID_FLIPPED_TEST_CASE("conv/2d/flipped/valid/2", "[conv][conv2][valid]") {
    etl::fast_matrix<T, 17, 17> a(etl::magic(17));
    etl::fast_matrix<T, 3, 3> b(fflip(etl::magic(3)));
    etl::fast_matrix<T, 15, 15> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 8640);
    REQUIRE_EQUALS(c(0, 1), 9495);
    REQUIRE_EQUALS(c(0, 2), 10350);
    REQUIRE_EQUALS(c(0, 3), 11205);

    REQUIRE_EQUALS(c(1, 0), 9450);
    REQUIRE_EQUALS(c(1, 1), 10305);
    REQUIRE_EQUALS(c(1, 2), 11160);
    REQUIRE_EQUALS(c(1, 3), 9703);

    REQUIRE_EQUALS(c(2, 0), 10260);
    REQUIRE_EQUALS(c(2, 1), 11115);
    REQUIRE_EQUALS(c(2, 2), 9658);
    REQUIRE_EQUALS(c(2, 3), 9357);

    REQUIRE_EQUALS(c(3, 0), 11070);
    REQUIRE_EQUALS(c(3, 1), 9613);
    REQUIRE_EQUALS(c(3, 2), 9312);
    REQUIRE_EQUALS(c(3, 3), 5832);
}

// conv_2d_full_flipped

CONV2_FULL_FLIPPED_TEST_CASE("conv/2d/full/flipped/1", "[conv][conv2][full]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 2.0};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(0.5));
}

CONV2_FULL_FLIPPED_TEST_CASE("conv/2d/full/flipped/2", "[conv][conv2][full]") {
    etl::fast_matrix<T, 3, 2> a = {1.0, 2.0, 0.0, 1.0, 3.0, 2.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 2.0};
    etl::fast_matrix<T, 4, 3> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(1.0));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.0));
}

CONV2_FULL_FLIPPED_TEST_CASE("conv/2d/full/flipped/3", "convolution_2d_full") {
    etl::fast_matrix<T, 33, 33> a(etl::magic(33));
    etl::fast_matrix<T, 9, 9> b(fflip(etl::magic(9)));
    etl::fast_matrix<T, 41, 41> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX_E(c(0, 0), T(26461), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1), T(60760), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2), T(103282), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 3), T(154412), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0), T(60150), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1), T(136700), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2), T(230420), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 3), T(296477), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(2, 0), T(101407), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 1), T(228500), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 2), T(336831), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 3), T(416899), base_eps * 10);

    REQUIRE_EQUALS_APPROX_E(c(3, 0), T(150572), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 1), T(291237), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 2), T(417946), base_eps * 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 3), T(516210), base_eps * 10);
}
