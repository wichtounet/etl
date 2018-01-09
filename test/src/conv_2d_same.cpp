//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

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
    etl::fast_matrix<T, 3, 3> a;
    etl::fast_matrix<T, 2, 2> b;
    etl::fast_matrix<T, 3, 3> c;

    a = etl::magic(3);
    b = etl::magic(2);

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
    etl::fast_matrix<T, 5, 5> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 5, 5> c;

    a = etl::magic(5);
    b = etl::magic(3);

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
    etl::fast_matrix<T, 17, 17> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 17, 17> c;

    a = etl::magic(17);
    b = etl::magic(3);

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
    etl::fast_matrix<T, 33, 33> a;
    etl::fast_matrix<T, 9, 9> b;
    etl::fast_matrix<T, 33, 33> c;

    a = etl::magic(33);
    b = etl::magic(9);

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
    etl::fast_matrix<T, 33, 33> a;
    etl::fast_matrix<T, 9, 9> b;
    etl::fast_matrix<T, 33, 33> c;

    a = etl::magic(33);
    b = fflip(etl::magic(9));

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

// convolution_subs

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

// Mixed tests

TEST_CASE("conv2/same/mixed/0", "convolution_2d_same") {
    etl::fast_matrix<float, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<float, 3, 3> c;

    c = conv_2d_same(a, b);

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

TEST_CASE("conv2/same/mixed/1", "convolution_2d_same") {
    etl::fast_matrix<float, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix_cm<float, 2, 2> b = {2.0, 0.5, 0.0, 0.5};
    etl::fast_matrix<float, 3, 3> c;

    c = conv_2d_same(a, b);

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
