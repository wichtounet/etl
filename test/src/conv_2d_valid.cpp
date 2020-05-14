//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

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
    etl::fast_matrix<T, 3, 3> a;
    etl::fast_matrix<T, 2, 2> b;
    etl::fast_matrix<T, 2, 2> c;

    a = etl::magic(3);
    b = etl::magic(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 34);
    REQUIRE_EQUALS(c(0, 1), 48);

    REQUIRE_EQUALS(c(1, 0), 47);
    REQUIRE_EQUALS(c(1, 1), 67);
}

CONV2_VALID_TEST_CASE("convolution_2d/valid_5", "convolution_2d_valid") {
    etl::fast_matrix<T, 5, 5> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 3, 3> c;

    a = etl::magic(5);
    b = etl::magic(3);

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
    etl::fast_matrix<T, 17, 17> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 15, 15> c;

    a = etl::magic(17);
    b = etl::magic(3);

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
    etl::fast_matrix<T, 33, 33> a;
    etl::fast_matrix<T, 9, 9> b;
    etl::fast_matrix<T, 25, 25> c;

    a = etl::magic(33);
    b = etl::magic(9);

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
    etl::fast_matrix<T, 17, 17> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 15, 15> c;

    a = etl::magic(17);
    b = fflip(etl::magic(3));

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

/* Mixed tests */

ETL_TEST_CASE("conv2/valid/mixed/0", "convolution_2d_valid") {
    etl::fast_matrix<float, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<float, 2, 2> c;

    c = etl::conv_2d_valid(a, b);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
}

ETL_TEST_CASE("conv2/valid/mixed/1", "convolution_2d_valid") {
    etl::fast_matrix<float, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix_cm<float, 2, 2> b = {2.0, 0.5, 0.0, 0.5};
    etl::fast_matrix<float, 2, 2> c;

    c = etl::conv_2d_valid(a, b);

    REQUIRE_EQUALS(c(0, 0), 3.5);
    REQUIRE_EQUALS(c(0, 1), 4.5);

    REQUIRE_EQUALS(c(1, 0), 4.5);
    REQUIRE_EQUALS(c(1, 1), 3.0);
}
