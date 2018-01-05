//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_2d_valid tests with stride and/or padding

CONV2_VALID_TEST_CASE("conv/2/stride/valid/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::template apply<2, 2>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(27.5));
}

CONV2_VALID_TEST_CASE("conv/2/stride/valid/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    Impl::template apply<1, 1, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4.5));
}

CONV2_VALID_TEST_CASE("conv/2/stride/valid/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::template apply<2, 2, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(2.0));
}

// conv_2d_valid_flipped tests with stride and/or padding

CONV2_VALID_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 2, 2> c;

    Impl::template apply<2, 2>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(27.5));
}

CONV2_VALID_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 4, 4> c;

    Impl::template apply<1, 1, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4.5));
}

CONV2_VALID_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 2, 2> c;

    Impl::template apply<2, 2, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(2.0));
}

// conv_2d_valid tests with dynamic stride and/or padding

DYN_CONV2_VALID_TEST_CASE("conv/2/dyn_stride/valid/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::template apply(a, b, c, 2, 2);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(27.5));
}

DYN_CONV2_VALID_TEST_CASE("conv/2/dyn_stride/valid/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c, 1, 1, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4.5));
}

DYN_CONV2_VALID_TEST_CASE("conv/2/dyn_stride/valid/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2> b = {1.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c, 2, 2, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(2.0));
}

// conv_2d_valid_flipped tests with dynamic stride and/or padding

DYN_CONV2_VALID_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c, 2, 2);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(27.5));
}

DYN_CONV2_VALID_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 4, 4> c;

    Impl::apply(a, b, c, 1, 1, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4.5));
}

DYN_CONV2_VALID_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2> b = {0.5, 0.5, 0.0, 1.0};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c, 2, 2, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(2.0));
}

// conv_2d_valid_multi tests with stride and/or padding

CONV2_VALID_MULTI_TEST_CASE("conv/2/stride/valid/multi/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply<2, 2>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(27.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 27.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 27.5));
}

CONV2_VALID_MULTI_TEST_CASE("conv/2/stride/valid/multi/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 4, 4> c;

    Impl::template apply<1, 1, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0, 2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(0, 3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 3), T(4.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 2), T(2.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 3), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 3), T(2.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(1, 2, 0), T(2.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1), T(2.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 2), T(2.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 3), T(2.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(1, 3, 0), T(2.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 1), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 3), T(2.0 * 4.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 2), T(3.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 3), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 3), T(3.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(2, 2, 0), T(3.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(2, 2, 1), T(3.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 2), T(3.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 3), T(3.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(2, 3, 0), T(3.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 1), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 3), T(3.0 * 4.5));
}

CONV2_VALID_MULTI_TEST_CASE("conv/2/stride/valid/multi/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply<2, 2, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(2.0));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 2.0));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 2.0));
}

// conv_2d_valid_multi_flipped tests with stride and/or padding

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/multi/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply<2, 2>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(27.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 27.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 27.5));
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/multi/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 4, 4> c;

    Impl::template apply<1, 1, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0, 2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(0, 3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 3), T(4.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 2), T(2.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 3), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 3), T(2.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(1, 2, 0), T(2.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1), T(2.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 2), T(2.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 3), T(2.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(1, 3, 0), T(2.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 1), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 3), T(2.0 * 4.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 2), T(3.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 3), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 3), T(3.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(2, 2, 0), T(3.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(2, 2, 1), T(3.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 2), T(3.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 3), T(3.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(2, 3, 0), T(3.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 1), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 3), T(3.0 * 4.5));
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/stride/valid/flipped/multi/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply<2, 2, 1, 1>(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(2.0));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 2.0));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 2.0));
}

// conv_2d_valid_multi tests with dynamic stride and/or padding

DYN_CONV2_VALID_MULTI_TEST_CASE("conv/2/dyn_stride/valid/multi/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 2, 2> c;

    auto expr = etl::conv_2d_valid_multi(a, b, 2, 2, 0, 0);

    Impl::template apply(a, b, c, 2, 2);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(27.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 27.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 27.5));
}

DYN_CONV2_VALID_MULTI_TEST_CASE("conv/2/dyn_stride/valid/multi/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 4, 4> c;

    Impl::template apply(a, b, c, 1, 1, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0, 2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(0, 3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 3), T(4.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 2), T(2.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 3), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 3), T(2.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(1, 2, 0), T(2.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1), T(2.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 2), T(2.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 3), T(2.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(1, 3, 0), T(2.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 1), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 3), T(2.0 * 4.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 2), T(3.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 3), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 3), T(3.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(2, 2, 0), T(3.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(2, 2, 1), T(3.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 2), T(3.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 3), T(3.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(2, 3, 0), T(3.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 1), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 3), T(3.0 * 4.5));
}

DYN_CONV2_VALID_MULTI_TEST_CASE("conv/2/dyn_stride/valid/multi/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> b = {1.0, 0.0, 0.5, 0.5, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.5, 1.5};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply(a, b, c, 2, 2, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(2.0));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 2.0));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 2.0));
}

// conv_2d_valid_multi_flipped tests with dynamic stride and/or padding

DYN_CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/multi/1", "[conv][stride]") {
    etl::fast_matrix<T, 4, 4> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply(a, b, c, 2, 2);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(11.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(23.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(27.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 27.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 11.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 23.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 27.5));
}

DYN_CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/multi/2", "[conv][stride]") {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 4, 4> c;

    Impl::template apply(a, b, c, 1, 1, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(4.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(6.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0, 2, 0), T(9.0));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1), T(12.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 2), T(14.5));
    REQUIRE_EQUALS_APPROX(c(0, 2, 3), T(3.0));

    REQUIRE_EQUALS_APPROX(c(0, 3, 0), T(3.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 1), T(7.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 2), T(8.5));
    REQUIRE_EQUALS_APPROX(c(0, 3, 3), T(4.5));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 2), T(2.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 3), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 3), T(2.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(1, 2, 0), T(2.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1), T(2.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 2), T(2.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(1, 2, 3), T(2.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(1, 3, 0), T(2.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 1), T(2.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 2), T(2.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(1, 3, 3), T(2.0 * 4.5));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 2.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 2), T(3.0 * 3.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 3), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 4.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 6.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 3), T(3.0 * 1.5));

    REQUIRE_EQUALS_APPROX(c(2, 2, 0), T(3.0 * 9.0));
    REQUIRE_EQUALS_APPROX(c(2, 2, 1), T(3.0 * 12.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 2), T(3.0 * 14.5));
    REQUIRE_EQUALS_APPROX(c(2, 2, 3), T(3.0 * 3.0));

    REQUIRE_EQUALS_APPROX(c(2, 3, 0), T(3.0 * 3.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 1), T(3.0 * 7.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 2), T(3.0 * 8.5));
    REQUIRE_EQUALS_APPROX(c(2, 3, 3), T(3.0 * 4.5));
}

DYN_CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv/2/dyn_stride/valid/flipped/multi/3", "[conv][stride]") {
    etl::fast_matrix<T, 2, 2> a = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> b = {0.5, 0.5, 0.0, 1.0,  1.0, 1.0, 0.0, 2.0,  1.5, 1.5, 0.0, 3.0};
    etl::fast_matrix<T, 3, 2, 2> c;

    Impl::template apply(a, b, c, 2, 2, 1, 1);

    REQUIRE_EQUALS_APPROX(c(0, 0, 0), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1), T(2.0));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0), T(2.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1), T(2.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(1, 1, 0), T(2.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1), T(2.0 * 2.0));

    REQUIRE_EQUALS_APPROX(c(2, 0, 0), T(3.0 * 1.0));
    REQUIRE_EQUALS_APPROX(c(2, 0, 1), T(3.0 * 0.0));

    REQUIRE_EQUALS_APPROX(c(2, 1, 0), T(3.0 * 1.5));
    REQUIRE_EQUALS_APPROX(c(2, 1, 1), T(3.0 * 2.0));
}
