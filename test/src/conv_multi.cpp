//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

// conv_2d_valid_multi

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/1", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5};

    etl::fast_matrix<Z, 2, 2, 2> c_1;
    etl::fast_matrix<Z, 2, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/2", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<Z, 2, 2, 2> c_1;
    etl::fast_matrix<Z, 2, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/3", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<Z, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<Z, 3, 2, 2> c_1;
    etl::fast_matrix<Z, 3, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/4", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 5, 5> I(etl::magic<Z>(5));
    etl::fast_matrix<Z, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<Z, 3, 4, 4> c_1;
    etl::fast_matrix<Z, 3, 4, 4> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/5", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 5, 5> I(etl::magic<Z>(5));
    etl::fast_matrix<Z, 3, 3, 3> K;

    K(0) = etl::magic<Z>(3);
    K(1) = etl::magic<Z>(3) * 2.0;
    K(2) = etl::magic<Z>(3) * etl::magic<Z>(3);

    etl::fast_matrix<Z, 3, 3, 3> c_1;
    etl::fast_matrix<Z, 3, 3, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/6", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 7, 7> I(etl::magic<Z>(7));
    etl::fast_matrix<Z, 3, 5, 3> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<Z, 3, 3, 5> c_1;
    etl::fast_matrix<Z, 3, 3, 5> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/7", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<Z, 3, 5, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<Z, 3, 5, 3> c_1;
    etl::fast_matrix<Z, 3, 5, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/8", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<Z, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<Z, 3, 7, 3> c_1;
    etl::fast_matrix<Z, 3, 7, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    conv_2d_valid_multi(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}

TEMPLATE_TEST_CASE_2("conv_2d_valid_multi/9", "[conv] [conv2]", Z, double, float) {
    etl::fast_matrix<Z, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<Z, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<Z, 3, 7, 3> c_1;
    etl::fast_matrix<Z, 3, 7, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    K(0).fflip_inplace();
    K(1).fflip_inplace();
    K(2).fflip_inplace();

    conv_2d_valid_multi_flipped(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE(c_1[i] == Approx(c_2[i]));
    }
}
