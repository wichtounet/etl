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

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/3", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<T, 3, 2, 2> c_1;
    etl::fast_matrix<T, 3, 2, 2> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/4", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I(etl::magic<T>(5));
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<T, 3, 4, 4> c_1;
    etl::fast_matrix<T, 3, 4, 4> c_2;
    etl::fast_matrix<T, 3, 4, 4> c_3;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/5", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I(etl::magic<T>(5));
    etl::fast_matrix<T, 3, 3, 3> K;

    K(0) = etl::magic<T>(3);
    K(1) = etl::magic<T>(3) * 2.0;
    K(2) = etl::magic<T>(3) * etl::magic<T>(3);

    etl::fast_matrix<T, 3, 3, 3> c_1;
    etl::fast_matrix<T, 3, 3, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/6", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 7, 7> I(etl::magic<T>(7));
    etl::fast_matrix<T, 3, 5, 3> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 3, 5> c_1;
    etl::fast_matrix<T, 3, 3, 5> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/7", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 5, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 5, 3> c_1;
    etl::fast_matrix<T, 3, 5, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/8", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 3> c_1;
    etl::fast_matrix<T, 3, 7, 3> c_2;
    etl::fast_matrix<T, 3, 7, 3> c_3;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 3> c_1;
    etl::fast_matrix<T, 3, 7, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 4, 3, 5> K;

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 7, 3> c_1;
    etl::fast_matrix<T, 4, 7, 3> c_2;

    c_1(0) = conv_2d_valid(I, K(0));
    c_1(1) = conv_2d_valid(I, K(1));
    c_1(2) = conv_2d_valid(I, K(2));
    c_1(3) = conv_2d_valid(I, K(3));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

//conv_2d_full_multi

CONV2_FULL_MULTI_TEST_CASE("conv_2d/full/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 5, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 13, 11> c_1;
    etl::fast_matrix<T, 3, 13, 11> c_2;

    c_1(0) = conv_2d_full(I, K(0));
    c_1(1) = conv_2d_full(I, K(1));
    c_1(2) = conv_2d_full(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_TEST_CASE("conv_2d/full/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 11, 11> c_1;
    etl::fast_matrix<T, 3, 11, 11> c_2;

    c_1(0) = conv_2d_full(I, K(0));
    c_1(1) = conv_2d_full(I, K(1));
    c_1(2) = conv_2d_full(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_FLIPPED_TEST_CASE("conv_2d/full/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 11, 11> c_1;
    etl::fast_matrix<T, 3, 11, 11> c_2;

    c_1(0) = conv_2d_full(I, K(0));
    c_1(1) = conv_2d_full(I, K(1));
    c_1(2) = conv_2d_full(I, K(2));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_FLIPPED_TEST_CASE("conv_2d/full/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 4, 3, 5> K;

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 11, 11> c_1;
    etl::fast_matrix<T, 4, 11, 11> c_2;

    c_1(0) = conv_2d_full(I, K(0));
    c_1(1) = conv_2d_full(I, K(1));
    c_1(2) = conv_2d_full(I, K(2));
    c_1(3) = conv_2d_full(I, K(3));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

// conv_2d_same_multi

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I    = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 3, 3> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 1.0, 2.0};

    etl::fast_matrix<T, 2, 3, 3> c_1;
    etl::fast_matrix<T, 2, 3, 3> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 2, 2> I    = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/3", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 2, 2> I    = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<T, 3, 2, 2> c_1;
    etl::fast_matrix<T, 3, 2, 2> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/4", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I(etl::magic<T>(5));
    etl::fast_matrix<T, 3, 5, 5> K;
    K(0) = 2.0 * etl::magic<T>(5);
    K(1) = 3.0 * etl::magic<T>(5);
    K(2) = 4.0 * etl::magic<T>(5);

    etl::fast_matrix<T, 3, 5, 5> c_1;
    etl::fast_matrix<T, 3, 5, 5> c_2;
    etl::fast_matrix<T, 3, 5, 5> c_3;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/5", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I(etl::magic<T>(3));
    etl::fast_matrix<T, 3, 3, 3> K;

    K(0) = etl::magic<T>(3);
    K(1) = etl::magic<T>(3) * 2.0;
    K(2) = etl::magic<T>(3) * etl::magic<T>(3);

    etl::fast_matrix<T, 3, 3, 3> c_1;
    etl::fast_matrix<T, 3, 3, 3> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/6", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 7, 7> I(etl::magic<T>(7));
    etl::fast_matrix<T, 3, 5, 3> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 7> c_1;
    etl::fast_matrix<T, 3, 7, 7> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/7", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 5, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/8", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;
    etl::fast_matrix<T, 3, 9, 7> c_3;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_FLIPPED_TEST_CASE("conv_2d/same/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 3, 3, 5> K;

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_FLIPPED_TEST_CASE("conv_2d/same/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I(0.5 * etl::sequence_generator(42.0));
    etl::fast_matrix<T, 4, 3, 5> K;

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 9, 7> c_1;
    etl::fast_matrix<T, 4, 9, 7> c_2;

    c_1(0) = conv_2d_same(I, K(0));
    c_1(1) = conv_2d_same(I, K(1));
    c_1(2) = conv_2d_same(I, K(2));
    c_1(3) = conv_2d_same(I, K(3));

    K.deep_fflip_inplace();

    Impl::apply(I, K, c_2);

    for (std::size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_MULTI_TEST_CASE("conv_2d/valid/multi_multi/1", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 5, 7, 7> I;
    etl::fast_matrix<T, 3, 5, 3> K;

    etl::fast_matrix<T, 3, 5, 3, 5> C;
    etl::fast_matrix<T, 3, 5, 3, 5> C_ref;

    I = 0.5 * etl::sequence_generator(1.0);
    K = 0.123 * etl::sequence_generator(1.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}

CONV2_VALID_MULTI_MULTI_TEST_CASE("conv_2d/valid/multi_multi/2", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 6, 9, 7> I;
    etl::fast_matrix<T, 9, 5, 2> K;

    etl::fast_matrix<T, 9, 6, 5, 6> C;
    etl::fast_matrix<T, 9, 6, 5, 6> C_ref;

    I = -0.5 * etl::sequence_generator(10.0);
    K = 0.123 * etl::sequence_generator(2.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}

CONV2_VALID_MULTI_MULTI_TEST_CASE("conv_2d/valid/multi_multi/3", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 7, 10, 7> I;
    etl::fast_matrix<T, 8, 5, 2> K;

    etl::fast_matrix<T, 8, 7, 6, 6> C;
    etl::fast_matrix<T, 8, 7, 6, 6> C_ref;

    I = -0.66 * etl::sequence_generator(3.0);
    K = 0.23 * etl::sequence_generator(2.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}

CONV2_VALID_MULTI_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_multi_flipped/1", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 5, 7, 7> I;
    etl::fast_matrix<T, 3, 5, 3> K;

    etl::fast_matrix<T, 3, 5, 3, 5> C;
    etl::fast_matrix<T, 3, 5, 3, 5> C_ref;

    I = 0.5 * etl::sequence_generator(1.0);
    K = 0.123 * etl::sequence_generator(1.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid_flipped(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}

CONV2_VALID_MULTI_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_multi_flipped/2", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 6, 9, 7> I;
    etl::fast_matrix<T, 9, 5, 2> K;

    etl::fast_matrix<T, 9, 6, 5, 6> C;
    etl::fast_matrix<T, 9, 6, 5, 6> C_ref;

    I = -0.5 * etl::sequence_generator(10.0);
    K = 0.123 * etl::sequence_generator(2.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid_flipped(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}

CONV2_VALID_MULTI_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_multi_flipped/3", "[conv][conv2][conv_multi_multi]") {
    etl::fast_matrix<T, 7, 10, 7> I;
    etl::fast_matrix<T, 8, 5, 2> K;

    etl::fast_matrix<T, 8, 7, 6, 6> C;
    etl::fast_matrix<T, 8, 7, 6, 6> C_ref;

    I = -0.66 * etl::sequence_generator(3.0);
    K = 0.23 * etl::sequence_generator(2.0);

    for (size_t k = 0; k < etl::dim<0>(K); ++k) {
        for (size_t i = 0; i < etl::dim<0>(I); ++i) {
            C_ref(k)(i) = conv_2d_valid_flipped(I(i), K(k));
        }
    }

    Impl::apply(I, K, C);

    for (std::size_t i = 0; i < etl::size(C_ref); ++i) {
        REQUIRE_EQUALS_APPROX(C[i], C_ref[i]);
    }
}
