//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

// conv_2d_valid_multi

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/3", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<T, 3, 2, 2> c_1;
    etl::fast_matrix<T, 3, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/4", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I;
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    I = etl::magic<T>(5);

    etl::fast_matrix<T, 3, 4, 4> c_1;
    etl::fast_matrix<T, 3, 4, 4> c_2;
    etl::fast_matrix<T, 3, 4, 4> c_3;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/5", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I;
    etl::fast_matrix<T, 3, 3, 3> K;

    I = etl::magic<T>(5);

    K(0) = etl::magic<T>(3);
    K(1) = etl::magic<T>(3) * 2.0;
    K(2) = etl::magic<T>(3) * etl::magic<T>(3);

    etl::fast_matrix<T, 3, 3, 3> c_1;
    etl::fast_matrix<T, 3, 3, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/6", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 7, 7> I;
    etl::fast_matrix<T, 3, 5, 3> K;

    I = etl::magic<T>(7);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 3, 5> c_1;
    etl::fast_matrix<T, 3, 3, 5> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/7", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 5, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 5, 3> c_1;
    etl::fast_matrix<T, 3, 5, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_TEST_CASE("conv_2d/valid/multi/8", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 3> c_1;
    etl::fast_matrix<T, 3, 7, 3> c_2;
    etl::fast_matrix<T, 3, 7, 3> c_3;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
        c_1(2) = conv_2d_valid(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 3> c_1;
    etl::fast_matrix<T, 3, 7, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid_flipped(I, K(0));
        c_1(1) = conv_2d_valid_flipped(I, K(1));
        c_1(2) = conv_2d_valid_flipped(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_VALID_MULTI_FLIPPED_TEST_CASE("conv_2d/valid/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 4, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 7, 3> c_1;
    etl::fast_matrix<T, 4, 7, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid_flipped(I, K(0));
        c_1(1) = conv_2d_valid_flipped(I, K(1));
        c_1(2) = conv_2d_valid_flipped(I, K(2));
        c_1(3) = conv_2d_valid_flipped(I, K(3));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

//conv_2d_full_multi

CONV2_FULL_MULTI_TEST_CASE("conv_2d/full/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 5, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 13, 11> c_1;
    etl::fast_matrix<T, 3, 13, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_full(I, K(0));
        c_1(1) = conv_2d_full(I, K(1));
        c_1(2) = conv_2d_full(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_TEST_CASE("conv_2d/full/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 11, 11> c_1;
    etl::fast_matrix<T, 3, 11, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_full(I, K(0));
        c_1(1) = conv_2d_full(I, K(1));
        c_1(2) = conv_2d_full(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_FLIPPED_TEST_CASE("conv_2d/full/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 11, 11> c_1;
    etl::fast_matrix<T, 3, 11, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_full_flipped(I, K(0));
        c_1(1) = conv_2d_full_flipped(I, K(1));
        c_1(2) = conv_2d_full_flipped(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_FULL_MULTI_FLIPPED_TEST_CASE("conv_2d/full/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 4, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 11, 11> ref;
    etl::fast_matrix<T, 4, 11, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        ref(0) = conv_2d_full_flipped(I, K(0));
        ref(1) = conv_2d_full_flipped(I, K(1));
        ref(2) = conv_2d_full_flipped(I, K(2));
        ref(3) = conv_2d_full_flipped(I, K(3));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(ref); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], ref[i]);
    }
}

// conv_2d_same_multi

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 3, 3> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5, 1.0, 2.0};

    etl::fast_matrix<T, 2, 3, 3> c_1;
    etl::fast_matrix<T, 2, 3, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 2, 2> I = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<T, 2, 2, 2> c_1;
    etl::fast_matrix<T, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/3", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 2, 2> I = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<T, 3, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1, 0.0, -1.0, 1.0, 1.0};

    etl::fast_matrix<T, 3, 2, 2> c_1;
    etl::fast_matrix<T, 3, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/4", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 5, 5> I;
    etl::fast_matrix<T, 3, 5, 5> K;

    I = etl::magic<T>(5);

    K(0) = 2.0 * etl::magic<T>(5);
    K(1) = 3.0 * etl::magic<T>(5);
    K(2) = 4.0 * etl::magic<T>(5);

    etl::fast_matrix<T, 3, 5, 5> c_1;
    etl::fast_matrix<T, 3, 5, 5> c_2;
    etl::fast_matrix<T, 3, 5, 5> c_3;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/5", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 3, 3> I;
    etl::fast_matrix<T, 3, 3, 3> K;

    I = etl::magic<T>(3);

    K(0) = etl::magic<T>(3);
    K(1) = etl::magic<T>(3) * 2.0;
    K(2) = etl::magic<T>(3) * etl::magic<T>(3);

    etl::fast_matrix<T, 3, 3, 3> c_1;
    etl::fast_matrix<T, 3, 3, 3> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/6", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 7, 7> I;
    etl::fast_matrix<T, 3, 5, 3> K;

    I = etl::magic<T>(7);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 7, 7> c_1;
    etl::fast_matrix<T, 3, 7, 7> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/7", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 5, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_TEST_CASE("conv_2d/same/multi/8", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;
    etl::fast_matrix<T, 3, 9, 7> c_3;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
        c_1(2) = conv_2d_same(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_FLIPPED_TEST_CASE("conv_2d/same/multi_flipped/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 3, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -1.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<T, 3, 9, 7> c_1;
    etl::fast_matrix<T, 3, 9, 7> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same_flipped(I, K(0));
        c_1(1) = conv_2d_same_flipped(I, K(1));
        c_1(2) = conv_2d_same_flipped(I, K(2));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

CONV2_SAME_MULTI_FLIPPED_TEST_CASE("conv_2d/same/multi_flipped/2", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<T, 9, 7> I;
    etl::fast_matrix<T, 4, 3, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 0.5 * etl::sequence_generator(100.0);
    K(1) = -1.3 * etl::sequence_generator(40.0);
    K(2) = 2.5 * etl::sequence_generator(133.0);
    K(3) = 666.666 * etl::sequence_generator(121.0);

    etl::fast_matrix<T, 4, 9, 7> c_1;
    etl::fast_matrix<T, 4, 9, 7> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same_flipped(I, K(0));
        c_1(1) = conv_2d_same_flipped(I, K(1));
        c_1(2) = conv_2d_same_flipped(I, K(2));
        c_1(3) = conv_2d_same_flipped(I, K(3));
    }

    Impl::apply(I, K, c_2);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

/*  Mixed Tests */

ETL_TEST_CASE("conv2/valid/multi/mixed/0", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<double, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 2.0, 0.0, 0.5, 0.5};

    etl::fast_matrix<float, 2, 2, 2> c_1;
    etl::fast_matrix<float, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
    }

    c_2 = etl::conv_2d_valid_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

ETL_TEST_CASE("conv2/valid/multi/mixed/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 3, 3> I = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix_cm<float, 2, 2, 2> K = {2.0, 0.5, 0.0, 0.5, 2.0, 0.5, 0.0, 0.5};

    etl::fast_matrix<float, 2, 2, 2> c_1;
    etl::fast_matrix<float, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_valid(I, K(0));
        c_1(1) = conv_2d_valid(I, K(1));
    }

    c_2 = etl::conv_2d_valid_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

ETL_TEST_CASE("conv2/same/multi/mixed/0", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 2, 2> I = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<double, 2, 2, 2> K = {2.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.7, 0.1};

    etl::fast_matrix<float, 2, 2, 2> c_1;
    etl::fast_matrix<float, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
    }

    c_2 = etl::conv_2d_same_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

ETL_TEST_CASE("conv2/same/multi/mixed/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 2, 2> I = {1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix_cm<float, 2, 2, 2> K = {2.0, 0.5, 0.0, 0.5, 1.0, 0.7, 0.5, 0.1};

    etl::fast_matrix<float, 2, 2, 2> c_1;
    etl::fast_matrix<float, 2, 2, 2> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_same(I, K(0));
        c_1(1) = conv_2d_same(I, K(1));
    }

    c_2 = etl::conv_2d_same_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

ETL_TEST_CASE("conv2/full/multi/mixed/0", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 9, 7> I;
    etl::fast_matrix<double, 3, 5, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<float, 3, 13, 11> c_1;
    etl::fast_matrix<float, 3, 13, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_full(I, K(0));
        c_1(1) = conv_2d_full(I, K(1));
        c_1(2) = conv_2d_full(I, K(2));
    }

    c_2 = etl::conv_2d_full_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}

ETL_TEST_CASE("conv2/full/multi/mixed/1", "[conv][conv2][conv_multi]") {
    etl::fast_matrix<float, 9, 7> I;
    etl::fast_matrix_cm<float, 3, 5, 5> K;

    I = 0.5 * etl::sequence_generator(42.0);

    K(0) = 1.5 * etl::sequence_generator(10.0);
    K(1) = -2.5 * etl::sequence_generator(5.0);
    K(2) = 1.3 * etl::sequence_generator(12.0);

    etl::fast_matrix<float, 3, 13, 11> c_1;
    etl::fast_matrix<float, 3, 13, 11> c_2;

    SELECTED_SECTION(etl::conv_impl::STD) {
        c_1(0) = conv_2d_full(I, K(0));
        c_1(1) = conv_2d_full(I, K(1));
        c_1(2) = conv_2d_full(I, K(2));
    }

    c_2 = etl::conv_2d_full_multi(I, K);

    for (size_t i = 0; i < etl::size(c_1); ++i) {
        REQUIRE_EQUALS_APPROX(c_2[i], c_1[i]);
    }
}
