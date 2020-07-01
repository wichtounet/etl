//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("transpose_front/0", "[transpose_front]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 5> a;
    etl::fast_matrix<Z, 2, 3, 5> b;
    etl::fast_matrix<Z, 2, 3, 5> ref;

    a = etl::sequence_generator<Z>(5) * 0.2;

    b = transpose_front(a);

    for (size_t i = 0; i < etl::dim<0>(a); ++i) {
        for (size_t j = 0; j < etl::dim<1>(a); ++j) {
            ref(j)(i) = a(i)(j);
        }
    }

    for (size_t i = 0; i < etl::size(a); ++i) {
        REQUIRE(ref[i] == b[i]);
    }
}

TEMPLATE_TEST_CASE_2("transpose_front/1", "[transpose_front]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(3, 2, 5);
    etl::dyn_matrix<Z, 3> b(2, 3, 5);
    etl::dyn_matrix<Z, 3> ref(2, 3, 5);

    a = etl::sequence_generator<Z>(3) * 0.1;

    b = transpose_front(a);

    for (size_t i = 0; i < etl::dim<0>(a); ++i) {
        for (size_t j = 0; j < etl::dim<1>(a); ++j) {
            ref(j)(i) = a(i)(j);
        }
    }

    for (size_t i = 0; i < etl::size(a); ++i) {
        REQUIRE(ref[i] == b[i]);
    }
}

TEMPLATE_TEST_CASE_2("transpose_front/2", "[transpose_front]", Z, float, double) {
    etl::fast_matrix<Z, 7, 5, 15> a;
    etl::fast_matrix<Z, 5, 7, 15> b;
    etl::fast_matrix<Z, 5, 7, 15> ref;

    a = etl::sequence_generator<Z>(5) * 0.2;

    b = transpose_front(a);

    for (size_t i = 0; i < etl::dim<0>(a); ++i) {
        for (size_t j = 0; j < etl::dim<1>(a); ++j) {
            ref(j)(i) = a(i)(j);
        }
    }

    for (size_t i = 0; i < etl::size(a); ++i) {
        REQUIRE(ref[i] == b[i]);
    }
}

TEMPLATE_TEST_CASE_2("transpose_front/3", "[transpose_front]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(7, 5, 15);
    etl::dyn_matrix<Z, 3> b(5, 7, 15);
    etl::dyn_matrix<Z, 3> ref(5, 7, 15);

    a = etl::sequence_generator<Z>(3) * 0.1;

    b = transpose_front(a);

    for (size_t i = 0; i < etl::dim<0>(a); ++i) {
        for (size_t j = 0; j < etl::dim<1>(a); ++j) {
            ref(j)(i) = a(i)(j);
        }
    }

    for (size_t i = 0; i < etl::size(a); ++i) {
        REQUIRE(ref[i] == b[i]);
    }
}
