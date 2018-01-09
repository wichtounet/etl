//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

// p_max_pool_h

TEMPLATE_TEST_CASE_2("p_max_pool_h_1", "p_max_pool_h_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 4, 4> b;
    b = etl::p_max_pool_h<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(2, 3), 0.00089);
}

TEMPLATE_TEST_CASE_2("p_max_pool_h_2", "p_max_pool_h_3d", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 4, 4> b;
    b = etl::p_max_pool_h<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(0, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 2, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(1, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 2, 3), 0.00089);
}

TEMPLATE_TEST_CASE_2("p_max_pool_h_3", "p_max_pool_h_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 4, 4> b;

    b = etl::p_max_pool_h<2, 2>(a + 1.0);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.21789);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.08015);
    REQUIRE_EQUALS_APPROX(b(1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 3), 0.97719);
    REQUIRE_EQUALS_APPROX(b(2, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(2, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(2, 3), 0.00089);
}

TEMPLATE_TEST_CASE_2("p_max_pool_h_4", "p_max_pool_h_3d", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 4, 4> b;

    b = etl::p_max_pool_h<2, 2>(a + 1.0);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0), 0.21789);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1), 0.08015);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 1, 3), 0.97719);
    REQUIRE_EQUALS_APPROX(b(0, 2, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(0, 2, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(0, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 2, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0), 0.21789);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1), 0.08015);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 1, 3), 0.97719);
    REQUIRE_EQUALS_APPROX(b(1, 2, 0), 0.59229);
    REQUIRE_EQUALS_APPROX(b(1, 2, 1), 0.02948);
    REQUIRE_EQUALS_APPROX(b(1, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 2, 3), 0.00089);
}

TEMPLATE_TEST_CASE_2("p_max_pool_h_5", "p_max_pool_h_4d", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 4, 4> a({
        1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5,
        1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5
        });
    etl::fast_matrix<Z, 2, 2, 4, 4> b;
    b = etl::p_max_pool_h<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 0, 2, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(0, 1, 2, 3), 0.00089);

    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 0, 2, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 3), 0.00089);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 1), 0.07045);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 3), 0.97686);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2, 0), 0.52059);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2, 1), 0.02591);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2, 2), 0.01085);
    REQUIRE_EQUALS_APPROX(b(1, 1, 2, 3), 0.00089);
}

// p_max_pool_p

TEMPLATE_TEST_CASE_2("p_max_pool_p_1", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2> b;
    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_2", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2> b;
    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_3", "p_max_pool_p_3d", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2, 2> b;
    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_4", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_5", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_6", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5});
    etl::fast_matrix<Z, 2, 2, 2> b;

    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1), 0.00054);
}

TEMPLATE_TEST_CASE_2("p_max_pool_p_7", "p_max_pool_p_2d", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 4, 4> a({
        1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5,
        1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5, 2.54, 2.0, 1.0, 1.5, 1.0, 1.25, 0.05, 2.5, 1.0, -2.0, 3.0, 0.5, 0.0, -1, 3.0, 7.5
        });
    etl::fast_matrix<Z, 2, 2, 2, 2> b;

    b = etl::p_max_pool_p<2, 2>(a);

    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 0, 0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 0, 1, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(0, 1, 0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(0, 1, 1, 1), 0.00054);

    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 0, 0, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 0, 1, 1), 0.00054);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 0), 0.03666);
    REQUIRE_EQUALS_APPROX(b(1, 1, 0, 1), 0.04665);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 0), 0.19151);
    REQUIRE_EQUALS_APPROX(b(1, 1, 1, 1), 0.00054);
}
