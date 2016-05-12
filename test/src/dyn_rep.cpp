//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// Tests for dyn_rep_r

TEMPLATE_TEST_CASE_2("dyn_rep/fast_matrix_1", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 3> b(etl::rep(a, 3));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(0, 2) == 1.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == -2.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == 3.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/fast_matrix_2", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 3> b;

    b = etl::rep(a, 3);

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(0, 2) == 1.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == -2.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == 3.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/fast_matrix_3", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 3, 2> b;

    b = etl::rep(a, 3, 2);

    REQUIRE(b(0, 0, 0) == 1.0);
    REQUIRE(b(0, 0, 1) == 1.0);
    REQUIRE(b(0, 1, 0) == 1.0);
    REQUIRE(b(0, 1, 1) == 1.0);
    REQUIRE(b(0, 2, 0) == 1.0);
    REQUIRE(b(0, 2, 1) == 1.0);
    REQUIRE(b(1, 0, 0) == -2.0);
    REQUIRE(b(1, 0, 1) == -2.0);
    REQUIRE(b(1, 1, 0) == -2.0);
    REQUIRE(b(1, 1, 1) == -2.0);
    REQUIRE(b(1, 2, 0) == -2.0);
    REQUIRE(b(1, 2, 1) == -2.0);
    REQUIRE(b(2, 0, 0) == 3.0);
    REQUIRE(b(2, 0, 1) == 3.0);
    REQUIRE(b(2, 1, 0) == 3.0);
    REQUIRE(b(2, 1, 1) == 3.0);
    REQUIRE(b(2, 2, 0) == 3.0);
    REQUIRE(b(2, 2, 1) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/fast_matrix_4", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 1> a({1.0});
    etl::fast_matrix<Z, 1, 3, 2, 5, 7> b;

    b = etl::rep(a, 3, 2, 5, 7);

    for (auto v : b) {
        REQUIRE(v == 1.0);
    }
}

TEMPLATE_TEST_CASE_2("dyn_rep/dyn_matrix_1", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z> b(etl::rep(a, 3));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(0, 2) == 1.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == -2.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == 3.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/dyn_matrix_2", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z> b(3, 3);

    b = etl::rep(a, 3);

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(0, 2) == 1.0);
    REQUIRE(b(1, 0) == -2.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == -2.0);
    REQUIRE(b(2, 0) == 3.0);
    REQUIRE(b(2, 1) == 3.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/dyn_matrix_3", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z, 3> b(3, 3, 2);

    b = etl::rep(a, 3, 2);

    REQUIRE(b(0, 0, 0) == 1.0);
    REQUIRE(b(0, 0, 1) == 1.0);
    REQUIRE(b(0, 1, 0) == 1.0);
    REQUIRE(b(0, 1, 1) == 1.0);
    REQUIRE(b(0, 2, 0) == 1.0);
    REQUIRE(b(0, 2, 1) == 1.0);
    REQUIRE(b(1, 0, 0) == -2.0);
    REQUIRE(b(1, 0, 1) == -2.0);
    REQUIRE(b(1, 1, 0) == -2.0);
    REQUIRE(b(1, 1, 1) == -2.0);
    REQUIRE(b(1, 2, 0) == -2.0);
    REQUIRE(b(1, 2, 1) == -2.0);
    REQUIRE(b(2, 0, 0) == 3.0);
    REQUIRE(b(2, 0, 1) == 3.0);
    REQUIRE(b(2, 1, 0) == 3.0);
    REQUIRE(b(2, 1, 1) == 3.0);
    REQUIRE(b(2, 2, 0) == 3.0);
    REQUIRE(b(2, 2, 1) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep/dyn_matrix_4", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(1, 1.0);
    etl::dyn_matrix<Z, 5> b(1, 3, 2, 5, 7);

    b = etl::rep(a, 3, 2, 5, 7);

    for (auto v : b) {
        REQUIRE(v == 1.0);
    }
}

TEMPLATE_TEST_CASE_2("dyn_rep/dyn_matrix_5", "dyn_rep", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3);
    a = 1.0;
    a(1, 2) = 3.0;

    etl::dyn_matrix<Z, 6> b(etl::rep(a, 3, 2, 5, 7));

    REQUIRE(b.dim(0) == 2);
    REQUIRE(b.dim(1) == 3);
    REQUIRE(b.dim(2) == 3);
    REQUIRE(b.dim(3) == 2);
    REQUIRE(b.dim(4) == 5);
    REQUIRE(b.dim(5) == 7);

    REQUIRE(b(1, 2, 0, 0, 0, 0) == 3.0);
    REQUIRE(b(0, 2, 0, 0, 0, 0) == 1.0);
    REQUIRE(b(1, 2, 0, 1, 0, 0) == 3.0);
    REQUIRE(b(0, 2, 0, 0, 1, 0) == 1.0);
    REQUIRE(b(1, 2, 1, 0, 0, 0) == 3.0);
    REQUIRE(b(0, 2, 1, 0, 0, 0) == 1.0);
    REQUIRE(b(1, 2, 1, 1, 0, 0) == 3.0);
    REQUIRE(b(0, 2, 1, 0, 1, 0) == 1.0);
    REQUIRE(b(1, 2, 1, 0, 0, 6) == 3.0);
    REQUIRE(b(0, 2, 1, 0, 0, 6) == 1.0);
    REQUIRE(b(1, 2, 1, 1, 0, 6) == 3.0);
    REQUIRE(b(0, 2, 1, 0, 1, 6) == 1.0);
}

// Tests for dyn_rep_l

TEMPLATE_TEST_CASE_2("dyn_rep_l/fast_matrix_1", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 3> b(etl::rep_l(a, 3));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == -2.0);
    REQUIRE(b(0, 2) == 3.0);
    REQUIRE(b(1, 0) == 1.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == 3.0);
    REQUIRE(b(2, 0) == 1.0);
    REQUIRE(b(2, 1) == -2.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/fast_matrix_2", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 3> b;

    b = etl::rep_l(a, 3);

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == -2.0);
    REQUIRE(b(0, 2) == 3.0);
    REQUIRE(b(1, 0) == 1.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == 3.0);
    REQUIRE(b(2, 0) == 1.0);
    REQUIRE(b(2, 1) == -2.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/fast_matrix_3", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<Z, 3, 2, 3> b;

    b = etl::rep_l(a, 3, 2);

    REQUIRE(b(0, 0, 0) == 1.0);
    REQUIRE(b(0, 0, 1) == -2.0);
    REQUIRE(b(0, 0, 2) == 3.0);
    REQUIRE(b(0, 1, 0) == 1.0);
    REQUIRE(b(0, 1, 1) == -2.0);
    REQUIRE(b(0, 1, 2) == 3.0);
    REQUIRE(b(1, 0, 0) == 1.0);
    REQUIRE(b(1, 0, 1) == -2.0);
    REQUIRE(b(1, 0, 2) == 3.0);
    REQUIRE(b(1, 1, 0) == 1.0);
    REQUIRE(b(1, 1, 1) == -2.0);
    REQUIRE(b(1, 1, 2) == 3.0);
    REQUIRE(b(2, 0, 0) == 1.0);
    REQUIRE(b(2, 0, 1) == -2.0);
    REQUIRE(b(2, 0, 2) == 3.0);
    REQUIRE(b(2, 1, 0) == 1.0);
    REQUIRE(b(2, 1, 1) == -2.0);
    REQUIRE(b(2, 1, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/fast_matrix_4", "dyn_rep", Z, float, double) {
    etl::fast_matrix<Z, 1> a({1.0});
    etl::fast_matrix<Z, 3, 2, 5, 7, 1> b;

    b = etl::rep_l(a, 3, 2, 5, 7);

    for (auto v : b) {
        REQUIRE(v == 1.0);
    }
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/dyn_matrix_1", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z> b(etl::rep_l(a, 3));

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == -2.0);
    REQUIRE(b(0, 2) == 3.0);
    REQUIRE(b(1, 0) == 1.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == 3.0);
    REQUIRE(b(2, 0) == 1.0);
    REQUIRE(b(2, 1) == -2.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/dyn_matrix_2", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z> b(3, 3);

    b = etl::rep_l(a, 3);

    REQUIRE(b(0, 0) == 1.0);
    REQUIRE(b(0, 1) == -2.0);
    REQUIRE(b(0, 2) == 3.0);
    REQUIRE(b(1, 0) == 1.0);
    REQUIRE(b(1, 1) == -2.0);
    REQUIRE(b(1, 2) == 3.0);
    REQUIRE(b(2, 0) == 1.0);
    REQUIRE(b(2, 1) == -2.0);
    REQUIRE(b(2, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/dyn_matrix_3", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<Z, 3> b(3, 2, 3);

    b = etl::rep_l(a, 3, 2);

    REQUIRE(b(0, 0, 0) == 1.0);
    REQUIRE(b(0, 0, 1) == -2.0);
    REQUIRE(b(0, 0, 2) == 3.0);
    REQUIRE(b(0, 1, 0) == 1.0);
    REQUIRE(b(0, 1, 1) == -2.0);
    REQUIRE(b(0, 1, 2) == 3.0);
    REQUIRE(b(1, 0, 0) == 1.0);
    REQUIRE(b(1, 0, 1) == -2.0);
    REQUIRE(b(1, 0, 2) == 3.0);
    REQUIRE(b(1, 1, 0) == 1.0);
    REQUIRE(b(1, 1, 1) == -2.0);
    REQUIRE(b(1, 1, 2) == 3.0);
    REQUIRE(b(2, 0, 0) == 1.0);
    REQUIRE(b(2, 0, 1) == -2.0);
    REQUIRE(b(2, 0, 2) == 3.0);
    REQUIRE(b(2, 1, 0) == 1.0);
    REQUIRE(b(2, 1, 1) == -2.0);
    REQUIRE(b(2, 1, 2) == 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/dyn_matrix_4", "dyn_rep", Z, float, double) {
    etl::dyn_vector<Z> a(1, 1.0);
    etl::dyn_matrix<Z, 5> b(etl::rep_l(a, 3, 2, 5, 7));

    for (auto v : b) {
        REQUIRE(v == 1.0);
    }
}

TEMPLATE_TEST_CASE_2("dyn_rep_l/dyn_matrix_5", "dyn_rep", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2);
    a = 2.0;

    a(1, 1) = 1.0;

    etl::dyn_matrix<Z, 6> b(etl::rep_l(a, 1, 2, 3, 4));

    REQUIRE(b.dim(0) == 1);
    REQUIRE(b.dim(1) == 2);
    REQUIRE(b.dim(2) == 3);
    REQUIRE(b.dim(3) == 4);
    REQUIRE(b.dim(4) == 2);
    REQUIRE(b.dim(5) == 2);

    REQUIRE(b(0, 0, 0, 0, 1, 0) == 2.0);
    REQUIRE(b(0, 0, 0, 0, 1, 1) == 1.0);
    REQUIRE(b(0, 1, 0, 0, 1, 0) == 2.0);
    REQUIRE(b(0, 1, 0, 0, 1, 0) == 2.0);
    REQUIRE(b(0, 0, 1, 0, 1, 1) == 1.0);
    REQUIRE(b(0, 0, 1, 0, 1, 1) == 1.0);
}

//TODO Add tests for dyn_rep_l and dyn_rep_r on matrices
