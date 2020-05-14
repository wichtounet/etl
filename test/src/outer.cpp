//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// outer product

TEMPLATE_TEST_CASE_2("fast_vector/outer_1", "sum", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<Z, 3> b = {4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 4.0);
    REQUIRE_EQUALS(c(0, 1), 5.0);
    REQUIRE_EQUALS(c(0, 2), 6.0);

    REQUIRE_EQUALS(c(1, 0), 8.0);
    REQUIRE_EQUALS(c(1, 1), 10.0);
    REQUIRE_EQUALS(c(1, 2), 12.0);

    REQUIRE_EQUALS(c(2, 0), 12.0);
    REQUIRE_EQUALS(c(2, 1), 15.0);
    REQUIRE_EQUALS(c(2, 2), 18.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/outer_2", "sum", Z, float, double) {
    etl::fast_vector<Z, 2> a = {1.0, 2.0};
    etl::fast_vector<Z, 4> b = {2.0, 3.0, 4.0, 5.0};
    etl::fast_matrix<Z, 2, 4> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 2);
    REQUIRE_EQUALS(c(0, 1), 3);
    REQUIRE_EQUALS(c(0, 2), 4);
    REQUIRE_EQUALS(c(0, 3), 5);

    REQUIRE_EQUALS(c(1, 0), 4);
    REQUIRE_EQUALS(c(1, 1), 6);
    REQUIRE_EQUALS(c(1, 2), 8);
    REQUIRE_EQUALS(c(1, 3), 10);
}

TEMPLATE_TEST_CASE_2("fast_vector/outer_3", "sum", Z, float, double) {
    etl::fast_vector<Z, 2> a = {1.0, 2.0};
    etl::dyn_vector<Z> b(4, etl::values(2.0, 3.0, 4.0, 5.0));
    etl::fast_matrix<Z, 2, 4> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 2);
    REQUIRE_EQUALS(c(0, 1), 3);
    REQUIRE_EQUALS(c(0, 2), 4);
    REQUIRE_EQUALS(c(0, 3), 5);

    REQUIRE_EQUALS(c(1, 0), 4);
    REQUIRE_EQUALS(c(1, 1), 6);
    REQUIRE_EQUALS(c(1, 2), 8);
    REQUIRE_EQUALS(c(1, 3), 10);
}

// batch outer product

TEMPLATE_TEST_CASE_2("batch_outer/1", "[outer]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 3> b = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    etl::fast_matrix<Z, 3, 3> c;
    etl::fast_matrix<Z, 3, 3> c_ref;

    c = batch_outer(a, b);

    c_ref = 0;

    for (size_t bb = 0; bb < 2; ++bb) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_outer/2", "[outer]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(32, 31);
    etl::dyn_matrix<Z, 2> b(32, 23);

    a = Z(0.01) * etl::sequence_generator<Z>(1.0);
    b = Z(-0.032) * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 2> c(31, 23);
    etl::dyn_matrix<Z, 2> c_ref(31, 23);

    c = batch_outer(a, b);

    c_ref = 0;

    for (size_t bb = 0; bb < 32; ++bb) {
        for (size_t i = 0; i < 31; ++i) {
            for (size_t j = 0; j < 23; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_outer/3", "[outer]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(32, 24);
    etl::dyn_matrix<Z, 2> b(32, 33);

    a = Z(0.01) * etl::sequence_generator<Z>(1.0);
    b = Z(-0.032) * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 2> c(24, 33);
    etl::dyn_matrix<Z, 2> c_ref(24, 33);

    c = batch_outer(a, b);

    c_ref = 0;

    for (size_t bb = 0; bb < 32; ++bb) {
        for (size_t i = 0; i < 24; ++i) {
            for (size_t j = 0; j < 33; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_outer/4", "[outer]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 3> b = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    etl::fast_matrix<Z, 3, 3> c;
    etl::fast_matrix<Z, 3, 3> c_ref;

    c = 1.0;
    c -= batch_outer(a, b);

    c_ref = 0;

    for (size_t bb = 0; bb < 2; ++bb) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], 1.0 - c_ref[i]);
    }
}
