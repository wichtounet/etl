//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

/// sequence_generator

TEMPLATE_TEST_CASE_2("sequence/fast_vector_1", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::sequence_generator();

    REQUIRE(b[0] == 0.0);
    REQUIRE(b[1] == 1.0);
    REQUIRE(b[2] == 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_vector_2", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::sequence_generator(99);

    REQUIRE(b[0] == 99.0);
    REQUIRE(b[1] == 100.0);
    REQUIRE(b[2] == 101.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_vector_3", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b(etl::sequence_generator(99));

    REQUIRE(b[0] == 99.0);
    REQUIRE(b[1] == 100.0);
    REQUIRE(b[2] == 101.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_1", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = etl::sequence_generator();

    REQUIRE(b(0, 0) == 0.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(1, 0) == 2.0);
    REQUIRE(b(1, 1) == 3.0);
    REQUIRE(b(2, 0) == 4.0);
    REQUIRE(b(2, 1) == 5.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_2", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = 0.1 * etl::sequence_generator();

    REQUIRE(etl::etl_traits<std::decay_t<decltype(0.1 * etl::sequence_generator())>>::is_generator);

    REQUIRE(b(0, 0) == Approx(0.0));
    REQUIRE(b(0, 1) == Approx(0.1));
    REQUIRE(b(1, 0) == Approx(0.2));
    REQUIRE(b(1, 1) == Approx(0.3));
    REQUIRE(b(2, 0) == Approx(0.4));
    REQUIRE(b(2, 1) == Approx(0.5));
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_3", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b(1.0);

    b = 0.1 * etl::sequence_generator() + b;

    REQUIRE(b(0, 0) == Approx(1.0));
    REQUIRE(b(0, 1) == Approx(1.1));
    REQUIRE(b(1, 0) == Approx(1.2));
    REQUIRE(b(1, 1) == Approx(1.3));
    REQUIRE(b(2, 0) == Approx(1.4));
    REQUIRE(b(2, 1) == Approx(1.5));
}

TEMPLATE_TEST_CASE_2("sequence/dyn_vector_1", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::sequence_generator();

    REQUIRE(b[0] == 0.0);
    REQUIRE(b[1] == 1.0);
    REQUIRE(b[2] == 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/dyn_vector_2", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3, etl::sequence_generator());

    REQUIRE(b[0] == 0.0);
    REQUIRE(b[1] == 1.0);
    REQUIRE(b[2] == 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/dyn_matrix_1", "generator", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::sequence_generator();

    REQUIRE(b(0, 0) == 0.0);
    REQUIRE(b(0, 1) == 1.0);
    REQUIRE(b(1, 0) == 2.0);
    REQUIRE(b(1, 1) == 3.0);
    REQUIRE(b(2, 0) == 4.0);
    REQUIRE(b(2, 1) == 5.0);
}

/// normal_generator

//Simply ensures that it compiles

TEMPLATE_TEST_CASE_2("normal/fast_vector_1", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::normal_generator();
}

TEMPLATE_TEST_CASE_2("normal/fast_matrix_1", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = etl::normal_generator();
}

TEMPLATE_TEST_CASE_2("normal/dyn_vector_1", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::normal_generator();
}

TEMPLATE_TEST_CASE_2("normal/dyn_matrix_1", "generator", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::normal_generator();
}
