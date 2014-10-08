//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/dyn_vector.hpp"

///{{{ sequence_generator

TEST_CASE( "sequence/fast_vector_1", "generator" ) {
    etl::fast_vector<double, 3> b;

    b = etl::sequence_generator();

    REQUIRE(b[0] == 0.0);
    REQUIRE(b[1] == 1.0);
    REQUIRE(b[2] == 2.0);
}

TEST_CASE( "sequence/fast_vector_2", "generator" ) {
    etl::fast_vector<double, 3> b;

    b = etl::sequence_generator(99);

    REQUIRE(b[0] == 99.0);
    REQUIRE(b[1] == 100.0);
    REQUIRE(b[2] == 101.0);
}

TEST_CASE( "sequence/fast_matrix_1", "generator" ) {
    etl::fast_matrix<double, 3,2> b;

    b = etl::sequence_generator();

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 2.0);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == 4.0);
    REQUIRE(b(2,1) == 5.0);
}

TEST_CASE( "sequence/fast_matrix_2", "generator" ) {
    etl::fast_matrix<double, 3,2> b;

    b = 0.1 * etl::sequence_generator();

    REQUIRE(b(0,0) == Approx(0.0));
    REQUIRE(b(0,1) == Approx(0.1));
    REQUIRE(b(1,0) == Approx(0.2));
    REQUIRE(b(1,1) == Approx(0.3));
    REQUIRE(b(2,0) == Approx(0.4));
    REQUIRE(b(2,1) == Approx(0.5));
}

TEST_CASE( "sequence/dyn_vector_1", "generator" ) {
    etl::dyn_vector<double> b(3);

    b = etl::sequence_generator();

    REQUIRE(b[0] == 0.0);
    REQUIRE(b[1] == 1.0);
    REQUIRE(b[2] == 2.0);
}

TEST_CASE( "sequence/dyn_matrix_1", "generator" ) {
    etl::dyn_matrix<double> b(3,2);

    b = etl::sequence_generator();

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 2.0);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == 4.0);
    REQUIRE(b(2,1) == 5.0);
}

///}}}

///{{{ normal_generator

TEST_CASE( "normal/fast_vector_1", "generator" ) {
    etl::fast_vector<double, 3> b;

    b = etl::normal_generator();
}

TEST_CASE( "normal/fast_vector_2", "generator" ) {
    etl::fast_vector<double, 3> b;

    b = etl::normal_generator();
}

TEST_CASE( "normal/fast_matrix_1", "generator" ) {
    etl::fast_matrix<double, 3,2> b;

    b = etl::normal_generator();
}

TEST_CASE( "normal/dyn_vector_1", "generator" ) {
    etl::dyn_vector<double> b(3);

    b = etl::normal_generator();
}

TEST_CASE( "normal/dyn_matrix_1", "generator" ) {
    etl::dyn_matrix<double> b(3,2);

    b = etl::normal_generator();
}

///}}}