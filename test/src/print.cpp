//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// to_octave

TEST_CASE( "to_octave/fast_vector", "to_octave" ) {
    etl::fast_vector<double, 3> test_vector({1.0, -2.0, 3.0});

    REQUIRE(to_octave(test_vector) == "[1.000000,-2.000000,3.000000]");
}

TEST_CASE( "to_octave/fast_matrix", "to_octave" ) {
    etl::fast_matrix<double, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(to_octave(test_matrix) == "[1.000000,-2.000000;3.000000,0.500000;0.000000,-1.000000]");
}

TEST_CASE( "to_octave/dyn_vector", "to_octave" ) {
    etl::dyn_vector<double> test_vector({1.0, -2.0, 3.0});

    REQUIRE(to_octave(test_vector) == "[1.000000,-2.000000,3.000000]");
}

TEST_CASE( "to_octave/dyn_matrix", "to_octave" ) {
    etl::dyn_matrix<double> test_matrix(3,2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    REQUIRE(to_octave(test_matrix) == "[1.000000,-2.000000;3.000000,0.500000;0.000000,-1.000000]");
}

// to_string

TEST_CASE( "to_string/fast_vector", "to_string" ) {
    etl::fast_vector<double, 3> test_vector({1.0, -2.0, 3.0});

    REQUIRE(to_string(test_vector) == "[1.000000,-2.000000,3.000000]");
}

TEST_CASE( "to_string/fast_matrix", "to_string" ) {
    etl::fast_matrix<double, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(to_string(test_matrix) == "[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]");
}

TEST_CASE( "to_string/fast_matrix_3d", "to_string" ) {
    etl::fast_matrix<double, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(to_string(test_matrix) == "[[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]\n[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]]");
}

TEST_CASE( "to_string/dyn_vector", "to_string" ) {
    etl::dyn_vector<double> test_vector({1.0, -2.0, 3.0});

    REQUIRE(to_string(test_vector) == "[1.000000,-2.000000,3.000000]");
}

TEST_CASE( "to_string/dyn_matrix", "to_string" ) {
    etl::dyn_matrix<double> test_matrix(3,2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    REQUIRE(to_string(test_matrix) == "[[1.000000,-2.000000]\n[3.000000,0.500000]\n[0.000000,-1.000000]]");
}
