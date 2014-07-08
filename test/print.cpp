#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"

TEST_CASE( "to_octave/fast_vector", "to_octave" ) {
    etl::fast_vector<double, 3> test_vector({1.0, -2.0, 3.0});

    REQUIRE(to_octave(test_vector) == "[1.000000,-2.000000,3.000000]");
}

TEST_CASE( "to_octave/fast_matrix", "to_octave" ) {
    etl::fast_matrix<double, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(to_octave(test_matrix) == "[1.000000,-2.000000,3.000000;0.500000,0.000000,-1.000000]");
}