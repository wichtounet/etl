#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "etl/fast_vector.hpp"

TEST_CASE( "fast_vector/init_1", "fast_vector::fast_vector(T)" ) {
    etl::fast_vector<double, 4> test_vector(3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
    }
}

TEST_CASE( "fast_vector/init_2", "fast_vector::operator=(T)" ) {
    etl::fast_vector<double, 4> test_vector;

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
    }
}

TEST_CASE( "fast_vector/mul_scalar", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector;

    test_vector[0] = -1.0;
    test_vector[1] = 2.0;
    test_vector[2] = 5.0;

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}