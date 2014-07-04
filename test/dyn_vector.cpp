#include "catch.hpp"

#include "etl/dyn_vector.hpp"

//{{{ Init tests

TEST_CASE( "dyn_vector/init_1", "dyn_vector::dyn_vector(T)" ) {
    etl::dyn_vector<double> test_vector(4, 3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
        REQUIRE(test_vector(i) == 3.3);
    }
}

TEST_CASE( "dyn_vector/init_2", "dyn_vector::operator=(T)" ) {
    etl::dyn_vector<double> test_vector(4);

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == 3.3);
        REQUIRE(test_vector(i) == 3.3);
    }
}

TEST_CASE( "dyn_vector/init_3", "dyn_vector::dyn_vector(initializer_list)" ) {
    etl::dyn_vector<double> test_vector = {1.0, 2.0, 3.0};

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

//}}} Init tests