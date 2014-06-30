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

TEST_CASE( "fast_vector/init_3", "fast_vector::fast_vector(initializer_list)" ) {
    etl::fast_vector<double, 3> test_vector = {1.0, 2.0, 3.0};

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

TEST_CASE( "fast_vector/add_scalar", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a + b;

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "fast_vector/sub_scalar", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "fast_vector/sub", "fast_vector::operator-" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a - b;

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "fast_vector/mul_scalar", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a * b;

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "fast_vector/div_scalar", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] ==  2.0 / 2.5);
    REQUIRE(test_vector[2] ==  5.0 / 2.5);
}

TEST_CASE( "fast_vector/mul", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c = a * b;

    REQUIRE(c[0] == 1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "fast_vector/mod_scalar", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod", "fast_vector::operator*" ) {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    etl::fast_vector<int, 3> c = a % b;

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}