//=======================================================================
// Copyright (c) 2014 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/etl.hpp"

//{{{ Init tests

TEST_CASE( "fast_dyn_matrix/init_1", "fast_dyn_matrix::fast_dyn_matrix(T)" ) {
    etl::fast_dyn_matrix<double, 2, 2> test_matrix(3.3);

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "fast_dyn_matrix/init_2", "fast_dyn_matrix::operator=(T)" ) {
    etl::fast_dyn_matrix<double, 2, 2> test_matrix;

    test_matrix = 3.3;

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "fast_dyn_matrix/init_3", "fast_dyn_matrix::fast_dyn_matrix(initializer_list)" ) {
    etl::fast_dyn_matrix<double, 2, 2> test_matrix = {1.0, 3.0, 5.0, 2.0};

    REQUIRE(test_matrix.size() == 4);

    REQUIRE(test_matrix[0] == 1.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 5.0);
}

//}}} Init tests

TEST_CASE( "fast_dyn_matrix/access", "fast_dyn_matrix::operator()" ) {
    etl::fast_dyn_matrix<double, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(test_matrix(0, 0, 0) == 1.0);
    REQUIRE(test_matrix(0, 0, 1) == -2.0);
    REQUIRE(test_matrix(0, 1, 0) == 3.0);
    REQUIRE(test_matrix(0, 1, 1) == 0.5);
    REQUIRE(test_matrix(0, 2, 0) == 0.0);
    REQUIRE(test_matrix(0, 2, 1) == -1);

    REQUIRE(test_matrix(1, 0, 0) == 1.0);
    REQUIRE(test_matrix(1, 0, 1) == -2.0);
    REQUIRE(test_matrix(1, 1, 0) == 3.0);
    REQUIRE(test_matrix(1, 1, 1) == 0.5);
    REQUIRE(test_matrix(1, 2, 0) == 0.0);
    REQUIRE(test_matrix(1, 2, 1) == -1);
}

//{{{ Binary operators test

TEST_CASE( "fast_dyn_matrix/add_scalar_1", "fast_dyn_matrix::operator+" ) {
    etl::fast_dyn_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 + test_matrix;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "fast_dyn_matrix/add_scalar_2", "fast_dyn_matrix::operator+" ) {
    etl::fast_dyn_matrix<double, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix + 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "fast_dyn_matrix/add_scalar_4", "fast_dyn_matrix::operator+=" ) {
    etl::fast_dyn_matrix<double, 2, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0, 1.0, 1.0, 1.0, 1.0};

    test_matrix += 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
    REQUIRE(test_matrix[7] == 2.0);
}

TEST_CASE( "fast_dyn_matrix/add_1", "fast_dyn_matrix::operator+" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_dyn_matrix<double, 2, 2> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "fast_dyn_matrix/add_2", "fast_dyn_matrix::operator+=" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "fast_dyn_matrix/log_1", "fast_dyn_matrix::log" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_dyn_matrix<double, 2, 2> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "fast_dyn_matrix/log_2", "fast_dyn_matrix::log" ) {
    etl::fast_dyn_matrix<double, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_dyn_matrix<double, 2, 2, 1> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "fast_dyn_matrix/unary_unary", "fast_dyn_matrix::abs" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_dyn_matrix<double, 2, 2> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_dyn_matrix/unary_binary_1", "fast_dyn_matrix::abs" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_dyn_matrix<double, 2, 2> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_dyn_matrix/min", "fast_dyn_matrix::min" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_dyn_matrix<double, 2, 2> d(min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
}

//}}} Unary operators test

//{{{ Complex tests

TEST_CASE( "fast_dyn_matrix/complex", "fast_dyn_matrix::complex" ) {
    etl::fast_dyn_matrix<double, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<double, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_dyn_matrix<double, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_dyn_matrix<double, 2, 2> d(2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

//}}} Complex tests
