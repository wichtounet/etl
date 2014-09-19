//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/dyn_matrix.hpp"

//{{{ Init tests

TEST_CASE( "dyn_matrix/init_1", "dyn_matrix::dyn_matrix(T)" ) {
    etl::dyn_matrix<double> test_matrix(3, 2, 3.3);

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "dyn_matrix/init_2", "dyn_matrix::operator=(T)" ) {
    etl::dyn_matrix<double> test_matrix(3, 2);

    test_matrix = 3.3;

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEST_CASE( "dyn_matrix/init_3", "dyn_matrix::dyn_matrix(initializer_list)" ) {
    etl::dyn_matrix<double> test_matrix(3,2, std::initializer_list<double>({1.0, 3.0, 5.0, 2.0, 3.0, 4.0}));

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix[0] == 1.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 5.0);
}

TEST_CASE( "dyn_matrix/init_5", "dyn_matrix::dyn_matrix(T)" ) {
    etl::dyn_matrix<double> a(3, 2, 4, 5);

    a = 3.4;

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 120);
    REQUIRE(a(0,0,0,0) == 3.4);
    REQUIRE(a(1,1,1,1) == 3.4);

    etl::dyn_matrix<double> b(3, 2, std::initializer_list<double>({1,2,3,4,5,6}));

    REQUIRE(b.rows() == 3);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 6);
    REQUIRE(b[0] == 1);
    REQUIRE(b[1] == 2);

    etl::dyn_matrix<double> c(3, 2, etl::init_flag, 3.3);

    REQUIRE(c.rows() == 3);
    REQUIRE(c.columns() == 2);
    REQUIRE(c.size() == 6);
    REQUIRE(c[0] == 3.3);
    REQUIRE(c[1] == 3.3);

    etl::dyn_matrix<double> d(3, 2, 3.3);

    REQUIRE(d.rows() == 3);
    REQUIRE(d.columns() == 2);
    REQUIRE(d.size() == 6);
    REQUIRE(d[0] == 3.3);
    REQUIRE(d[1] == 3.3);
}

TEST_CASE( "dyn_matrix/init_6", "dyn_matrix::dyn_matrix(T)" ) {
    etl::dyn_matrix<double> b(3, 2, etl::values(1,2,3,4,5,6));

    REQUIRE(b.rows() == 3);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 6);
    REQUIRE(b(0,0) == 1);
    REQUIRE(b(0,1) == 2);
}

//}}} Init tests

//{{{ Binary operators test

TEST_CASE( "dyn_matrix/add_scalar_1", "dyn_matrix::operator+" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 + test_matrix;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "dyn_matrix/add_scalar_2", "dyn_matrix::operator+" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix + 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "dyn_matrix/add_scalar_3", "dyn_matrix::operator+=" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix += 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEST_CASE( "dyn_matrix/add_1", "dyn_matrix::operator+" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<double> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "dyn_matrix/add_2", "dyn_matrix::operator+=" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEST_CASE( "dyn_matrix/sub_scalar_1", "dyn_matrix::operator+" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 - test_matrix;

    REQUIRE(test_matrix[0] == 2.0);
    REQUIRE(test_matrix[1] == -1.0);
    REQUIRE(test_matrix[2] == -4.5);
}

TEST_CASE( "dyn_matrix/sub_scalar_2", "dyn_matrix::operator+" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix - 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEST_CASE( "dyn_matrix/sub_scalar_3", "dyn_matrix::operator+=" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix -= 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEST_CASE( "dyn_matrix/sub_1", "dyn_matrix::operator-" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<double> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "dyn_matrix/sub_2", "dyn_matrix::operator-=" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEST_CASE( "dyn_matrix/mul_scalar_1", "dyn_matrix::operator*" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 * test_matrix;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEST_CASE( "dyn_matrix/mul_scalar_2", "dyn_matrix::operator*" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix * 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);

}

TEST_CASE( "dyn_matrix/mul_scalar_3", "dyn_matrix::operator*=" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix *= 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEST_CASE( "dyn_matrix/mul_1", "dyn_matrix::operator*" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<double> c(a * b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "dyn_matrix/mul_2", "dyn_matrix::operator*=" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEST_CASE( "dyn_matrix/div_scalar_1", "dyn_matrix::operator/" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix / 2.5;

    REQUIRE(test_matrix[0] == -1.0 / 2.5);
    REQUIRE(test_matrix[1] ==  2.0 / 2.5);
    REQUIRE(test_matrix[2] ==  5.0 / 2.5);
}

TEST_CASE( "dyn_matrix/div_scalar_2", "dyn_matrix::operator/" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 / test_matrix;

    REQUIRE(test_matrix[0] == 2.5 / -1.0);
    REQUIRE(test_matrix[1] == 2.5 /  2.0);
    REQUIRE(test_matrix[2] == 2.5 /  5.0);
}

TEST_CASE( "dyn_matrix/div_scalar_3", "dyn_matrix::operator/=" ) {
    etl::dyn_matrix<double> test_matrix(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix /= 2.5;

    REQUIRE(test_matrix[0] == -1.0 / 2.5);
    REQUIRE(test_matrix[1] ==  2.0 / 2.5);
    REQUIRE(test_matrix[2] ==  5.0 / 2.5);
}

TEST_CASE( "dyn_matrix/div_1", "dyn_matrix::operator/" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<double> c(a / b);

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "dyn_matrix/div_2", "dyn_matrix::operator/" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));

    a /= b;

    REQUIRE(a[0] == -1.0 / 2.5);
    REQUIRE(a[1] == 2.0 / 3.0);
    REQUIRE(a[2] == 5.0 / 4.0);
}

TEST_CASE( "dyn_matrix/mod_scalar_1", "dyn_matrix::operator%" ) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = test_matrix % 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "dyn_matrix/mod_scalar_2", "dyn_matrix::operator%" ) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = 2 % test_matrix;

    REQUIRE(test_matrix[0] == 2 % -1);
    REQUIRE(test_matrix[1] == 2 %  2);
    REQUIRE(test_matrix[2] == 2 %  5);
}

TEST_CASE( "dyn_matrix/mod_scalar_3", "dyn_matrix::operator%=" ) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix %= 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "dyn_matrix/mod_1", "dyn_matrix::operator%" ) {
    etl::dyn_matrix<int> a(2,2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2,2, std::initializer_list<int>({2, 3, 4, 1}));

    etl::dyn_matrix<int> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "dyn_matrix/mod_2", "dyn_matrix::operator%=" ) {
    etl::dyn_matrix<int> a(2,2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2,2, std::initializer_list<int>({2, 3, 4, 1}));

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "dyn_matrix/log", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));

    etl::dyn_matrix<double> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "dyn_matrix/abs", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_matrix/sign", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_matrix/unary_unary", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 3.0}));

    etl::dyn_matrix<double> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_matrix/unary_binary_1", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_matrix/unary_binary_2", "dyn_matrix::abs" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "dyn_matrix/sigmoid", "dyn_matrix::sigmoid" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(-1.0));
    REQUIRE(d[1] == etl::logistic_sigmoid(2.0));
    REQUIRE(d[2] == etl::logistic_sigmoid(0.0));
    REQUIRE(d[3] == etl::logistic_sigmoid(1.0));
}

TEST_CASE( "dyn_matrix/softplus", "dyn_matrix::softplus" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::softplus(a));

    REQUIRE(d[0] == etl::softplus(-1.0));
    REQUIRE(d[1] == etl::softplus(2.0));
    REQUIRE(d[2] == etl::softplus(0.0));
    REQUIRE(d[3] == etl::softplus(1.0));
}

TEST_CASE( "dyn_matrix/exp", "dyn_matrix::exp" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::exp(a));

    REQUIRE(d[0] == std::exp(-1.0));
    REQUIRE(d[1] == std::exp(2.0));
    REQUIRE(d[2] == std::exp(0.0));
    REQUIRE(d[3] == std::exp(1.0));
}

TEST_CASE( "dyn_matrix/max", "dyn_matrix::max" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::max(a, 1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
    REQUIRE(d[3] == 1.0);
}

TEST_CASE( "dyn_matrix/min", "dyn_matrix::min" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEST_CASE( "dyn_matrix/bernoulli", "dyn_matrix::bernoulli" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<double> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
    REQUIRE(binary(d[3]));
}

//}}} Unary operators test

//{{{ Complex tests

TEST_CASE( "dyn_matrix/complex", "dyn_matrix::complex" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<double> c(2,2, std::initializer_list<double>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<double> d(2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEST_CASE( "dyn_matrix/complex_2", "dyn_matrix::complex" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({1.1, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, -3.0, 4.0, 1.0}));
    etl::dyn_matrix<double> c(2,2, std::initializer_list<double>({2.2, 3.0, 3.5, 1.0}));

    etl::dyn_matrix<double> d(2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEST_CASE( "dyn_matrix/complex_3", "dyn_matrix::complex" ) {
    etl::dyn_matrix<double> a(2,2, std::initializer_list<double>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<double> b(2,2, std::initializer_list<double>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<double> c(2,2, std::initializer_list<double>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<double> d(2.5 / (a * b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests

//{{{ Reductions

TEST_CASE( "dny_matrix/sum", "sum" ) {
    etl::dyn_matrix<double> a(3, 1, std::initializer_list<double>({-1.0, 2.0, 8.5}));

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEST_CASE( "dyn_matrix/sum_2", "sum" ) {
    etl::dyn_matrix<double> a(3, 1, std::initializer_list<double>({-1.0, 2.0, 8.5}));

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEST_CASE( "dyn_matrix/sum_3", "sum" ) {
    etl::dyn_matrix<double> a(3, 1, std::initializer_list<double>({-1.0, 2.0, 8.5}));

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEST_CASE( "dyn_matrix/min_reduc", "min" ) {
    etl::dyn_matrix<double> a(3, 1, std::initializer_list<double>({-1.0, 2.0, 8.5}));

    auto d = min(a);

    REQUIRE(d == -1.0);
}

TEST_CASE( "dyn_matrix/max_reduc", "max" ) {
    etl::dyn_matrix<double> a(3, 1, std::initializer_list<double>({-1.0, 2.0, 8.5}));

    auto d = max(a);

    REQUIRE(d == 8.5);
}

//}}} Reductions