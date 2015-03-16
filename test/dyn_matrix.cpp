//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

//{{{ Init tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3, 2, 3.3);

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Approx(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_2", "dyn_matrix::operator=(T)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3, 2);

    test_matrix = 3.3;

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
    }
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_3", "dyn_matrix::dyn_matrix(initializer_list)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3,2, std::initializer_list<Z>({1.0, 3.0, 5.0, 2.0, 3.0, 4.0}));

    REQUIRE(test_matrix.rows() == 3);
    REQUIRE(test_matrix.columns() == 2);
    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix[0] == 1.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 5.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_5", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z,4> a(3, 2, 4, 5);

    a = 3.4;

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 120);
    REQUIRE(a(0,0,0,0) == 3.4);
    REQUIRE(a(1,1,1,1) == 3.4);

    etl::dyn_matrix<Z> b(3, 2, std::initializer_list<Z>({1,2,3,4,5,6}));

    REQUIRE(b.rows() == 3);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 6);
    REQUIRE(b[0] == 1);
    REQUIRE(b[1] == 2);

    etl::dyn_matrix<Z> c(3, 2, etl::init_flag, 3.3);

    REQUIRE(c.rows() == 3);
    REQUIRE(c.columns() == 2);
    REQUIRE(c.size() == 6);
    REQUIRE(c[0] == 3.3);
    REQUIRE(c[1] == 3.3);

    etl::dyn_matrix<Z> d(3, 2, 3.3);

    REQUIRE(d.rows() == 3);
    REQUIRE(d.columns() == 2);
    REQUIRE(d.size() == 6);
    REQUIRE(d[0] == 3.3);
    REQUIRE(d[1] == 3.3);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_6", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> b(3, 2, etl::values(1,2,3,4,5,6));

    REQUIRE(b.rows() == 3);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 6);
    REQUIRE(b(0,0) == 1);
    REQUIRE(b(0,1) == 2);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_7", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z, 4> b(2, 2, 2, 2, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));

    REQUIRE(b.rows() == 2);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 16);
    REQUIRE(b(0,0,0,0) == 1);
    REQUIRE(b(0,0,0,1) == 2);
    REQUIRE(b(0,0,1,0) == 3);
    REQUIRE(b(0,0,1,1) == 4);
    REQUIRE(b(0,1,0,0) == 5);
    REQUIRE(b(0,1,0,1) == 6);
    REQUIRE(b(0,1,1,0) == 7);
    REQUIRE(b(0,1,1,1) == 8);
    REQUIRE(b(1,0,0,0) == 9);
    REQUIRE(b(1,0,0,1) == 10);
    REQUIRE(b(1,0,1,0) == 11);
    REQUIRE(b(1,0,1,1) == 12);
    REQUIRE(b(1,1,0,0) == 13);
    REQUIRE(b(1,1,0,1) == 14);
    REQUIRE(b(1,1,1,0) == 15);
    REQUIRE(b(1,1,1,1) == 16);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/init_8", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z, 1> test_matrix(6, 3.3);

    REQUIRE(test_matrix.rows() == 6);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == 3.3);
        REQUIRE(test_matrix(i) == 3.3);
    }
}

//}}} Init tests

//{{{ Binary operators test

TEMPLATE_TEST_CASE_2( "dyn_matrix/add_scalar_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 + test_matrix;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/add_scalar_2", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix + 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/add_scalar_3", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix += 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/add_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/add_2", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sub_scalar_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 - test_matrix;

    REQUIRE(test_matrix[0] == 2.0);
    REQUIRE(test_matrix[1] == -1.0);
    REQUIRE(test_matrix[2] == -4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sub_scalar_2", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix - 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sub_scalar_3", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix -= 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sub_1", "dyn_matrix::operator-", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sub_2", "dyn_matrix::operator-=", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_scalar_1", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 * test_matrix;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_scalar_2", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix * 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);

}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_scalar_3", "dyn_matrix::operator*=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix *= 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_1", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a * b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_2", "dyn_matrix::operator*=", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix / 2.5;

    REQUIRE(test_matrix[0] == -1.0 / 2.5);
    REQUIRE(test_matrix[1] ==  2.0 / 2.5);
    REQUIRE(test_matrix[2] ==  5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 / test_matrix;

    REQUIRE(test_matrix[0] == 2.5 / -1.0);
    REQUIRE(test_matrix[1] == 2.5 /  2.0);
    REQUIRE(test_matrix[2] == 2.5 /  5.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_3", "dyn_matrix::operator/=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix /= 2.5;

    REQUIRE(test_matrix[0] == -1.0 / 2.5);
    REQUIRE(test_matrix[1] ==  2.0 / 2.5);
    REQUIRE(test_matrix[2] ==  5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a / b);

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a /= b;

    REQUIRE(a[0] == -1.0 / 2.5);
    REQUIRE(a[1] == 2.0 / 3.0);
    REQUIRE(a[2] == 5.0 / 4.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mod_scalar_1", "dyn_matrix::operator%", Z, double, float) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = test_matrix % 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mod_scalar_2", "dyn_matrix::operator%", Z, double, float) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = 2 % test_matrix;

    REQUIRE(test_matrix[0] == 2 % -1);
    REQUIRE(test_matrix[1] == 2 %  2);
    REQUIRE(test_matrix[2] == 2 %  5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mod_scalar_3", "dyn_matrix::operator%=", Z, double, float) {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix %= 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mod_1", "dyn_matrix::operator%", Z, double, float) {
    etl::dyn_matrix<int> a(2,2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2,2, std::initializer_list<int>({2, 3, 4, 1}));

    etl::dyn_matrix<int> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/mod_2", "dyn_matrix::operator%=", Z, double, float) {
    etl::dyn_matrix<int> a(2,2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2,2, std::initializer_list<int>({2, 3, 4, 1}));

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/log", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    etl::dyn_matrix<Z> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/abs", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sign", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/unary_unary", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 3.0}));

    etl::dyn_matrix<Z> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/unary_binary_1", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/unary_binary_2", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sigmoid", "dyn_matrix::sigmoid", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(-1.0));
    REQUIRE(d[1] == etl::logistic_sigmoid(2.0));
    REQUIRE(d[2] == etl::logistic_sigmoid(0.0));
    REQUIRE(d[3] == etl::logistic_sigmoid(1.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/softplus", "dyn_matrix::softplus", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::softplus(a));

    REQUIRE(d[0] == etl::softplus(-1.0));
    REQUIRE(d[1] == etl::softplus(2.0));
    REQUIRE(d[2] == etl::softplus(0.0));
    REQUIRE(d[3] == etl::softplus(1.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/exp", "dyn_matrix::exp", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::exp(a));

    REQUIRE(d[0] == std::exp(-1.0));
    REQUIRE(d[1] == std::exp(2.0));
    REQUIRE(d[2] == std::exp(0.0));
    REQUIRE(d[3] == std::exp(1.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/max", "dyn_matrix::max", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::max(a, 1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
    REQUIRE(d[3] == 1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/min", "dyn_matrix::min", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/bernoulli", "dyn_matrix::bernoulli", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
    REQUIRE(binary(d[3]));
}

//}}} Unary operators test

//{{{ Complex tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/complex", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2,2, std::initializer_list<Z>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d(2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/complex_2", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({1.1, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, -3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2,2, std::initializer_list<Z>({2.2, 3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d(2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/complex_3", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2,2, std::initializer_list<Z>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d(2.5 / (a * b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests

//{{{ Reductions

TEMPLATE_TEST_CASE_2( "dny_matrix/sum", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sum_2", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/sum_3", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/min_reduc", "min", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = min(a);

    REQUIRE(d == -1.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/max_reduc", "max", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = max(a);

    REQUIRE(d == 8.5);
}

//}}} Reductions

//{{{ is_finite tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_1", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    REQUIRE(a.is_finite());
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_2", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, NAN, 5.0, 1.0}));

    REQUIRE(a.is_finite() == false);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_3", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 1.0, INFINITY, 1.0}));

    REQUIRE(a.is_finite() == false);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_4", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    REQUIRE(etl::is_finite(a));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_5", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, NAN, 5.0, 1.0}));

    REQUIRE(etl::is_finite(a) == false);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_6", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 1.0, INFINITY, 1.0}));

    REQUIRE(etl::is_finite(a) == false);
}

//}}} is_finite tests
