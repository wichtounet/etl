//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl_light.hpp"

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
        REQUIRE(test_matrix[i] == Approx(3.3));
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
    REQUIRE(a(0,0,0,0) == Approx(3.4));
    REQUIRE(a(1,1,1,1) == Approx(3.4));

    etl::dyn_matrix<Z> b(3, 2, std::initializer_list<Z>({1,2,3,4,5,6}));

    REQUIRE(b.rows() == 3);
    REQUIRE(b.columns() == 2);
    REQUIRE(b.size() == 6);
    REQUIRE(b[0] == 1);
    REQUIRE(b[1] == 2);

    etl::dyn_matrix<Z> c(3, 2, etl::init_flag, Z(3.3));

    REQUIRE(c.rows() == 3);
    REQUIRE(c.columns() == 2);
    REQUIRE(c.size() == 6);
    REQUIRE(c[0] == Z(3.3));
    REQUIRE(c[1] == Z(3.3));

    etl::dyn_matrix<Z> d(3, 2, Z(3.3));

    REQUIRE(d.rows() == 3);
    REQUIRE(d.columns() == 2);
    REQUIRE(d.size() == 6);
    REQUIRE(d[0] == Z(3.3));
    REQUIRE(d[1] == Z(3.3));
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
    etl::dyn_matrix<Z, 1> test_matrix(6, Z(3.3));

    REQUIRE(test_matrix.rows() == 6);
    REQUIRE(test_matrix.size() == 6);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
        REQUIRE(test_matrix(i) == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/dim_0", "fast_matrix::fast_matrix(T)", Z, float, double ) {
    etl::dyn_matrix<Z, 6> test_matrix(2, 3, 4, 5, 6, 7);

    REQUIRE(test_matrix.template dim<0>() == 2);
    REQUIRE(test_matrix.template dim<1>() == 3);
    REQUIRE(test_matrix.template dim<2>() == 4);
    REQUIRE(test_matrix.template dim<3>() == 5);
    REQUIRE(test_matrix.template dim<4>() == 6);
    REQUIRE(test_matrix.template dim<5>() == 7);
    REQUIRE(test_matrix.dim(0) == 2);
    REQUIRE(test_matrix.dim(1) == 3);
    REQUIRE(test_matrix.dim(2) == 4);
    REQUIRE(test_matrix.dim(3) == 5);
    REQUIRE(test_matrix.dim(4) == 6);
    REQUIRE(test_matrix.dim(5) == 7);
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

    etl::dyn_matrix<Z> c(scale(a, b));

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

TEMPLATE_TEST_CASE_2( "dyn_matrix/mul_3", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a.scale(b));

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix / 2.5;

    REQUIRE(test_matrix[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_matrix[1] == Approx(2.0 / 2.5));
    REQUIRE(test_matrix[2] == Approx(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 / test_matrix;

    REQUIRE(test_matrix[0] == Approx(2.5 / -1.0));
    REQUIRE(test_matrix[1] == Approx(2.5 /  2.0));
    REQUIRE(test_matrix[2] == Approx(2.5 /  5.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_scalar_3", "dyn_matrix::operator/=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix /= 2.5;

    REQUIRE(test_matrix[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_matrix[1] == Approx(2.0 / 2.5));
    REQUIRE(test_matrix[2] == Approx(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c(a / b);

    REQUIRE(c[0] == Approx(-1.0 / 2.5));
    REQUIRE(c[1] == Approx(2.0 / 3.0));
    REQUIRE(c[2] == Approx(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/div_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a /= b;

    REQUIRE(a[0] == Approx(-1.0 / 2.5));
    REQUIRE(a[1] == Approx(2.0 / 3.0));
    REQUIRE(a[2] == Approx(5.0 / 4.0));
}

TEST_CASE( "dyn_matrix/mod_scalar_1", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = test_matrix % 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] == 2 % 2);
    REQUIRE(test_matrix[2] == 5 % 2);
}

TEST_CASE( "dyn_matrix/mod_scalar_2", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = 2 % test_matrix;

    REQUIRE(test_matrix[0] == 2 % -1);
    REQUIRE(test_matrix[1] == 2 %  2);
    REQUIRE(test_matrix[2] == 2 %  5);
}

TEST_CASE( "dyn_matrix/mod_scalar_3", "dyn_matrix::operator%=") {
    etl::dyn_matrix<int> test_matrix(2,2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix %= 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "dyn_matrix/mod_1", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> a(2,2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2,2, std::initializer_list<int>({2, 3, 4, 1}));

    etl::dyn_matrix<int> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "dyn_matrix/mod_2", "dyn_matrix::operator%=") {
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
    REQUIRE(d[1] == Approx(std::log(2.0)));
    REQUIRE(d[2] == Approx(std::log(5.0)));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/abs", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(abs(a));

    REQUIRE(d[0] == Z(1.0));
    REQUIRE(d[1] == Z(2.0));
    REQUIRE(d[2] == Z(0.0));
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

    REQUIRE(d[0] == Approx(etl::logistic_sigmoid(Z(-1.0))));
    REQUIRE(d[1] == Approx(etl::logistic_sigmoid(Z(2.0))));
    REQUIRE(d[2] == Approx(etl::logistic_sigmoid(Z(0.0))));
    REQUIRE(d[3] == Approx(etl::logistic_sigmoid(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/softplus", "dyn_matrix::softplus", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::softplus(a));

    REQUIRE(d[0] == Approx(etl::softplus(Z(-1.0))));
    REQUIRE(d[1] == Approx(etl::softplus(Z(2.0))));
    REQUIRE(d[2] == Approx(etl::softplus(Z(0.0))));
    REQUIRE(d[3] == Approx(etl::softplus(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/exp", "dyn_matrix::exp", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d(etl::exp(a));

    REQUIRE(d[0] == Approx(std::exp(Z(-1.0))));
    REQUIRE(d[1] == Approx(std::exp(Z(2.0))));
    REQUIRE(d[2] == Approx(std::exp(Z(0.0))));
    REQUIRE(d[3] == Approx(std::exp(Z(1.0))));
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

    etl::dyn_matrix<Z> d(2.5 * ((scale(a,b)) / (a + c)) / (1.5 * (a >> b) / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/complex_2", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({1.1, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, -3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2,2, std::initializer_list<Z>({2.2, 3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d(2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/complex_3", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2,2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2,2, std::initializer_list<Z>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d(2.5 / (a >> b));

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

    REQUIRE(!a.is_finite());
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/is_finite_3", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2,2, std::initializer_list<Z>({-1.0, 1.0, INFINITY, 1.0}));

    REQUIRE(!a.is_finite());
}

//}}} is_finite tests

//{{{ scale tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/scale_1", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));

    a *= 2.5;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] == 12.5);
    REQUIRE(a[3] == 2.5);
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/scale_2", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.5, 2.0, 3.0, -1.2));

    a *= b;

    REQUIRE(a[0] == Z(-2.5));
    REQUIRE(a[1] == Z( 4.0));
    REQUIRE(a[2] == Z(15.0));
    REQUIRE(a[3] == Z(-1.2));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/scale_3", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.5, 2.0, 3.0, -1.2));

    a.scale_inplace(b);

    REQUIRE(a[0] == Z(-2.5));
    REQUIRE(a[1] == Z( 4.0));
    REQUIRE(a[2] == Z(15.0));
    REQUIRE(a[3] == Z(-1.2));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/scale_4", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));

    a.scale_inplace(2.5);

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] == 12.5);
    REQUIRE(a[3] == 2.5);
}

//}}} scale tests

//{{{ swap tests

TEMPLATE_TEST_CASE_2( "dyn_matrix/swap_1", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<Z> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    etl::swap(a, b);

    REQUIRE(a[0] == Z(1.0));
    REQUIRE(a[1] == Z(3.3));
    REQUIRE(a[2] == Z(4.4));
    REQUIRE(a[3] == Z(9.0));
    REQUIRE(a[4] == Z(10.1));
    REQUIRE(a[5] == Z(-1.1));

    REQUIRE(b[0] == Z(-1.0));
    REQUIRE(b[1] == Z(2.0));
    REQUIRE(b[2] == Z(5.0));
    REQUIRE(b[3] == Z(1.0));
    REQUIRE(b[4] == Z(1.1));
    REQUIRE(b[5] == Z(1.9));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/swap_2", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<Z> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    a.swap(b);

    REQUIRE(a[0] == Z(1.0));
    REQUIRE(a[1] == Z(3.3));
    REQUIRE(a[2] == Z(4.4));
    REQUIRE(a[3] == Z(9.0));
    REQUIRE(a[4] == Z(10.1));
    REQUIRE(a[5] == Z(-1.1));

    REQUIRE(b[0] == Z(-1.0));
    REQUIRE(b[1] == Z(2.0));
    REQUIRE(b[2] == Z(5.0));
    REQUIRE(b[3] == Z(1.0));
    REQUIRE(b[4] == Z(1.1));
    REQUIRE(b[5] == Z(1.9));
}

TEMPLATE_TEST_CASE_2( "dyn_matrix/swap_3", "", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 4, Z(1));
    etl::dyn_matrix<Z> b(5, 6, Z(2));

    a.swap(b);

    REQUIRE(etl::size(a) == 30);
    REQUIRE(etl::size(b) == 8);
    REQUIRE(etl::dim<0>(a) == 5);
    REQUIRE(etl::dim<1>(a) == 6);
    REQUIRE(etl::dim<0>(b) == 2);
    REQUIRE(etl::dim<1>(b) == 4);
}

//}}} swap tests

//Make sure assign between matrices of different are compiling correctly

TEST_CASE( "dyn_matrix/assign_two_types", "" ) {
    etl::dyn_matrix<double> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<float> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    //This must compile
    a = b;
    b = a;

    etl::dyn_matrix<double> aa = b;
    etl::dyn_matrix<float> bb = a;

    etl::dyn_matrix<double> aaa(b);
    etl::dyn_matrix<float> bbb(a);
}

//Make sure default construction is possible and then size is modifiable

TEST_CASE( "dyn_matrix/default_constructor_1", "" ) {
    etl::dyn_matrix<double> def_a;
    etl::dyn_matrix<float> def_b;

    etl::dyn_matrix<double> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<float> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    def_a = a;
    def_b = b;

    REQUIRE(def_a.size() == a.size());
    REQUIRE(def_b.size() == b.size());

    REQUIRE(etl::dim<0>(def_a) == etl::dim<0>(a));
    REQUIRE(etl::dim<1>(def_a) == etl::dim<1>(a));

    REQUIRE(etl::dim<0>(def_b) == etl::dim<0>(b));
    REQUIRE(etl::dim<1>(def_b) == etl::dim<1>(b));

    REQUIRE(def_a(1, 1) == 1.0);
    REQUIRE(def_b(1, 1) == 9.0);
}

TEST_CASE( "dyn_matrix/default_constructor_2", "" ) {
    std::vector<etl::dyn_matrix<double>> values(10);

    REQUIRE(values[0].size() == 0);

    values[0] = etl::dyn_matrix<double>(3, 2);

    REQUIRE(values[0].size() == 6);
    REQUIRE(etl::dim<0>(values[0]) == 3);
    REQUIRE(etl::dim<1>(values[0]) == 2);
}

etl::dyn_matrix<double, 3> test_return(){
    return etl::dyn_matrix<double, 3>(3, 8, 1);
}

TEST_CASE( "dyn_matrix/default_constructor_3", "" ) {
    std::vector<etl::dyn_matrix<double, 3>> values;

    values.emplace_back();

    REQUIRE(values[0].size() == 0);

    values.emplace_back(5, 5, 1, 1.0);

    REQUIRE(values[0].size() == 0);
    REQUIRE(values[1].size() == 25);
    REQUIRE(values[1][0] == 1.0);

    values.push_back(etl::dyn_matrix<double, 3>(3,2,5,13.0));

    REQUIRE(values[0].size() == 0);
    REQUIRE(values[1].size() == 25);
    REQUIRE(values[1][0] == 1.0);
    REQUIRE(values[2].size() == 30);
    REQUIRE(values[2][0] == 13.0);

    values.shrink_to_fit();
    values.push_back(test_return());

    REQUIRE(values[0].size() == 0);
    REQUIRE(values[1].size() == 25);
    REQUIRE(values[1][0] == 1.0);
    REQUIRE(values[2].size() == 30);
    REQUIRE(values[2][0] == 13.0);
    REQUIRE(values[3].size() == 24);

    values.pop_back();
    values.shrink_to_fit();

    REQUIRE(values[0].size() == 0);
    REQUIRE(values[1].size() == 25);
    REQUIRE(values[1][0] == 1.0);
    REQUIRE(values[2].size() == 30);
    REQUIRE(values[2][0] == 13.0);

    std::vector<etl::dyn_matrix<double, 3>> values_2;
    values_2 = values;

    REQUIRE(values_2[0].size() == 0);
    REQUIRE(values_2[1].size() == 25);
    REQUIRE(values_2[1][0] == 1.0);
    REQUIRE(values_2[2].size() == 30);
    REQUIRE(values_2[2][0] == 13.0);
}
