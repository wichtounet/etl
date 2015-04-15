//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <cmath>

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

//{{{ Init tests

TEMPLATE_TEST_CASE_2( "fast_matrix/init_1", "fast_matrix::fast_matrix(T)", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix(3.3);

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_matrix/init_2", "fast_matrix::operator=(T)", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix;

    test_matrix = 3.3;

    REQUIRE(test_matrix.size() == 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_matrix/init_3", "fast_matrix::fast_matrix(initializer_list)", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {1.0, 3.0, 5.0, 2.0};

    REQUIRE(test_matrix.size() == 4);

    REQUIRE(test_matrix[0] == 1.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 5.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/init_4", "fast_matrix::fast_matrix(T)", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3, 4> test_matrix(3.3);

    REQUIRE(test_matrix.size() == 24);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_matrix/init_5", "fast_matrix::operator=(T)", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3, 4> test_matrix;

    test_matrix = 3.3;

    REQUIRE(test_matrix.size() == 24);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_matrix/init_6", "fast_matrix::operator=(T)", Z, float, double ) {
    etl::fast_matrix<Z, 5> test_matrix(3.3);

    REQUIRE(test_matrix.size() == 5);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == Z(3.3));
        REQUIRE(test_matrix(i) == Z(3.3));
    }
}

//}}} Init tests

TEMPLATE_TEST_CASE_2( "fast_matrix/access", "fast_matrix::operator()", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

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

TEMPLATE_TEST_CASE_2( "fast_matrix/add_scalar_1", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 + test_matrix;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_scalar_2", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix + 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_scalar_3", "fast_matrix::operator+=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix += 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_scalar_4", "fast_matrix::operator+=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0, 1.0, 1.0, 1.0, 1.0};

    test_matrix += 1.0;

    REQUIRE(test_matrix[0] == 0.0);
    REQUIRE(test_matrix[1] == 3.0);
    REQUIRE(test_matrix[2] == 6.5);
    REQUIRE(test_matrix[7] == 2.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_1", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_2", "fast_matrix::operator+=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_3", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 2> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
    REQUIRE(c[7] ==  2.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/add_4", "fast_matrix::operator+=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
    REQUIRE(a[7] ==  2.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sub_scalar_1", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 - test_matrix;

    REQUIRE(test_matrix[0] == 2.0);
    REQUIRE(test_matrix[1] == -1.0);
    REQUIRE(test_matrix[2] == -4.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sub_scalar_2", "fast_matrix::operator+", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix - 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sub_scalar_3", "fast_matrix::operator+=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix -= 1.0;

    REQUIRE(test_matrix[0] == -2.0);
    REQUIRE(test_matrix[1] == 1.0);
    REQUIRE(test_matrix[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sub_1", "fast_matrix::operator-", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sub_2", "fast_matrix::operator-=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/mul_scalar_1", "fast_matrix::operator*", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 * test_matrix;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/mul_scalar_2", "fast_matrix::operator*", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix * 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);

}

TEMPLATE_TEST_CASE_2( "fast_matrix/mul_scalar_3", "fast_matrix::operator*=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix *= 2.5;

    REQUIRE(test_matrix[0] == -2.5);
    REQUIRE(test_matrix[1] ==  5.0);
    REQUIRE(test_matrix[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/mul_1", "fast_matrix::operator*", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c(scale(a,b));

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/mul_2", "fast_matrix::operator*=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/div_scalar_1", "fast_matrix::operator/", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix / 2.5;

    REQUIRE(test_matrix[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_matrix[1] == Approx( 2.0 / 2.5));
    REQUIRE(test_matrix[2] == Approx( 5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/div_scalar_2", "fast_matrix::operator/", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 / test_matrix;

    REQUIRE(test_matrix[0] == Approx(2.5 / -1.0));
    REQUIRE(test_matrix[1] == Approx(2.5 /  2.0));
    REQUIRE(test_matrix[2] == Approx(2.5 /  5.0));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/div_scalar_3", "fast_matrix::operator/=", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix /= 2.5;

    REQUIRE(test_matrix[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_matrix[1] == Approx(2.0 / 2.5));
    REQUIRE(test_matrix[2] == Approx(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/div_1", "fast_matrix::operator/", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c(a / b);

    REQUIRE(c[0] == Approx(-1.0 / 2.5));
    REQUIRE(c[1] == Approx(2.0 / 3.0));
    REQUIRE(c[2] == Approx(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/div_2", "fast_matrix::operator/", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a /= b;

    REQUIRE(a[0] == Approx(-1.0 / 2.5));
    REQUIRE(a[1] == Approx(2.0 / 3.0));
    REQUIRE(a[2] == Approx(5.0 / 4.0));
}

TEST_CASE( "fast_matrix/mod_scalar_1", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = test_matrix % 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "fast_matrix/mod_scalar_2", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = 2 % test_matrix;

    REQUIRE(test_matrix[0] == 2 % -1);
    REQUIRE(test_matrix[1] == 2 %  2);
    REQUIRE(test_matrix[2] == 2 %  5);
}

TEST_CASE( "fast_matrix/mod_scalar_3", "fast_matrix::operator%=") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix %= 2;

    REQUIRE(test_matrix[0] == -1 % 2);
    REQUIRE(test_matrix[1] ==  2 % 2);
    REQUIRE(test_matrix[2] ==  5 % 2);
}

TEST_CASE( "fast_matrix/mod_1", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> a = {-1, 2, 5, 1};
    etl::fast_matrix<int, 2, 2> b = {2, 3, 4, 1};

    etl::fast_matrix<int, 2, 2> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "fast_matrix/mod_2", "fast_matrix::operator%=") {
    etl::fast_matrix<int, 2, 2> a = {-1, 2, 5, 1};
    etl::fast_matrix<int, 2, 2> b = {2, 3, 4, 1};

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEMPLATE_TEST_CASE_2( "fast_matrix/minus_1", "fast_matrix::minus", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(-a);

    REQUIRE(d(0,0) == 1.0);
    REQUIRE(d(0,1) == -2.0);
    REQUIRE(d(1,0) == -5.0);
    REQUIRE(d(1,1) == -1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/log_1", "fast_matrix::log", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == std::log(Z(2.0)));
    REQUIRE(d[2] == std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/log_2", "fast_matrix::log", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == std::log(Z(2.0)));
    REQUIRE(d[2] == std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sqrt_1", "fast_matrix::sqrt", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(sqrt(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == Approx(std::sqrt(Z(2.0))));
    REQUIRE(d[2] == Approx(std::sqrt(Z(5.0))));
    REQUIRE(d[3] == Approx(std::sqrt(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sqrt_2", "fast_matrix::sqrt", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d(sqrt(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == Approx(std::sqrt(Z(2.0))));
    REQUIRE(d[2] == Approx(std::sqrt(Z(5.0))));
    REQUIRE(d[3] == Approx(std::sqrt(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sqrt_3", "fast_matrix::sqrt", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d(sqrt(a>>a));

    REQUIRE(d[0] == Approx(std::sqrt(Z(1.0))));
    REQUIRE(d[1] == Approx(std::sqrt(Z(4.0))));
    REQUIRE(d[2] == Approx(std::sqrt(Z(25.0))));
    REQUIRE(d[3] == Approx(std::sqrt(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/abs", "fast_matrix::abs", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sign", "fast_matrix::sign", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/unary_unary", "fast_matrix::abs", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_matrix<Z, 2, 2> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/unary_binary_1", "fast_matrix::abs", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/unary_binary_2", "fast_matrix::abs", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/sigmoid", "fast_matrix::sigmoid", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(sigmoid(a));

    REQUIRE(d[0] == Approx(etl::logistic_sigmoid(Z(-1.0))));
    REQUIRE(d[1] == Approx(etl::logistic_sigmoid(Z(2.0))));
    REQUIRE(d[2] == Approx(etl::logistic_sigmoid(Z(0.0))));
    REQUIRE(d[3] == Approx(etl::logistic_sigmoid(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/softplus", "fast_matrix::softplus", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(softplus(a));

    REQUIRE(d[0] == Approx(etl::softplus(Z(-1.0))));
    REQUIRE(d[1] == Approx(etl::softplus(Z(2.0))));
    REQUIRE(d[2] == Approx(etl::softplus(Z(0.0))));
    REQUIRE(d[3] == Approx(etl::softplus(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/exp", "fast_matrix::exp", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(exp(a));

    REQUIRE(d[0] == Approx(std::exp(Z(-1.0))));
    REQUIRE(d[1] == Approx(std::exp(Z(2.0))));
    REQUIRE(d[2] == Approx(std::exp(Z(0.0))));
    REQUIRE(d[3] == Approx(std::exp(Z(1.0))));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/max", "fast_matrix::max", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(max(a,1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
    REQUIRE(d[3] == 1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/min", "fast_matrix::min", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/pow_1", "fast_matrix::pow_1", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d(pow(a, 2));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/pow_2", "fast_matrix::pow_1", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d(pow((a >> a)+1.0, 2));

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == 25.0);
    REQUIRE(d[2] == 1.0);
    REQUIRE(d[3] == 4.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2( "fast_matrix/bernoulli", "fast_matrix::bernoulli", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
    REQUIRE(binary(d[3]));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/r_bernoulli", "fast_matrix::r_bernoulli", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(etl::r_bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
    REQUIRE(binary(d[3]));
}

//}}} Unary operators test

//{{{ Complex tests

TEMPLATE_TEST_CASE_2( "fast_matrix/complex", "fast_matrix::complex", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2> d(2.5 * ((a >> b) / (a + c)) / (1.5 * (a >> b) / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/complex_2", "fast_matrix::complex", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {1.1, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, -3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2> c = {2.2, 3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2> d(2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/complex_3", "fast_matrix::complex", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d(2.5 / (a >> b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/complex_4", "fast_matrix::complex", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, -3.0, 4.0, 1.0, 2.5, -3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> c = {2.2, 3.0, 3.5, 1.0, 2.2, 3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2, 2> d(2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
    REQUIRE(d[4] == Approx(46.39429));
    REQUIRE(d[5] == Approx(9.13499));
    REQUIRE(d[6] == Approx(5.8273));
}

//}}} Complex tests

//{{{ is_finite tests

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_1", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    REQUIRE(a.is_finite());
}

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_2", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, NAN, 5.0, 1.0};

    REQUIRE(a.is_finite() == false);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_3", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 1.0, INFINITY, 1.0};

    REQUIRE(a.is_finite() == false);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_4", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    REQUIRE(etl::is_finite(a));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_5", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, NAN, 5.0, 1.0};

    REQUIRE(etl::is_finite(a) == false);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/is_finite_6", "fast_matrix::is_finite", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 1.0, INFINITY, 1.0};

    REQUIRE(etl::is_finite(a) == false);
}

//}}} is_finite tests

//{{{ scale tests

TEMPLATE_TEST_CASE_2( "fast_matrix/scale_1", "", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    a *= 2.5;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] == 12.5);
    REQUIRE(a[3] == 2.5);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/scale_2", "", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 2.0, 3.0, -1.2};

    a *= b;

    REQUIRE(a[0] == Z(-2.5));
    REQUIRE(a[1] == Z( 4.0));
    REQUIRE(a[2] == Z(15.0));
    REQUIRE(a[3] == Z(-1.2));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/scale_3", "", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 2.0, 3.0, -1.2};

    a.scale(b);

    REQUIRE(a[0] == Z(-2.5));
    REQUIRE(a[1] == Z( 4.0));
    REQUIRE(a[2] == Z(15.0));
    REQUIRE(a[3] == Z(-1.2));
}

TEMPLATE_TEST_CASE_2( "fast_matrix/scale_4", "", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    a.scale(2.5);

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] == 12.5);
    REQUIRE(a[3] == 2.5);
}

//}}} scale tests

//{{{ swap tests

TEMPLATE_TEST_CASE_2( "fast_matrix/swap_1", "", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {1.0, 3.3, 4.4, 9, 10.1, -1.1};

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

TEMPLATE_TEST_CASE_2( "fast_matrix/swap_2", "", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {1.0, 3.3, 4.4, 9, 10.1, -1.1};

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

//}}} swap tests

//{{{ comparisons operators

TEMPLATE_TEST_CASE_2( "fast_matrix/operator==/1", "[fast][matrix]", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {1.0, 3.3, 4.4, 9, 10.1, -1.1};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);
}

TEMPLATE_TEST_CASE_2( "fast_matrix/operator==/2", "[fast][matrix]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};

    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a != b);
    REQUIRE(b != a);
}

//}}} swap tests
