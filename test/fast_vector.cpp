//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_vector.hpp"

//{{{ Init tests

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

//}}} Init tests

//{{{ Binary operators test

TEST_CASE( "fast_vector/add_scalar_1", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add_scalar_2", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add_scalar_3", "fast_vector::operator+=" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEST_CASE( "fast_vector/add_1", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEST_CASE( "fast_vector/add_2", "fast_vector::operator+=" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEST_CASE( "fast_vector/sub_scalar_1", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE(test_vector[0] == 2.0);
    REQUIRE(test_vector[1] == -1.0);
    REQUIRE(test_vector[2] == -4.5);
}

TEST_CASE( "fast_vector/sub_scalar_2", "fast_vector::operator+" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "fast_vector/sub_scalar_3", "fast_vector::operator+=" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEST_CASE( "fast_vector/sub_1", "fast_vector::operator-" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEST_CASE( "fast_vector/sub_2", "fast_vector::operator-=" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEST_CASE( "fast_vector/mul_scalar_1", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul_scalar_2", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul_scalar_3", "fast_vector::operator*=" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEST_CASE( "fast_vector/mul_1", "fast_vector::operator*" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c(a * b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEST_CASE( "fast_vector/mul_2", "fast_vector::operator*=" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEST_CASE( "fast_vector/div_scalar_1", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] ==  2.0 / 2.5);
    REQUIRE(test_vector[2] ==  5.0 / 2.5);
}

TEST_CASE( "fast_vector/div_scalar_2", "fast_vector::operator/" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE(test_vector[0] == 2.5 / -1.0);
    REQUIRE(test_vector[1] == 2.5 /  2.0);
    REQUIRE(test_vector[2] == 2.5 /  5.0);
}

TEST_CASE( "fast_vector/div_scalar_3", "fast_vector::operator/=" ) {
    etl::fast_vector<double, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= 2.5;

    REQUIRE(test_vector[0] == -1.0 / 2.5);
    REQUIRE(test_vector[1] == 2.0 / 2.5);
    REQUIRE(test_vector[2] == 5.0 / 2.5);
}

TEST_CASE( "fast_vector/div_1", "fast_vector::operator/"){
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> c(a / b);

    REQUIRE(c[0] == -1.0 / 2.5);
    REQUIRE(c[1] == 2.0 / 3.0);
    REQUIRE(c[2] == 5.0 / 4.0);
}

TEST_CASE( "fast_vector/div_2", "fast_vector::operator/="){
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE(a[0] == -1.0 / 2.5);
    REQUIRE(a[1] == 2.0 / 3.0);
    REQUIRE(a[2] == 5.0 / 4.0);
}

TEST_CASE( "fast_vector/mod_scalar_1", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod_scalar_2", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE(test_vector[0] == 2 % -1);
    REQUIRE(test_vector[1] == 2 %  2);
    REQUIRE(test_vector[2] == 2 %  5);
}

TEST_CASE( "fast_vector/mod_scalar_3", "fast_vector::operator%=" ) {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod_1", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    etl::fast_vector<int, 3> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "fast_vector/mod_2", "fast_vector::operator%" ) {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEST_CASE( "fast_vector/log", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};

    etl::fast_vector<double, 3> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == log(2.0));
    REQUIRE(d[2] == log(5.0));
}

TEST_CASE( "fast_vector/abs", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/sign", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_unary", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_binary_1", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/unary_binary_2", "fast_vector::abs" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEST_CASE( "fast_vector/sigmoid", "fast_vector::sigmoid" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(-1.0));
    REQUIRE(d[1] == etl::logistic_sigmoid(2.0));
    REQUIRE(d[2] == etl::logistic_sigmoid(0.0));
}

TEST_CASE( "fast_vector/softplus", "fast_vector::softplus" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(etl::softplus(a));

    REQUIRE(d[0] == etl::softplus(-1.0));
    REQUIRE(d[1] == etl::softplus(2.0));
    REQUIRE(d[2] == etl::softplus(0.0));
}

TEST_CASE( "fast_vector/exp", "fast_vector::exp" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(etl::exp(a));

    REQUIRE(d[0] == std::exp(-1.0));
    REQUIRE(d[1] == std::exp(2.0));
    REQUIRE(d[2] == std::exp(0.0));
}

TEST_CASE( "fast_vector/max", "fast_vector::max" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(etl::max(a, 1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
}

TEST_CASE( "fast_vector/min", "fast_vector::min" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<double, 3> d(etl::min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEST_CASE( "fast_vector/bernoulli", "fast_vector::bernoulli" ) {
    etl::fast_vector<double, 3> a = {-1.0, 0.3, 0.7};

    etl::fast_vector<double, 3> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
}

//}}} Unary operators test

//{{{ Reductions

TEST_CASE( "fast_vector/sum", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEST_CASE( "fast_vector/sum_2", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEST_CASE( "fast_vector/sum_3", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEST_CASE( "fast_vector/dot_1", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<double, 3> b = {2.0, 3.0, 2.0};

    auto d = dot(a, b);

    REQUIRE(d == 21.0);
}

TEST_CASE( "fast_vector/dot_2", "sum" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<double, 3> b = {2.0, 3.0, 2.0};

    auto d = dot(a, -1 * b);

    REQUIRE(d == -21.0);
}

TEST_CASE( "fast_vector/min", "min" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = min(a);

    REQUIRE(d == -1.0);
}

TEST_CASE( "fast_vector/max", "max" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 8.5};

    auto d = max(a);

    REQUIRE(d == 8.5);
}

//}}} Reductions

//{{{ Complex tests

TEST_CASE( "fast_vector/complex", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<double, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<double, 3> d(2.5 * ((a * b) / (a + c)) / (1.5 * a * b / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEST_CASE( "fast_vector/complex_2", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {1.1, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, -3.0, 4.0};
    etl::fast_vector<double, 3> c = {2.2, 3.0, 3.5};

    etl::fast_vector<double, 3> d(2.5 * ((a * b) / (log(a) * abs(c))) / (1.5 * a * sign(b) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEST_CASE( "fast_vector/complex_3", "fast_vector::complex" ) {
    etl::fast_vector<double, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<double, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<double, 3> d(2.5 / (a * b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests