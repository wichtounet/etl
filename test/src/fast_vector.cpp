//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

//{{{ Init tests

TEMPLATE_TEST_CASE_2( "fast_vector/init_1", "fast_vector::fast_vector(T)", Z, float, double ) {
    etl::fast_vector<Z, 4> test_vector(3.3);

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_vector/init_2", "fast_vector::operator=(T)", Z, float, double ) {
    etl::fast_vector<Z, 4> test_vector;

    test_vector = 3.3;

    REQUIRE(test_vector.size() == 4);

    for(std::size_t i = 0; i < test_vector.size(); ++i){
        REQUIRE(test_vector[i] == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2( "fast_vector/init_3", "fast_vector::fast_vector(initializer_list)", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {1.0, 2.0, 3.0};

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1.0);
    REQUIRE(test_vector[1] == 2.0);
    REQUIRE(test_vector[2] == 3.0);
}

//}}} Init tests

//{{{ Binary operators test

TEMPLATE_TEST_CASE_2( "fast_vector/add_scalar_1", "fast_vector::operator+", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/add_scalar_2", "fast_vector::operator+", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/add_scalar_3", "fast_vector::operator+=", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE(test_vector[0] == 0.0);
    REQUIRE(test_vector[1] == 3.0);
    REQUIRE(test_vector[2] == 6.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/add_1", "fast_vector::operator+", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c(a + b);

    REQUIRE(c[0] ==  1.5);
    REQUIRE(c[1] ==  5.0);
    REQUIRE(c[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/add_2", "fast_vector::operator+=", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE(a[0] ==  1.5);
    REQUIRE(a[1] ==  5.0);
    REQUIRE(a[2] ==  9.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sub_scalar_1", "fast_vector::operator+", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE(test_vector[0] == 2.0);
    REQUIRE(test_vector[1] == -1.0);
    REQUIRE(test_vector[2] == -4.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sub_scalar_2", "fast_vector::operator+", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sub_scalar_3", "fast_vector::operator+=", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE(test_vector[0] == -2.0);
    REQUIRE(test_vector[1] == 1.0);
    REQUIRE(test_vector[2] == 4.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sub_1", "fast_vector::operator-", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c(a - b);

    REQUIRE(c[0] == -3.5);
    REQUIRE(c[1] == -1.0);
    REQUIRE(c[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sub_2", "fast_vector::operator-=", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE(a[0] == -3.5);
    REQUIRE(a[1] == -1.0);
    REQUIRE(a[2] ==  1.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_scalar_1", "fast_vector::operator*", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_scalar_2", "fast_vector::operator*", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_scalar_3", "fast_vector::operator*=", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE(test_vector[0] == -2.5);
    REQUIRE(test_vector[1] ==  5.0);
    REQUIRE(test_vector[2] == 12.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_1", "fast_vector::operator*", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c(a >> b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_2", "fast_vector::operator*=", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE(a[0] == -2.5);
    REQUIRE(a[1] ==  6.0);
    REQUIRE(a[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/mul_3", "fast_vector::operator*", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c(a >> b);

    REQUIRE(c[0] == -2.5);
    REQUIRE(c[1] ==  6.0);
    REQUIRE(c[2] == 20.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/div_scalar_1", "fast_vector::operator/", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE(test_vector[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_vector[1] == Approx( 2.0 / 2.5));
    REQUIRE(test_vector[2] == Approx( 5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "fast_vector/div_scalar_2", "fast_vector::operator/", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE(test_vector[0] == Approx(2.5 / -1.0));
    REQUIRE(test_vector[1] == Approx(2.5 /  2.0));
    REQUIRE(test_vector[2] == Approx(2.5 /  5.0));
}

TEMPLATE_TEST_CASE_2( "fast_vector/div_scalar_3", "fast_vector::operator/=", Z, float, double ) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= 2.5;

    REQUIRE(test_vector[0] == Approx(-1.0 / 2.5));
    REQUIRE(test_vector[1] == Approx(2.0 / 2.5));
    REQUIRE(test_vector[2] == Approx(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2( "fast_vector/div_1", "fast_vector::operator/", Z, float, double ){
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c(a / b);

    REQUIRE(c[0] == Approx(-1.0 / 2.5));
    REQUIRE(c[1] == Approx(2.0 / 3.0));
    REQUIRE(c[2] == Approx(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2( "fast_vector/div_2", "fast_vector::operator/=", Z, float, double ){
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE(a[0] == Approx(-1.0 / 2.5));
    REQUIRE(a[1] == Approx(2.0 / 3.0));
    REQUIRE(a[2] == Approx(5.0 / 4.0));
}

TEST_CASE( "fast_vector/mod_scalar_1", "fast_vector::operator%") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod_scalar_2", "fast_vector::operator%") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE(test_vector[0] == 2 % -1);
    REQUIRE(test_vector[1] == 2 %  2);
    REQUIRE(test_vector[2] == 2 %  5);
}

TEST_CASE( "fast_vector/mod_scalar_3", "fast_vector::operator%=") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE(test_vector[0] == -1 % 2);
    REQUIRE(test_vector[1] ==  2 % 2);
    REQUIRE(test_vector[2] ==  5 % 2);
}

TEST_CASE( "fast_vector/mod_1", "fast_vector::operator%") {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    etl::fast_vector<int, 3> c(a % b);

    REQUIRE(c[0] == -1 % 2);
    REQUIRE(c[1] == 2 % 3);
    REQUIRE(c[2] == 5 % 4);
}

TEST_CASE( "fast_vector/mod_2", "fast_vector::operator%") {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    a %= b;

    REQUIRE(a[0] == -1 % 2);
    REQUIRE(a[1] == 2 % 3);
    REQUIRE(a[2] == 5 % 4);
}

//}}} Binary operator tests

//{{{ Unary operator tests

TEMPLATE_TEST_CASE_2( "fast_vector/log", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};

    etl::fast_vector<Z, 3> d(log(a));

    REQUIRE(std::isnan(d[0]));
    REQUIRE(d[1] == std::log(Z(2.0)));
    REQUIRE(d[2] == std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2( "fast_vector/abs", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(abs(a));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sign", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(sign(a));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/unary_unary", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(abs(sign(a)));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/unary_binary_1", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(abs(a + a));

    REQUIRE(d[0] == 2.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/unary_binary_2", "fast_vector::abs", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(abs(a) + a);

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 4.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sigmoid", "fast_vector::sigmoid", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::sigmoid(a));

    REQUIRE(d[0] == etl::logistic_sigmoid(Z(-1.0)));
    REQUIRE(d[1] == etl::logistic_sigmoid(Z(2.0)));
    REQUIRE(d[2] == etl::logistic_sigmoid(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "fast_vector/fast_sigmoid", "fast_vector::sigmoid", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::fast_sigmoid(a));

    REQUIRE(d[0] == Approx(etl::logistic_sigmoid(Z(-1.0))).epsilon(0.05));
    REQUIRE(d[1] == Approx(etl::logistic_sigmoid(Z(2.0))).epsilon(0.05));
    REQUIRE(d[2] == Approx(etl::logistic_sigmoid(Z(0.0))).epsilon(0.05));
}

TEMPLATE_TEST_CASE_2( "fast_vector/softmax_1", "fast_vector::softmax", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};

    etl::fast_vector<Z, 3> d(etl::softmax(a));

    auto sum = std::exp(Z(1.0)) + std::exp(Z(2.0)) + std::exp(Z(3.0));

    REQUIRE(d[0] == Approx(std::exp(Z(1.0)) / sum));
    REQUIRE(d[1] == Approx(std::exp(Z(2.0)) / sum));
    REQUIRE(d[2] == Approx(std::exp(Z(3.0)) / sum));

    REQUIRE(etl::mean(d) == Approx(1.0 / 3.0));
}

TEMPLATE_TEST_CASE_2( "fast_vector/softmax_2", "fast_vector::softmax", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 4.0, 5.0};

    etl::fast_vector<Z, 3> d(etl::softmax(a));

    auto sum = std::exp(Z(-1.0)) + std::exp(Z(4.0)) + std::exp(Z(5.0));

    REQUIRE(d[0] == Approx(std::exp(Z(-1.0)) / sum));
    REQUIRE(d[1] == Approx(std::exp(Z(4.0)) / sum));
    REQUIRE(d[2] == Approx(std::exp(Z(5.0)) / sum));

    REQUIRE(etl::mean(d) == Approx(1.0 / 3.0));
}

TEMPLATE_TEST_CASE_2( "softmax_3", "fast_vector::softmax", Z, float, double ) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    etl::fast_matrix<Z, 3, 3> d(etl::softmax(a));

    REQUIRE(etl::mean(d) == Approx(1.0 / 9.0));
}

TEMPLATE_TEST_CASE_2( "softmax_4", "fast_vector::softmax", Z, float, double ) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<Z, 3, 3> b(1.4);

    etl::fast_matrix<Z, 3, 3> d(etl::softmax(a + (a * 2) + b + (b >> a)));

    REQUIRE(etl::mean(d) == Approx(1.0 / 9.0));
}

TEMPLATE_TEST_CASE_2( "fast_vector/stable_softmax_1", "stable_softmax", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 4.0, 5.0};

    etl::fast_vector<Z, 3> d(etl::stable_softmax(a));

    auto sum = std::exp(Z(-1.0)) + std::exp(Z(4.0)) + std::exp(Z(5.0));

    REQUIRE(d[0] == Approx(std::exp(Z(-1.0)) / sum));
    REQUIRE(d[1] == Approx(std::exp(Z(4.0)) / sum));
    REQUIRE(d[2] == Approx(std::exp(Z(5.0)) / sum));

    REQUIRE(etl::mean(d) == Approx(1.0 / 3.0));
}

TEMPLATE_TEST_CASE_2( "fast_vector/softplus", "fast_vector::softplus", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::softplus(a));

    REQUIRE(d[0] == Approx(etl::softplus(Z(-1.0))));
    REQUIRE(d[1] == Approx(etl::softplus(Z(2.0))));
    REQUIRE(d[2] == Approx(etl::softplus(Z(0.0))));
}

TEMPLATE_TEST_CASE_2( "fast_vector/exp", "fast_vector::exp", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::exp(a));

    REQUIRE(d[0] == Approx(std::exp(Z(-1.0))));
    REQUIRE(d[1] == Approx(std::exp(Z(2.0))));
    REQUIRE(d[2] == Approx(std::exp(Z(0.0))));
}

TEMPLATE_TEST_CASE_2( "fast_vector/max", "fast_vector::max", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::max(a, 1.0));

    REQUIRE(d[0] == 1.0);
    REQUIRE(d[1] == 2.0);
    REQUIRE(d[2] == 1.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/min", "fast_vector::min", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::min(a, 1.0));

    REQUIRE(d[0] == -1.0);
    REQUIRE(d[1] == 1.0);
    REQUIRE(d[2] == 0.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/one_if", "fast_vector::one_if", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d(etl::one_if(a, 0.0));

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 0.0);
    REQUIRE(d[2] == 1.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/one_if_max", "fast_vector::one_if_max", Z, float, double ) {
    etl::fast_vector<Z, 5> a = {-1.0, 2.0, 0.0, 3.0, -4};

    etl::fast_vector<Z, 5> d(etl::one_if_max(a));

    REQUIRE(d[0] == 0.0);
    REQUIRE(d[1] == 0.0);
    REQUIRE(d[2] == 0.0);
    REQUIRE(d[3] == 1.0);
    REQUIRE(d[4] == 0.0);
}

constexpr bool binary(double a){
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2( "fast_vector/bernoulli", "fast_vector::bernoulli", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 0.3, 0.7};

    etl::fast_vector<Z, 3> d(etl::bernoulli(a));

    REQUIRE(binary(d[0]));
    REQUIRE(binary(d[1]));
    REQUIRE(binary(d[2]));
}

//}}} Unary operators test

//{{{ Reductions

TEMPLATE_TEST_CASE_2( "fast_vector/sum", "sum", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE(d == 9.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sum_2", "sum", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE(d == 19);
}

TEMPLATE_TEST_CASE_2( "fast_vector/sum_3", "sum", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE(d == 23.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/min_reduc_1", "min", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = min(a);

    REQUIRE(d == -1.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/max_reduc_1", "max", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = max(a);

    REQUIRE(d == 8.5);
}

TEMPLATE_TEST_CASE_2( "fast_vector/min_reduc_2", "min", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto& d = min(a);

    REQUIRE(d == -1.0);

    d = 3.3;

    REQUIRE(d == Z(3.3));
}

TEMPLATE_TEST_CASE_2( "fast_vector/max_reduc_2", "max", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto& d = max(a);

    REQUIRE(d == 8.5);

    d = 3.2;

    REQUIRE(d == Z(3.2));
}

//}}} Reductions

//{{{ Complex tests

TEMPLATE_TEST_CASE_2( "fast_vector/complex", "fast_vector::complex", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<Z, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<Z, 3> d(2.5 * ((a >> b) / (a + c)) / ((1.5 * a >> b) / c));

    REQUIRE(d[0] == Approx(10.0));
    REQUIRE(d[1] == Approx(5.0));
    REQUIRE(d[2] == Approx(0.68627));
}

TEMPLATE_TEST_CASE_2( "fast_vector/complex_2", "fast_vector::complex", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.1, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, -3.0, 4.0};
    etl::fast_vector<Z, 3> c = {2.2, 3.0, 3.5};

    etl::fast_vector<Z, 3> d(2.5 * ((a >> b) / (log(a) >> abs(c))) / ((1.5 * a >> sign(b)) / c) + 2.111 / log(c));

    REQUIRE(d[0] == Approx(46.39429));
    REQUIRE(d[1] == Approx(9.13499));
    REQUIRE(d[2] == Approx(5.8273));
}

TEMPLATE_TEST_CASE_2( "fast_vector/complex_3", "fast_vector::complex", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> d(2.5 / (a >> b));

    REQUIRE(d[0] == Approx(-1.0));
    REQUIRE(d[1] == Approx(0.416666));
    REQUIRE(d[2] == Approx(0.125));
}

//}}} Complex tests

TEMPLATE_TEST_CASE_2( "fast_vector/swap_1", "fast_vector::swap", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    using std::swap;
    swap(a, b);

    REQUIRE(a[0] == 2.5);
    REQUIRE(a[1] == 3.0);
    REQUIRE(a[2] == 4.0);

    REQUIRE(b[0] == -1.0);
    REQUIRE(b[1] == 2.0);
    REQUIRE(b[2] == 5.0);
}

//{{{ dot

TEMPLATE_TEST_CASE_2( "fast_vector/dot_1", "sum", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<Z, 3> b = {2.0, 3.0, 2.0};

    auto d = dot(a, b);

    REQUIRE(d == 21.0);
}

TEMPLATE_TEST_CASE_2( "fast_vector/dot_2", "sum", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<Z, 3> b = {2.0, 3.0, 2.0};

    auto d = dot(a, -1 * b);

    REQUIRE(d == -21.0);
}

//}}}
