//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include "sum_test.hpp"

#include <vector>
#include <list>
#include <deque>

// Init tests

TEMPLATE_TEST_CASE_2("dyn_vector/init_1", "dyn_vector::dyn_vector(T)", Z, double, float) {
    etl::dyn_vector<Z> test_vector(4, 3.3);

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], Z(3.3));
        REQUIRE_EQUALS(test_vector(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("dyn_vector/init_2", "dyn_vector::operator=(T)", Z, double, float) {
    etl::dyn_vector<Z> test_vector(4);

    test_vector = 3.3;

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], Z(3.3));
        REQUIRE_EQUALS(test_vector(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("dyn_vector/init_3", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    etl::dyn_vector<Z> test_vector({1.0, 2.0, 3.0});

    REQUIRE_EQUALS(test_vector.size(), 3UL);

    REQUIRE_EQUALS(test_vector[0], 1.0);
    REQUIRE_EQUALS(test_vector[1], 2.0);
    REQUIRE_EQUALS(test_vector[2], 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/init_4", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    std::vector<Z> vec{1.0, 2.0, 3.0};
    std::list<Z> list{1.0, 2.0, 3.0};
    std::deque<Z> deq{1.0, 2.0, 3.0};

    etl::dyn_vector<Z> a(vec);
    etl::dyn_vector<Z> b(list);
    etl::dyn_vector<Z> c(deq);

    REQUIRE_EQUALS(a.size(), 3UL);
    REQUIRE_EQUALS(b.size(), 3UL);
    REQUIRE_EQUALS(c.size(), 3UL);

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 2.0);
    REQUIRE_EQUALS(a[2], 3.0);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 2.0);
    REQUIRE_EQUALS(b[2], 3.0);

    REQUIRE_EQUALS(c[0], 1.0);
    REQUIRE_EQUALS(c[1], 2.0);
    REQUIRE_EQUALS(c[2], 3.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/init_5", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    etl::dyn_vector<Z> a(3);
    etl::dyn_vector<Z> b(3);
    etl::dyn_vector<Z> c(3);

    std::vector<Z> vec{1.0, 2.0, 3.0};
    std::list<Z> list{1.0, 2.0, 3.0};
    std::deque<Z> deq{1.0, 2.0, 3.0};

    a = vec;
    b = list;
    c = deq;

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 2.0);
    REQUIRE_EQUALS(a[2], 3.0);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 2.0);
    REQUIRE_EQUALS(b[2], 3.0);

    REQUIRE_EQUALS(c[0], 1.0);
    REQUIRE_EQUALS(c[1], 2.0);
    REQUIRE_EQUALS(c[2], 3.0);
}

// Binary operators test

TEMPLATE_TEST_CASE_2("dyn_vector/add_scalar_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/add_scalar_2", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/add_scalar_3", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/add_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/add_2", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sub_scalar_1", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE_EQUALS(test_vector[0], 2.0);
    REQUIRE_EQUALS(test_vector[1], -1.0);
    REQUIRE_EQUALS(test_vector[2], -4.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sub_scalar_2", "dyn_vector::operator+", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE_EQUALS(test_vector[0], -2.0);
    REQUIRE_EQUALS(test_vector[1], 1.0);
    REQUIRE_EQUALS(test_vector[2], 4.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sub_scalar_3", "dyn_vector::operator+=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE_EQUALS(test_vector[0], -2.0);
    REQUIRE_EQUALS(test_vector[1], 1.0);
    REQUIRE_EQUALS(test_vector[2], 4.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sub_1", "dyn_vector::operator-", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c;
    c = a - b;

    REQUIRE_EQUALS(c[0], -3.5);
    REQUIRE_EQUALS(c[1], -1.0);
    REQUIRE_EQUALS(c[2], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sub_2", "dyn_vector::operator-=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE_EQUALS(a[0], -3.5);
    REQUIRE_EQUALS(a[1], -1.0);
    REQUIRE_EQUALS(a[2], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mul_scalar_1", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mul_scalar_2", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mul_scalar_3", "dyn_vector::operator*=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mul_1", "dyn_vector::operator*", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c;
    c = a >> b;

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mul_2", "dyn_vector::operator*=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 6.0);
    REQUIRE_EQUALS(a[2], 20.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/div_scalar_1", "dyn_vector::operator/", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / Z(2.5);

    REQUIRE_EQUALS(test_vector[0], Z(-1.0 / 2.5));
    REQUIRE_EQUALS(test_vector[1], Z(2.0 / 2.5));
    REQUIRE_EQUALS(test_vector[2], Z(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2("dyn_vector/div_scalar_2", "dyn_vector::operator/", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE_EQUALS(test_vector[0], 2.5 / -1.0);
    REQUIRE_EQUALS(test_vector[1], 2.5 / 2.0);
    REQUIRE_EQUALS(test_vector[2], 2.5 / 5.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/div_scalar_3", "dyn_vector::operator/=", Z, double, float) {
    etl::dyn_vector<Z> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= Z(2.5);

    REQUIRE_EQUALS(test_vector[0], Z(-1.0 / 2.5));
    REQUIRE_EQUALS(test_vector[1], Z(2.0 / 2.5));
    REQUIRE_EQUALS(test_vector[2], Z(5.0 / 2.5));
}

TEMPLATE_TEST_CASE_2("dyn_vector/div_1", "dyn_vector::operator/", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    etl::dyn_vector<Z> c;
    c = a / b;

    REQUIRE_EQUALS(c[0], Z(-1.0 / 2.5));
    REQUIRE_EQUALS(c[1], Z(2.0 / 3.0));
    REQUIRE_EQUALS(c[2], Z(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2("dyn_vector/div_2", "dyn_vector::operator/=", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE_EQUALS(a[0], Z(-1.0 / 2.5));
    REQUIRE_EQUALS(a[1], Z(2.0 / 3.0));
    REQUIRE_EQUALS(a[2], Z(5.0 / 4.0));
}

TEMPLATE_TEST_CASE_2("dyn_vector/mod_scalar_1", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE_EQUALS(test_vector[0], -1 % 2);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 5 % 2);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mod_scalar_2", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE_EQUALS(test_vector[0], 2 % -1);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 2 % 5);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mod_scalar_3", "dyn_vector::operator%=", Z, double, float) {
    etl::dyn_vector<int> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE_EQUALS(test_vector[0], -1 % 2);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 5 % 2);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mod_1", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    etl::dyn_vector<int> c;
    c = a % b;

    REQUIRE_EQUALS(c[0], -1 % 2);
    REQUIRE_EQUALS(c[1], 2 % 3);
    REQUIRE_EQUALS(c[2], 5 % 4);
}

TEMPLATE_TEST_CASE_2("dyn_vector/mod_2", "dyn_vector::operator%", Z, double, float) {
    etl::dyn_vector<int> a = {-1, 2, 5};
    etl::dyn_vector<int> b = {2, 3, 4};

    a %= b;

    REQUIRE_EQUALS(a[0], -1 % 2);
    REQUIRE_EQUALS(a[1], 2 % 3);
    REQUIRE_EQUALS(a[2], 5 % 4);
}

// Unary operator tests

TEMPLATE_TEST_CASE_2("dyn_vector/log", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};

    etl::dyn_vector<Z> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("dyn_vector/abs", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = abs(a);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sign", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = sign(a);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/unary_unary", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = abs(sign(a));

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/unary_binary_1", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/unary_binary_2", "dyn_vector::abs", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = abs(a) + a;

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sigmoid", "dyn_vector::sigmoid", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::sigmoid(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::logistic_sigmoid(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("dyn_vector/softplus", "dyn_vector::softplus", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::softplus(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::softplus(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::softplus(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::softplus(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("dyn_vector/exp/1", "dyn_vector::exp", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("dyn_vector/exp/2", "dyn_vector::exp", Z, double, float) {
    etl::dyn_vector<Z> a(1033);
    a = 0.01 * etl::sequence_generator(1.0);
    etl::dyn_vector<Z> c;
    c = etl::exp(a);

    for(size_t i = 0; i < etl::size(a); ++i){
        REQUIRE_EQUALS_APPROX(c[i], std::exp(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("dyn_vector/max", "dyn_vector::max", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::max(a, 1.0);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/min", "dyn_vector::min", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::min(a, 1.0);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

constexpr bool binary(double a) {
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2("dyn_vector/bernoulli/1", "dyn_vector::bernoulli", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::dyn_vector<Z> d;
    d = etl::bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
}

TEMPLATE_TEST_CASE_2("dyn_vector/bernoulli/2", "dyn_vector::bernoulli", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 0.0};

    etl::random_engine engine(42);

    etl::dyn_vector<Z> d;
    d = etl::bernoulli(engine, a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
}

// Reductions

// TODO Need more sum tests

SUM_TEST_CASE("sum/0", "sum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(9.5));
}

SUM_TEST_CASE("sum/1", "sum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5, 1.0, 3.0, -5.0, 11.4};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(19.9));
}

SUM_TEST_CASE("sum/2", "sum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5, 1.0, 3.0, -5.0, 11.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(64.9));
}

ASUM_TEST_CASE("asum/0", "asum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(11.5));
}

ASUM_TEST_CASE("asum/1", "asum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5, 1.0, 3.0, -5.0, 11.4};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(31.9));
}

ASUM_TEST_CASE("asum/2", "asum") {
    etl::dyn_vector<T> a = {-1.0, 2.0, 8.5, 1.0, 3.0, -5.0, 11.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    T value = 0;
    Impl::apply(a, value);

    REQUIRE_EQUALS(value, T(76.9));
}

TEMPLATE_TEST_CASE_2("dyn_vector/sum_2", "sum", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE_EQUALS(d, 19);
}

TEMPLATE_TEST_CASE_2("dyn_vector/sum_3", "sum", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE_EQUALS(d, 23.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/norm_1", "[dyn][reduc][norm]", Z, double, float) {
    etl::dyn_vector<Z> a{-1.0, 2.0, 8.0};

    auto d = norm(a);

    REQUIRE_EQUALS_APPROX(d, 8.30662);
}

// Complex tests

TEMPLATE_TEST_CASE_2("dyn_vector/complex", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<Z> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<Z> d;
    d = 2.5 * ((a >> b) / (a + c)) / (1.5 * scale(a, b) / c);

    REQUIRE_EQUALS_APPROX(d[0], 10.0);
    REQUIRE_EQUALS_APPROX(d[1], 5.0);
    REQUIRE_EQUALS_APPROX(d[2], 0.68627);
}

TEMPLATE_TEST_CASE_2("dyn_vector/complex_2", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {1.1, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, -3.0, 4.0};
    etl::dyn_vector<Z> c = {2.2, 3.0, 3.5};

    etl::dyn_vector<Z> d;
    d = 2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c);

    REQUIRE_EQUALS_APPROX(d[0], 46.39429);
    REQUIRE_EQUALS_APPROX(d[1], 9.13499);
    REQUIRE_EQUALS_APPROX(d[2], 5.8273);
}

TEMPLATE_TEST_CASE_2("dyn_vector/complex_3", "dyn_vector::complex", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};
    etl::dyn_vector<Z> c = {1.2, -3.0, 3.5};

    etl::dyn_vector<Z> d;
    d = 2.5 / (a >> b);

    REQUIRE_EQUALS_APPROX(d[0], -1.0);
    REQUIRE_EQUALS_APPROX(d[1], 0.416666);
    REQUIRE_EQUALS_APPROX(d[2], 0.125);
}

// Complex content

TEMPLATE_TEST_CASE_2("dyn_vector/complex_content_1", "dyn_vector<dyn_matrix>>", Z, double, float) {
    etl::dyn_vector<etl::dyn_matrix<Z>> a(11, etl::dyn_matrix<Z>(3, 3));

    //It is enough for this test to compile
    REQUIRE_DIRECT(true);
}

TEMPLATE_TEST_CASE_2("dyn_vector/complex_content_2", "vector<dyn_vector<dyn_matrix>>>", Z, double, float) {
    std::vector<etl::dyn_vector<etl::dyn_matrix<Z>>> a;

    a.emplace_back(11, etl::dyn_matrix<Z>(3, 3));

    //It is enough for this test to compile
    REQUIRE_DIRECT(true);
}

TEMPLATE_TEST_CASE_2("dyn_vector/swap_1", "dyn_vector::swap", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0};

    using std::swap;
    swap(a, b);

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);

    REQUIRE_EQUALS(b[0], -1.0);
    REQUIRE_EQUALS(b[1], 2.0);
    REQUIRE_EQUALS(b[2], 5.0);
}

TEMPLATE_TEST_CASE_2("dyn_vector/swap_2", "dyn_vector::swap", Z, double, float) {
    etl::dyn_vector<Z> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<Z> b = {2.5, 3.0, 4.0, 5.0};

    using std::swap;
    swap(a, b);

    REQUIRE_EQUALS(etl::size(a), 4UL);
    REQUIRE_EQUALS(etl::size(b), 3UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 4UL);
    REQUIRE_EQUALS(etl::dim<0>(b), 3UL);

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);
    REQUIRE_EQUALS(a[3], 5.0);

    REQUIRE_EQUALS(b[0], -1.0);
    REQUIRE_EQUALS(b[1], 2.0);
    REQUIRE_EQUALS(b[2], 5.0);
}
