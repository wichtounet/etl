//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <vector>
#include <list>
#include <deque>

// Init tests

TEMPLATE_TEST_CASE_2("fast_vector/init_1", "fast_vector::fast_vector(T)", Z, float, double) {
    etl::fast_vector<Z, 4> test_vector(3.3);

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_vector/init_2", "fast_vector::operator=(T)", Z, float, double) {
    etl::fast_vector<Z, 4> test_vector;

    test_vector = 3.3;

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_vector/init_3", "fast_vector::fast_vector(initializer_list)", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {1.0, 2.0, 3.0};

    REQUIRE_EQUALS(test_vector.size(), 3UL);

    REQUIRE_EQUALS(test_vector[0], 1.0);
    REQUIRE_EQUALS(test_vector[1], 2.0);
    REQUIRE_EQUALS(test_vector[2], 3.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/init_4", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    std::vector<Z> vec{1.0, 2.0, 3.0};
    std::list<Z> list{1.0, 2.0, 3.0};
    std::deque<Z> deq{1.0, 2.0, 3.0};

    etl::fast_vector<Z, 3> a(vec);
    etl::fast_vector<Z, 3> b(list);
    etl::fast_vector<Z, 3> c(deq);

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

TEMPLATE_TEST_CASE_2("fast_vector/init_5", "dyn_vector::dyn_vector(initializer_list)", Z, double, float) {
    etl::fast_vector<Z,3> a(3);
    etl::fast_vector<Z,3> b(3);
    etl::fast_vector<Z,3> c(3);

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

TEMPLATE_TEST_CASE_2("fast_vector/add_scalar_1", "fast_vector::operator+", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 + test_vector;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/add_scalar_2", "fast_vector::operator+", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector + 1.0;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/add_scalar_3", "fast_vector::operator+=", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector += 1.0;

    REQUIRE_EQUALS(test_vector[0], 0.0);
    REQUIRE_EQUALS(test_vector[1], 3.0);
    REQUIRE_EQUALS(test_vector[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/add_1", "fast_vector::operator+", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/add_2", "fast_vector::operator+=", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/sub_scalar_1", "fast_vector::operator+", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = 1.0 - test_vector;

    REQUIRE_EQUALS(test_vector[0], 2.0);
    REQUIRE_EQUALS(test_vector[1], -1.0);
    REQUIRE_EQUALS(test_vector[2], -4.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/sub_scalar_2", "fast_vector::operator+", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector = test_vector - 1.0;

    REQUIRE_EQUALS(test_vector[0], -2.0);
    REQUIRE_EQUALS(test_vector[1], 1.0);
    REQUIRE_EQUALS(test_vector[2], 4.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/sub_scalar_3", "fast_vector::operator+=", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.5};

    test_vector -= 1.0;

    REQUIRE_EQUALS(test_vector[0], -2.0);
    REQUIRE_EQUALS(test_vector[1], 1.0);
    REQUIRE_EQUALS(test_vector[2], 4.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/sub_1", "fast_vector::operator-", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c;
    c = a - b;

    REQUIRE_EQUALS(c[0], -3.5);
    REQUIRE_EQUALS(c[1], -1.0);
    REQUIRE_EQUALS(c[2], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/sub_2", "fast_vector::operator-=", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a -= b;

    REQUIRE_EQUALS(a[0], -3.5);
    REQUIRE_EQUALS(a[1], -1.0);
    REQUIRE_EQUALS(a[2], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_scalar_1", "fast_vector::operator*", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 * test_vector;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_scalar_2", "fast_vector::operator*", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector * 2.5;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_scalar_3", "fast_vector::operator*=", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector *= 2.5;

    REQUIRE_EQUALS(test_vector[0], -2.5);
    REQUIRE_EQUALS(test_vector[1], 5.0);
    REQUIRE_EQUALS(test_vector[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_1", "fast_vector::operator*", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c;
    c = a >> b;

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_2", "fast_vector::operator*=", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a *= b;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 6.0);
    REQUIRE_EQUALS(a[2], 20.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/mul_3", "fast_vector::operator*", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c;
    c = a >> b;

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/div_scalar_1", "fast_vector::operator/", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = test_vector / 2.5;

    REQUIRE_EQUALS_APPROX(test_vector[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_vector[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_vector[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/div_scalar_2", "fast_vector::operator/", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector = 2.5 / test_vector;

    REQUIRE_EQUALS_APPROX(test_vector[0], 2.5 / -1.0);
    REQUIRE_EQUALS_APPROX(test_vector[1], 2.5 / 2.0);
    REQUIRE_EQUALS_APPROX(test_vector[2], 2.5 / 5.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/div_scalar_3", "fast_vector::operator/=", Z, float, double) {
    etl::fast_vector<Z, 3> test_vector = {-1.0, 2.0, 5.0};

    test_vector /= 2.5;

    REQUIRE_EQUALS_APPROX(test_vector[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_vector[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_vector[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/div_1", "fast_vector::operator/", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> c;
    c = a / b;

    REQUIRE_EQUALS_APPROX(c[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0 / 4.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/div_2", "fast_vector::operator/=", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a /= b;

    REQUIRE_EQUALS_APPROX(a[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(a[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(a[2], 5.0 / 4.0);
}

ETL_TEST_CASE("fast_vector/mod_scalar_1", "fast_vector::operator%") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector % 2;

    REQUIRE_EQUALS(test_vector[0], -1 % 2);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 5 % 2);
}

ETL_TEST_CASE("fast_vector/mod_scalar_2", "fast_vector::operator%") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector = 2 % test_vector;

    REQUIRE_EQUALS(test_vector[0], 2 % -1);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 2 % 5);
}

ETL_TEST_CASE("fast_vector/mod_scalar_3", "fast_vector::operator%=") {
    etl::fast_vector<int, 3> test_vector = {-1, 2, 5};

    test_vector %= 2;

    REQUIRE_EQUALS(test_vector[0], -1 % 2);
    REQUIRE_EQUALS(test_vector[1], 2 % 2);
    REQUIRE_EQUALS(test_vector[2], 5 % 2);
}

ETL_TEST_CASE("fast_vector/mod_1", "fast_vector::operator%") {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    etl::fast_vector<int, 3> c;
    c = a % b;

    REQUIRE_EQUALS(c[0], -1 % 2);
    REQUIRE_EQUALS(c[1], 2 % 3);
    REQUIRE_EQUALS(c[2], 5 % 4);
}

ETL_TEST_CASE("fast_vector/mod_2", "fast_vector::operator%") {
    etl::fast_vector<int, 3> a = {-1, 2, 5};
    etl::fast_vector<int, 3> b = {2, 3, 4};

    a %= b;

    REQUIRE_EQUALS(a[0], -1 % 2);
    REQUIRE_EQUALS(a[1], 2 % 3);
    REQUIRE_EQUALS(a[2], 5 % 4);
}

// Unary operator tests

TEMPLATE_TEST_CASE_2("fast_vector/log", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};

    etl::fast_vector<Z, 3> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("fast_vector/abs", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = abs(a);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/sign", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = sign(a);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/unary_unary", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = abs(sign(a));

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/unary_binary_1", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/unary_binary_2", "fast_vector::abs", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = abs(a) + a;

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/sigmoid", "fast_vector::sigmoid", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::sigmoid(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::logistic_sigmoid(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("fast_sigmoid/1", "[sigmoid]", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::fast_sigmoid(a);

    REQUIRE_EQUALS_APPROX_E(d[0], etl::math::logistic_sigmoid(Z(-1.0)), base_eps * 10000);
    REQUIRE_EQUALS_APPROX_E(d[1], etl::math::logistic_sigmoid(Z(2.0)), base_eps * 10000);
    REQUIRE_EQUALS_APPROX_E(d[2], etl::math::logistic_sigmoid(Z(0.0)), base_eps * 10000);
}

TEMPLATE_TEST_CASE_2("hard_sigmoid/1", "[sigmoid]", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::hard_sigmoid(a);

    REQUIRE_EQUALS_APPROX_E(d[0], etl::math::logistic_sigmoid(Z(-1.0)), base_eps * 10000);
    REQUIRE_EQUALS_APPROX_E(d[1], etl::math::logistic_sigmoid(Z(2.0)), base_eps * 10000);
    REQUIRE_EQUALS_APPROX_E(d[2], etl::math::logistic_sigmoid(Z(0.0)), base_eps * 10000);
}

TEMPLATE_TEST_CASE_2("fast_vector/softplus", "fast_vector::softplus", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::softplus(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::softplus(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::softplus(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::softplus(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("fast_vector/exp", "fast_vector::exp", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
}

TEMPLATE_TEST_CASE_2("max/1", "[fast][max]", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::max(a, 1.0);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 1.0);
}

TEMPLATE_TEST_CASE_2("max/2", "[fast][max]", Z, float, double) {
    etl::fast_vector<Z, 4> a = {-1.0, -3.0, 0.0, 3.4};
    etl::fast_vector<Z, 4> b = {1.0, -2.0, 5.0, 3.5};

    etl::fast_vector<Z, 4> d;
    d = etl::max(a, b);

    REQUIRE_EQUALS(d[0], Z(1.0));
    REQUIRE_EQUALS(d[1], Z(-2.0));
    REQUIRE_EQUALS(d[2], Z(5.0));
    REQUIRE_EQUALS(d[3], Z(3.5));
}

TEMPLATE_TEST_CASE_2("min/1", "[fast][vector][min]", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::min(a, 1.0);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("min/2", "[fast][vector][min]", Z, float, double) {
    etl::fast_vector<Z, 4> a = {-1.0, -3.0, 0.0, 3.4};
    etl::fast_vector<Z, 4> b = {1.0, -2.0, 5.0, 3.5};

    etl::fast_vector<Z, 4> d;
    d = etl::min(a, b);

    REQUIRE_EQUALS(d[0], Z(-1.0));
    REQUIRE_EQUALS(d[1], Z(-3.0));
    REQUIRE_EQUALS(d[2], Z(0.0));
    REQUIRE_EQUALS(d[3], Z(3.4));
}

TEMPLATE_TEST_CASE_2("clip/1", "[fast][vector][clip]", Z, float, double) {
    etl::fast_vector<Z, 5> a = {-1.0, 0.3, 0.0, 0.5, 1.65};

    etl::fast_vector<Z, 5> d;
    d = etl::clip(a, 0.0, 1.0);

    REQUIRE_EQUALS(d[0], Z(0.0));
    REQUIRE_EQUALS(d[1], Z(0.3));
    REQUIRE_EQUALS(d[2], Z(0.0));
    REQUIRE_EQUALS(d[3], Z(0.5));
    REQUIRE_EQUALS(d[4], Z(1.0));
}
TEMPLATE_TEST_CASE_2("fast_vector/one_if", "fast_vector::one_if", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 0.0};

    etl::fast_vector<Z, 3> d;
    d = etl::one_if(a, 0.0);

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 0.0);
    REQUIRE_EQUALS(d[2], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/one_if_max", "fast_vector::one_if_max", Z, float, double) {
    etl::fast_vector<Z, 5> a = {-1.0, 2.0, 0.0, 3.0, -4};

    etl::fast_vector<Z, 5> d;
    d = etl::one_if_max(a);

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 0.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
    REQUIRE_EQUALS(d[4], 0.0);
}

constexpr bool binary(double a) {
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2("fast_vector/bernoulli", "fast_vector::bernoulli", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 0.3, 0.7};

    etl::fast_vector<Z, 3> d;
    d = etl::bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
}

// Reductions

TEMPLATE_TEST_CASE_2("fast_vector/sum", "sum", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a);

    REQUIRE_EQUALS(d, 9.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/sum_2", "sum", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(a + a);

    REQUIRE_EQUALS(d, 19);
}

TEMPLATE_TEST_CASE_2("fast_vector/sum_3", "sum", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = sum(abs(a + a));

    REQUIRE_EQUALS(d, 23.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/min_reduc_1", "min", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = min(a);

    REQUIRE_EQUALS(d, -1.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/max_reduc_1", "max", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto d = max(a);

    REQUIRE_EQUALS(d, 8.5);
}

TEMPLATE_TEST_CASE_2("fast_vector/min_reduc_2", "min", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto& d = min(a);

    REQUIRE_EQUALS(d, -1.0);

    d = 3.3;

    REQUIRE_EQUALS(d, Z(3.3));
}

TEMPLATE_TEST_CASE_2("fast_vector/max_reduc_2", "max", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 8.5};

    auto& d = max(a);

    REQUIRE_EQUALS(d, 8.5);

    d = 3.2;

    REQUIRE_EQUALS(d, Z(3.2));
}

TEMPLATE_TEST_CASE_2("fast_vector/mean_1", "mean", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.5, 2.5, 8.0};

    auto d = mean(a);

    REQUIRE_EQUALS(d, Z(3.0));
}

TEMPLATE_TEST_CASE_2("fast_vector/mean_2", "mean", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.5, 2.5, 8.0};

    auto d = mean(a + a);

    REQUIRE_EQUALS(d, Z(6.0));
}

TEMPLATE_TEST_CASE_2("fast_vector/stddev_1", "stddev", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.5, 2.5, 8.0};

    auto d = stddev(a);

    REQUIRE_EQUALS_APPROX(d, Z(3.89444));
}

TEMPLATE_TEST_CASE_2("fast_vector/stddev_2", "stddev", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.5, 2.5, 8.0};

    auto d = stddev(a + a);

    REQUIRE_EQUALS_APPROX(d, Z(7.78888));
}

// Complex tests

TEMPLATE_TEST_CASE_2("fast_vector/complex", "fast_vector::complex", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};
    etl::fast_vector<Z, 3> c = {1.2, -3.0, 3.5};

    etl::fast_vector<Z, 3> d;
    d = 2.5 * ((a >> b) / (a + c)) / ((1.5 * a >> b) / c);

    REQUIRE_EQUALS_APPROX(d[0], 10.0);
    REQUIRE_EQUALS_APPROX(d[1], 5.0);
    REQUIRE_EQUALS_APPROX(d[2], 0.68627);
}

TEMPLATE_TEST_CASE_2("fast_vector/complex_2", "fast_vector::complex", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.1, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, -3.0, 4.0};
    etl::fast_vector<Z, 3> c = {2.2, 3.0, 3.5};

    etl::fast_vector<Z, 3> d;
    d = 2.5 * ((a >> b) / (log(a) >> abs(c))) / ((1.5 * a >> sign(b)) / c) + 2.111 / log(c);

    REQUIRE_EQUALS_APPROX(d[0], 46.39429);
    REQUIRE_EQUALS_APPROX(d[1], 9.13499);
    REQUIRE_EQUALS_APPROX(d[2], 5.8273);
}

TEMPLATE_TEST_CASE_2("fast_vector/complex_3", "fast_vector::complex", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    etl::fast_vector<Z, 3> d;
    d = 2.5 / (a >> b);

    REQUIRE_EQUALS_APPROX(d[0], -1.0);
    REQUIRE_EQUALS_APPROX(d[1], 0.416666);
    REQUIRE_EQUALS_APPROX(d[2], 0.125);
}

TEMPLATE_TEST_CASE_2("fast_vector/swap_1", "fast_vector::swap", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 2.0, 5.0};
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    using std::swap;
    swap(a, b);

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);

    REQUIRE_EQUALS(b[0], -1.0);
    REQUIRE_EQUALS(b[1], 2.0);
    REQUIRE_EQUALS(b[2], 5.0);
}

// Column Major

TEMPLATE_TEST_CASE_2("fast_vector/cm/1", "[fast][cm]", Z, float, double) {
    etl::fast_vector_cm<Z, 3> a;
    etl::fast_vector<Z, 3> b = {2.5, 3.0, 4.0};

    a = b;

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/cm/2", "[fast][cm]", Z, float, double) {
    etl::fast_vector<Z, 3> a;
    etl::fast_vector_cm<Z, 3> b = {2.5, 3.0, 4.0};

    a = b;

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/cm/3", "[fast][cm]", Z, float, double) {
    etl::fast_vector_cm<Z, 3> a;
    etl::fast_vector_cm<Z, 3> b = {2.5, 3.0, 4.0};

    a = b;

    REQUIRE_EQUALS(a[0], 2.5);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], 4.0);
}
