//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

// Init tests

TEMPLATE_TEST_CASE_2("dyn_matrix/init_1", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3, 2, 3.3);

    REQUIRE_EQUALS(test_matrix.rows(), 3UL);
    REQUIRE_EQUALS(test_matrix.columns(), 2UL);
    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS_APPROX(test_matrix[i], 3.3);
    }
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_2", "dyn_matrix::operator=(T)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3, 2);

    test_matrix = 3.3;

    REQUIRE_EQUALS(test_matrix.rows(), 3UL);
    REQUIRE_EQUALS(test_matrix.columns(), 2UL);
    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS_APPROX(test_matrix[i], 3.3);
    }
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_3", "dyn_matrix::dyn_matrix(initializer_list)", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(3, 2, std::initializer_list<Z>({1.0, 3.0, 5.0, 2.0, 3.0, 4.0}));

    REQUIRE_EQUALS(test_matrix.rows(), 3UL);
    REQUIRE_EQUALS(test_matrix.columns(), 2UL);
    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    REQUIRE_EQUALS(test_matrix[0], 1.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 5.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_5", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z, 4> a(3, 2, 4, 5);

    a = 3.4;

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 2UL);
    REQUIRE_EQUALS(a.size(), 120UL);
    REQUIRE_EQUALS_APPROX(a(0, 0, 0, 0), 3.4);
    REQUIRE_EQUALS_APPROX(a(1, 1, 1, 1), 3.4);

    etl::dyn_matrix<Z> b(3, 2, std::initializer_list<Z>({1, 2, 3, 4, 5, 6}));

    REQUIRE_EQUALS(b.rows(), 3UL);
    REQUIRE_EQUALS(b.columns(), 2UL);
    REQUIRE_EQUALS(b.size(), 6UL);
    REQUIRE_EQUALS(b[0], 1);
    REQUIRE_EQUALS(b[1], 2);

    etl::dyn_matrix<Z> d(3, 2, Z(3.3));

    REQUIRE_EQUALS(d.rows(), 3UL);
    REQUIRE_EQUALS(d.columns(), 2UL);
    REQUIRE_EQUALS(d.size(), 6UL);
    REQUIRE_EQUALS(d[0], Z(3.3));
    REQUIRE_EQUALS(d[1], Z(3.3));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_6", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> b(3, 2, etl::values(1, 2, 3, 4, 5, 6));

    REQUIRE_EQUALS(b.rows(), 3UL);
    REQUIRE_EQUALS(b.columns(), 2UL);
    REQUIRE_EQUALS(b.size(), 6UL);
    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), 2);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_7", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z, 4> b(2, 2, 2, 2, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));

    REQUIRE_EQUALS(b.rows(), 2UL);
    REQUIRE_EQUALS(b.columns(), 2UL);
    REQUIRE_EQUALS(b.size(), 16UL);
    REQUIRE_EQUALS(b(0, 0, 0, 0), 1);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 2);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 3);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 4);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 5);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 6);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 7);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 8);
    REQUIRE_EQUALS(b(1, 0, 0, 0), 9);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 10);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 11);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 12);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 13);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 14);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 15);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 16);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/init_8", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z, 1> test_matrix(6, Z(3.3));

    REQUIRE_EQUALS(test_matrix.rows(), 6UL);
    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
        REQUIRE_EQUALS(test_matrix(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("dyn_matrix/dim_0", "fast_matrix::fast_matrix(T)", Z, float, double) {
    etl::dyn_matrix<Z, 6> test_matrix(2, 3, 4, 5, 6, 7);

    REQUIRE_EQUALS(test_matrix.template dim<0>(), 2UL);
    REQUIRE_EQUALS(test_matrix.template dim<1>(), 3UL);
    REQUIRE_EQUALS(test_matrix.template dim<2>(), 4UL);
    REQUIRE_EQUALS(test_matrix.template dim<3>(), 5UL);
    REQUIRE_EQUALS(test_matrix.template dim<4>(), 6UL);
    REQUIRE_EQUALS(test_matrix.template dim<5>(), 7UL);
    REQUIRE_EQUALS(test_matrix.dim(0), 2UL);
    REQUIRE_EQUALS(test_matrix.dim(1), 3UL);
    REQUIRE_EQUALS(test_matrix.dim(2), 4UL);
    REQUIRE_EQUALS(test_matrix.dim(3), 5UL);
    REQUIRE_EQUALS(test_matrix.dim(4), 6UL);
    REQUIRE_EQUALS(test_matrix.dim(5), 7UL);
}

// Binary operators test

TEMPLATE_TEST_CASE_2("dyn_matrix/add_scalar_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 + test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/add_scalar_2", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix + 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/add_scalar_3", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix += 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/add_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/add_2", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_scalar_1", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = 1.0 - test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 2.0);
    REQUIRE_EQUALS(test_matrix[1], -1.0);
    REQUIRE_EQUALS(test_matrix[2], -4.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_scalar_2", "dyn_matrix::operator+", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix = test_matrix - 1.0;

    REQUIRE_EQUALS(test_matrix[0], -2.0);
    REQUIRE_EQUALS(test_matrix[1], 1.0);
    REQUIRE_EQUALS(test_matrix[2], 4.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_scalar_3", "dyn_matrix::operator+=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.5, 1.0}));

    test_matrix -= 1.0;

    REQUIRE_EQUALS(test_matrix[0], -2.0);
    REQUIRE_EQUALS(test_matrix[1], 1.0);
    REQUIRE_EQUALS(test_matrix[2], 4.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_1", "dyn_matrix::operator-", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c;
    c = a - b;

    REQUIRE_EQUALS(c[0], -3.5);
    REQUIRE_EQUALS(c[1], -1.0);
    REQUIRE_EQUALS(c[2], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_2", "dyn_matrix::operator-=", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a -= b;

    REQUIRE_EQUALS(a[0], -3.5);
    REQUIRE_EQUALS(a[1], -1.0);
    REQUIRE_EQUALS(a[2], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_scalar_1", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 * test_matrix;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_scalar_2", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix * 2.5;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_scalar_3", "dyn_matrix::operator*=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix *= 2.5;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_1", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c;
    c = scale(a, b);

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_2", "dyn_matrix::operator*=", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a *= b;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 6.0);
    REQUIRE_EQUALS(a[2], 20.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/mul_3", "dyn_matrix::operator*", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c;
    c = a.scale(b);

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/div_scalar_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = test_matrix / 2.5;

    REQUIRE_EQUALS_APPROX(test_matrix[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/div_scalar_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix = 2.5 / test_matrix;

    REQUIRE_EQUALS_APPROX(test_matrix[0], 2.5 / -1.0);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.5 / 2.0);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 2.5 / 5.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/div_scalar_3", "dyn_matrix::operator/=", Z, double, float) {
    etl::dyn_matrix<Z> test_matrix(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    test_matrix /= 2.5;

    REQUIRE_EQUALS_APPROX(test_matrix[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/div_1", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    etl::dyn_matrix<Z> c;
    c = a / b;

    REQUIRE_EQUALS_APPROX(c[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0 / 4.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/div_2", "dyn_matrix::operator/", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));

    a /= b;

    REQUIRE_EQUALS_APPROX(a[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(a[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(a[2], 5.0 / 4.0);
}

ETL_TEST_CASE("dyn_matrix/mod_scalar_1", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> test_matrix(2, 2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = test_matrix % 2;

    REQUIRE_EQUALS(test_matrix[0], -1 % 2);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 5 % 2);
}

ETL_TEST_CASE("dyn_matrix/mod_scalar_2", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> test_matrix(2, 2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix = 2 % test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 2 % -1);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 2 % 5);
}

ETL_TEST_CASE("dyn_matrix/mod_scalar_3", "dyn_matrix::operator%=") {
    etl::dyn_matrix<int> test_matrix(2, 2, std::initializer_list<int>({-1, 2, 5, 1}));

    test_matrix %= 2;

    REQUIRE_EQUALS(test_matrix[0], -1 % 2);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 5 % 2);
}

ETL_TEST_CASE("dyn_matrix/mod_1", "dyn_matrix::operator%") {
    etl::dyn_matrix<int> a(2, 2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2, 2, std::initializer_list<int>({2, 3, 4, 1}));

    etl::dyn_matrix<int> c;
    c = a % b;

    REQUIRE_EQUALS(c[0], -1 % 2);
    REQUIRE_EQUALS(c[1], 2 % 3);
    REQUIRE_EQUALS(c[2], 5 % 4);
}

ETL_TEST_CASE("dyn_matrix/mod_2", "dyn_matrix::operator%=") {
    etl::dyn_matrix<int> a(2, 2, std::initializer_list<int>({-1, 2, 5, 1}));
    etl::dyn_matrix<int> b(2, 2, std::initializer_list<int>({2, 3, 4, 1}));

    a %= b;

    REQUIRE_EQUALS(a[0], -1 % 2);
    REQUIRE_EQUALS(a[1], 2 % 3);
    REQUIRE_EQUALS(a[2], 5 % 4);
}

// Unary operator tests

TEMPLATE_TEST_CASE_2("dyn_matrix/log", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(2.0));
    REQUIRE_EQUALS_APPROX(d[2], std::log(5.0));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/abs", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = abs(a);

    REQUIRE_EQUALS(d[0], Z(1.0));
    REQUIRE_EQUALS(d[1], Z(2.0));
    REQUIRE_EQUALS(d[2], Z(0.0));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sign", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = sign(a);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/unary_binary_1", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/unary_binary_2", "dyn_matrix::abs", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = abs(a) + a;

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sigmoid", "dyn_matrix::sigmoid", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::sigmoid(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::logistic_sigmoid(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::logistic_sigmoid(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/softplus", "dyn_matrix::softplus", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::softplus(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::softplus(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::softplus(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::softplus(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::softplus(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/exp", "dyn_matrix::exp", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::exp(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/max", "dyn_matrix::max", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::max(a, 1.0);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 1.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/min", "dyn_matrix::min", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::min(a, 1.0);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

constexpr bool binary(double a) {
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2("dyn_matrix/bernoulli", "dyn_matrix::bernoulli", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 0.0, 1.0}));

    etl::dyn_matrix<Z> d;
    d = etl::bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}

// Complex tests

TEMPLATE_TEST_CASE_2("dyn_matrix/complex", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2, 2, std::initializer_list<Z>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d;
    d = 2.5 * ((scale(a, b)) / (a + c)) / (1.5 * (a >> b) / c);

    REQUIRE_EQUALS_APPROX(d[0], 10.0);
    REQUIRE_EQUALS_APPROX(d[1], 5.0);
    REQUIRE_EQUALS_APPROX(d[2], 0.68627);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/complex_2", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({1.1, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, -3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2, 2, std::initializer_list<Z>({2.2, 3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d;
    d = 2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c);

    REQUIRE_EQUALS_APPROX(d[0], 46.39429);
    REQUIRE_EQUALS_APPROX(d[1], 9.13499);
    REQUIRE_EQUALS_APPROX(d[2], 5.8273);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/complex_3", "dyn_matrix::complex", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));
    etl::dyn_matrix<Z> b(2, 2, std::initializer_list<Z>({2.5, 3.0, 4.0, 1.0}));
    etl::dyn_matrix<Z> c(2, 2, std::initializer_list<Z>({1.2, -3.0, 3.5, 1.0}));

    etl::dyn_matrix<Z> d;
    d = 2.5 / (a >> b);

    REQUIRE_EQUALS_APPROX(d[0], -1.0);
    REQUIRE_EQUALS_APPROX(d[1], 0.416666);
    REQUIRE_EQUALS_APPROX(d[2], 0.125);
}

// Reductions

TEMPLATE_TEST_CASE_2("dyn_matrix/sum", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(a);

    REQUIRE_EQUALS(d, 9.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sum_2", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(a + a);

    REQUIRE_EQUALS(d, 19);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sum_3", "sum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = sum(abs(a + a));

    REQUIRE_EQUALS(d, 23.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sum/4", "sum", Z, int, unsigned int) {
    etl::dyn_matrix<Z> a(4, 1, std::initializer_list<Z>({1, 2, 8, 4}));

    auto d = sum(a + a);
    REQUIRE_EQUALS(d, Z(30));

    d = sum(a);
    REQUIRE_EQUALS(d, Z(15));

    d = sum(a(1));
    REQUIRE_EQUALS(d, Z(2));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/asum/1", "asum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, -2.0, 8.5}));

    auto d = asum(a);

    REQUIRE_EQUALS(d, 11.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/asum/2", "asum", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, -2.0, -8.5}));

    auto d = asum(a + a);

    REQUIRE_EQUALS(d, 23.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/min_reduc", "min", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = min(a);

    REQUIRE_EQUALS(d, -1.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/max_reduc", "max", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 1, std::initializer_list<Z>({-1.0, 2.0, 8.5}));

    auto d = max(a);

    REQUIRE_EQUALS(d, 8.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/norm_1", "[dyn][reduc][norm]", Z, double, float) {
    etl::dyn_matrix<Z> a{2, 2, std::initializer_list<Z>({-5.0, -7.0, -2.0, 8.0})};

    auto d = norm(a);

    REQUIRE_EQUALS_APPROX(d, 11.916375);
}

// is_finite tests

TEMPLATE_TEST_CASE_2("dyn_matrix/is_finite_1", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 2.0, 5.0, 1.0}));

    REQUIRE_DIRECT(a.is_finite());
}

TEMPLATE_TEST_CASE_2("dyn_matrix/is_finite_2", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, NAN, 5.0, 1.0}));

    REQUIRE_DIRECT(!a.is_finite());
}

TEMPLATE_TEST_CASE_2("dyn_matrix/is_finite_3", "dyn_matrix::is_finite", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 2, std::initializer_list<Z>({-1.0, 1.0, INFINITY, 1.0}));

    REQUIRE_DIRECT(!a.is_finite());
}

// scale tests

TEMPLATE_TEST_CASE_2("dyn_matrix/scale_1", "", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));

    a *= 2.5;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 12.5);
    REQUIRE_EQUALS(a[3], 2.5);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/scale_2", "", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.5, 2.0, 3.0, -1.2));

    a *= b;

    REQUIRE_EQUALS(a[0], Z(-2.5));
    REQUIRE_EQUALS(a[1], Z(4.0));
    REQUIRE_EQUALS(a[2], Z(15.0));
    REQUIRE_EQUALS(a[3], Z(-1.2));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/scale_3", "", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));
    etl::dyn_matrix<Z> b(2, 2, etl::values(2.5, 2.0, 3.0, -1.2));

    a.scale_inplace(b);

    REQUIRE_EQUALS(a[0], Z(-2.5));
    REQUIRE_EQUALS(a[1], Z(4.0));
    REQUIRE_EQUALS(a[2], Z(15.0));
    REQUIRE_EQUALS(a[3], Z(-1.2));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/scale_4", "", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 2, etl::values(-1.0, 2.0, 5.0, 1.0));

    a.scale_inplace(2.5);

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 12.5);
    REQUIRE_EQUALS(a[3], 2.5);
}

// swap tests

TEMPLATE_TEST_CASE_2("dyn_matrix/swap_1", "", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<Z> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    etl::swap(a, b);

    REQUIRE_EQUALS(a[0], Z(1.0));
    REQUIRE_EQUALS(a[1], Z(3.3));
    REQUIRE_EQUALS(a[2], Z(4.4));
    REQUIRE_EQUALS(a[3], Z(9.0));
    REQUIRE_EQUALS(a[4], Z(10.1));
    REQUIRE_EQUALS(a[5], Z(-1.1));

    REQUIRE_EQUALS(b[0], Z(-1.0));
    REQUIRE_EQUALS(b[1], Z(2.0));
    REQUIRE_EQUALS(b[2], Z(5.0));
    REQUIRE_EQUALS(b[3], Z(1.0));
    REQUIRE_EQUALS(b[4], Z(1.1));
    REQUIRE_EQUALS(b[5], Z(1.9));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/swap_2", "", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<Z> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    a.swap(b);

    REQUIRE_EQUALS(a[0], Z(1.0));
    REQUIRE_EQUALS(a[1], Z(3.3));
    REQUIRE_EQUALS(a[2], Z(4.4));
    REQUIRE_EQUALS(a[3], Z(9.0));
    REQUIRE_EQUALS(a[4], Z(10.1));
    REQUIRE_EQUALS(a[5], Z(-1.1));

    REQUIRE_EQUALS(b[0], Z(-1.0));
    REQUIRE_EQUALS(b[1], Z(2.0));
    REQUIRE_EQUALS(b[2], Z(5.0));
    REQUIRE_EQUALS(b[3], Z(1.0));
    REQUIRE_EQUALS(b[4], Z(1.1));
    REQUIRE_EQUALS(b[5], Z(1.9));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/swap_3", "", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 4, Z(1));
    etl::dyn_matrix<Z> b(5, 6, Z(2));

    a.swap(b);

    REQUIRE_EQUALS(etl::size(a), 30UL);
    REQUIRE_EQUALS(etl::size(b), 8UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 6UL);
    REQUIRE_EQUALS(etl::dim<0>(b), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(b), 4UL);
}

//Make sure assign between matrices of different are compiling correctly

ETL_TEST_CASE("dyn_matrix/assign_two_types", "") {
    etl::dyn_matrix<double> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<float> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    //This must compile
    a = b;
    b = a;

    etl::dyn_matrix<double> aaa;
    aaa = b;
    etl::dyn_matrix<float> bbb;
    bbb = a;
}

//Make sure dyn matrix can inherit from expression

ETL_TEST_CASE("dyn_matrix/inherit", "") {
    etl::dyn_matrix<double, 3> a;

    etl::dyn_matrix<double, 3> b(3, 4, 5);

    a = b + b;

    REQUIRE_EQUALS(a.size(), b.size());
    REQUIRE_EQUALS(etl::dim<0>(a), etl::dim<0>(b));
    REQUIRE_EQUALS(etl::dim<1>(a), etl::dim<1>(b));
    REQUIRE_EQUALS(etl::dim<2>(a), etl::dim<2>(b));

    REQUIRE_EQUALS(etl::dim<0>(a), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 4UL);
    REQUIRE_EQUALS(etl::dim<2>(a), 5UL);
}

//Make sure default construction is possible and then size is modifiable

ETL_TEST_CASE("dyn_matrix/default_constructor_1", "") {
    etl::dyn_matrix<double> def_a;
    etl::dyn_matrix<float> def_b;

    etl::dyn_matrix<double> a(3, 2, etl::values(-1.0, 2.0, 5.0, 1.0, 1.1, 1.9));
    etl::dyn_matrix<float> b(3, 2, etl::values(1.0, 3.3, 4.4, 9, 10.1, -1.1));

    def_a = a;
    def_b = b;

    REQUIRE_EQUALS(def_a.size(), a.size());
    REQUIRE_EQUALS(def_b.size(), b.size());

    REQUIRE_EQUALS(etl::dim<0>(def_a), etl::dim<0>(a));
    REQUIRE_EQUALS(etl::dim<1>(def_a), etl::dim<1>(a));

    REQUIRE_EQUALS(etl::dim<0>(def_b), etl::dim<0>(b));
    REQUIRE_EQUALS(etl::dim<1>(def_b), etl::dim<1>(b));

    REQUIRE_EQUALS(def_a(1, 1), 1.0);
    REQUIRE_EQUALS(def_b(1, 1), 9.0);
}

ETL_TEST_CASE("dyn_matrix/resize/1", "[dyn][resize]") {
    etl::dyn_matrix<float> a(10, 2);

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);

    a.resize(4, 4);

    REQUIRE_EQUALS(a.size(), 16UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 4UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 4UL);

    for (size_t i = 0; i < 16; ++i) {
        REQUIRE_EQUALS(a[i], i * 5.0);
    }
}

ETL_TEST_CASE("dyn_matrix/resize/2", "[dyn][resize]") {
    etl::dyn_matrix<float> a;

    a.resize(10, 2);

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);
}

ETL_TEST_CASE("dyn_matrix/resize/3", "[dyn][resize]") {
    etl::dyn_matrix<float> a(10, 2);

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);

    a.resize(5, 5);

    REQUIRE_EQUALS(a.size(), 25UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 5UL);

    for (size_t i = 0; i < 20; ++i) {
        REQUIRE_EQUALS(a[i], i * 5.0);
    }
}

ETL_TEST_CASE("dyn_matrix/resize_array/1", "[dyn][resize]") {
    etl::dyn_matrix<float> a(10, 2);

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);

    std::array<size_t, 2> dims{{4, 4}};
    a.resize_arr(dims);

    REQUIRE_EQUALS(a.size(), 16UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 4UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 4UL);

    for (size_t i = 0; i < 16; ++i) {
        REQUIRE_EQUALS(a[i], i * 5.0);
    }
}

ETL_TEST_CASE("dyn_matrix/resize_array/2", "[dyn][resize]") {
    etl::dyn_matrix<float> a;

    a.resize_arr(std::array<size_t, 2>{{10, 2}});

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);
}

ETL_TEST_CASE("dyn_matrix/resize_array/3", "[dyn][resize]") {
    etl::dyn_matrix<float> a(10, 2);

    for (size_t i = 0; i < 20; ++i) {
        a[i] = i * 5.0;
    }

    REQUIRE_EQUALS(a.size(), 20UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 10UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);

    a.resize_arr(std::array<size_t, 2>{{5, 5}});

    REQUIRE_EQUALS(a.size(), 25UL);
    REQUIRE_EQUALS(etl::dim<0>(a), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 5UL);

    for (size_t i = 0; i < 20; ++i) {
        REQUIRE_EQUALS(a[i], i * 5.0);
    }
}

ETL_TEST_CASE("dyn_matrix/default_constructor_2", "") {
    std::vector<etl::dyn_matrix<double>> values(10);

    REQUIRE_EQUALS(values[0].size(), 0UL);

    values[0] = etl::dyn_matrix<double>(3, 2);

    REQUIRE_EQUALS(values[0].size(), 6UL);
    REQUIRE_EQUALS(etl::dim<0>(values[0]), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(values[0]), 2UL);
}

etl::dyn_matrix<double, 3> test_return() {
    return etl::dyn_matrix<double, 3>(3, 8, 1);
}

ETL_TEST_CASE("dyn_matrix/default_constructor_3", "") {
    std::vector<etl::dyn_matrix<double, 3>> values;

    values.emplace_back();

    REQUIRE_EQUALS(values[0].size(), 0UL);

    values.emplace_back(5, 5, 1, 1.0);

    REQUIRE_EQUALS(values[0].size(), 0UL);
    REQUIRE_EQUALS(values[1].size(), 25UL);
    REQUIRE_EQUALS(values[1][0], 1.0);

    values.push_back(etl::dyn_matrix<double, 3>(3, 2, 5, 13.0));

    REQUIRE_EQUALS(values[0].size(), 0UL);
    REQUIRE_EQUALS(values[1].size(), 25UL);
    REQUIRE_EQUALS(values[1][0], 1.0);
    REQUIRE_EQUALS(values[2].size(), 30UL);
    REQUIRE_EQUALS(values[2][0], 13.0);

    values.shrink_to_fit();
    values.push_back(test_return());

    REQUIRE_EQUALS(values[0].size(), 0UL);
    REQUIRE_EQUALS(values[1].size(), 25UL);
    REQUIRE_EQUALS(values[1][0], 1.0);
    REQUIRE_EQUALS(values[2].size(), 30UL);
    REQUIRE_EQUALS(values[2][0], 13.0);
    REQUIRE_EQUALS(values[3].size(), 24UL);

    values.pop_back();
    values.shrink_to_fit();

    REQUIRE_EQUALS(values[0].size(), 0UL);
    REQUIRE_EQUALS(values[1].size(), 25UL);
    REQUIRE_EQUALS(values[1][0], 1.0);
    REQUIRE_EQUALS(values[2].size(), 30UL);
    REQUIRE_EQUALS(values[2][0], 13.0);

    std::vector<etl::dyn_matrix<double, 3>> values_2;
    values_2 = values;

    REQUIRE_EQUALS(values_2[0].size(), 0UL);
    REQUIRE_EQUALS(values_2[1].size(), 25UL);
    REQUIRE_EQUALS(values_2[1][0], 1.0);
    REQUIRE_EQUALS(values_2[2].size(), 30UL);
    REQUIRE_EQUALS(values_2[2][0], 13.0);
}

template <typename T>
bool one_of(T first, T v1, T v2, T v3, T v4, T v5) {
    return first == v1 || first == v2 || first == v3 || first == v4 || first == v5;
}

ETL_TEST_CASE("dyn_matrix/vector_shuffle", "") {
    std::vector<etl::dyn_matrix<double, 3>> values;

    values.emplace_back(5, 5, 1, 1.0);
    values.emplace_back(2, 2, 2, 2.0);
    values.emplace_back(3, 1, 2, 3.0);
    values.emplace_back(10, 10, 10, 4.0);
    values.emplace_back(1, 1, 1, 5.0);

    std::random_device rd;
    std::default_random_engine g(rd());

    std::shuffle(values.begin(), values.end(), g);

    REQUIRE_DIRECT(one_of(values[0](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values[1](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values[2](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values[3](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values[4](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));

    REQUIRE_DIRECT(one_of(etl::size(values[0]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values[1]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values[2]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values[3]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values[4]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
}

ETL_TEST_CASE("dyn_matrix/parallel_vector_shuffle", "") {
    std::vector<etl::dyn_matrix<double, 3>> values_1;
    std::vector<etl::dyn_matrix<double, 3>> values_2;

    values_1.emplace_back(5, 5, 1, 1.0);
    values_1.emplace_back(2, 2, 2, 2.0);
    values_1.emplace_back(3, 1, 2, 3.0);
    values_1.emplace_back(10, 10, 10, 4.0);
    values_1.emplace_back(1, 1, 1, 5.0);

    values_2.emplace_back(50, 5, 1, 10.0);
    values_2.emplace_back(20, 2, 2, 20.0);
    values_2.emplace_back(30, 1, 2, 30.0);
    values_2.emplace_back(100, 10, 10, 40.0);
    values_2.emplace_back(10, 1, 1, 50.0);

    std::random_device rd;
    std::default_random_engine g(rd());

    cpp::parallel_shuffle(values_1.begin(), values_1.end(), values_2.begin(), values_2.end(), g);

    REQUIRE_DIRECT(one_of(values_1[0](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values_1[1](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values_1[2](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values_1[3](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));
    REQUIRE_DIRECT(one_of(values_1[4](0, 0, 0), 1.0, 2.0, 3.0, 4.0, 5.0));

    REQUIRE_DIRECT(one_of(etl::size(values_1[0]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values_1[1]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values_1[2]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values_1[3]), 25UL, 8UL, 6Ul, 1000UL, 1UL));
    REQUIRE_DIRECT(one_of(etl::size(values_1[4]), 25UL, 8UL, 6Ul, 1000UL, 1UL));

    REQUIRE_DIRECT(one_of(values_2[0](0, 0, 0), 10.0, 20.0, 30.0, 40.0, 50.0));
    REQUIRE_DIRECT(one_of(values_2[1](0, 0, 0), 10.0, 20.0, 30.0, 40.0, 50.0));
    REQUIRE_DIRECT(one_of(values_2[2](0, 0, 0), 10.0, 20.0, 30.0, 40.0, 50.0));
    REQUIRE_DIRECT(one_of(values_2[3](0, 0, 0), 10.0, 20.0, 30.0, 40.0, 50.0));
    REQUIRE_DIRECT(one_of(values_2[4](0, 0, 0), 10.0, 20.0, 30.0, 40.0, 50.0));

    REQUIRE_DIRECT(one_of(etl::size(values_2[0]), 250UL, 80UL, 60Ul, 10000UL, 10UL));
    REQUIRE_DIRECT(one_of(etl::size(values_2[1]), 250UL, 80UL, 60Ul, 10000UL, 10UL));
    REQUIRE_DIRECT(one_of(etl::size(values_2[2]), 250UL, 80UL, 60Ul, 10000UL, 10UL));
    REQUIRE_DIRECT(one_of(etl::size(values_2[3]), 250UL, 80UL, 60Ul, 10000UL, 10UL));
    REQUIRE_DIRECT(one_of(etl::size(values_2[4]), 250UL, 80UL, 60Ul, 10000UL, 10UL));
}

ETL_TEST_CASE("dyn_matrix/floor/1", "") {
    etl::dyn_matrix<double, 2> a(2, 2, etl::values(1.1, 1.0, 2.9, -1.3));;
    etl::dyn_matrix<double, 2> b;

    b = floor(a);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 1.0);
    REQUIRE_EQUALS(b[2], 2.0);
    REQUIRE_EQUALS(b[3], -2.0);
}

ETL_TEST_CASE("dyn_matrix/ceil/1", "") {
    etl::dyn_matrix<double, 2> a(2, 2, etl::values(1.1, 1.0, 2.9, -1.3));;
    etl::dyn_matrix<double, 2> b;

    b = ceil(a);

    REQUIRE_EQUALS(b[0], 2.0);
    REQUIRE_EQUALS(b[1], 1.0);
    REQUIRE_EQUALS(b[2], 3.0);
    REQUIRE_EQUALS(b[3], -1.0);
}
