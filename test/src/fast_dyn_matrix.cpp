//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// Init tests

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/init_1", "fast_dyn_matrix::fast_dyn_matrix(T)", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> test_matrix(3.3);

    REQUIRE_EQUALS(test_matrix.size(), 4UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/init_2", "fast_dyn_matrix::operator=(T)", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> test_matrix;

    test_matrix = 3.3;

    REQUIRE_EQUALS(test_matrix.size(), 4UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/init_3", "fast_dyn_matrix::fast_dyn_matrix(initializer_list)", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> test_matrix = {1.0, 3.0, 5.0, 2.0};

    REQUIRE_EQUALS(test_matrix.size(), 4UL);

    REQUIRE_EQUALS(test_matrix[0], 1.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 5.0);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/access", "fast_dyn_matrix::operator()", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE_EQUALS(test_matrix(0, 0, 0), 1.0);
    REQUIRE_EQUALS(test_matrix(0, 0, 1), -2.0);
    REQUIRE_EQUALS(test_matrix(0, 1, 0), 3.0);
    REQUIRE_EQUALS(test_matrix(0, 1, 1), 0.5);
    REQUIRE_EQUALS(test_matrix(0, 2, 0), 0.0);
    REQUIRE_EQUALS(test_matrix(0, 2, 1), -1);

    REQUIRE_EQUALS(test_matrix(1, 0, 0), 1.0);
    REQUIRE_EQUALS(test_matrix(1, 0, 1), -2.0);
    REQUIRE_EQUALS(test_matrix(1, 1, 0), 3.0);
    REQUIRE_EQUALS(test_matrix(1, 1, 1), 0.5);
    REQUIRE_EQUALS(test_matrix(1, 2, 0), 0.0);
    REQUIRE_EQUALS(test_matrix(1, 2, 1), -1);
}

// Binary operators test

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/add_scalar_1", "fast_dyn_matrix::operator+", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 + test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/add_scalar_2", "fast_dyn_matrix::operator+", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix + 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/add_scalar_4", "fast_dyn_matrix::operator+=", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0, 1.0, 1.0, 1.0, 1.0};

    test_matrix += 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
    REQUIRE_EQUALS(test_matrix[7], 2.0);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/add_1", "fast_dyn_matrix::operator+", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/add_2", "fast_dyn_matrix::operator+=", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
}

// Unary operator tests

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/log_1", "fast_dyn_matrix::log", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/log_2", "fast_dyn_matrix::log", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2, 1> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/unary_unary", "fast_dyn_matrix::abs", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_dyn_matrix<Z, 2, 2> d;
    d = abs(sign(a));

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/unary_binary_1", "fast_dyn_matrix::abs", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/min", "fast_dyn_matrix::min", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2> d;
    d = min(a, 1.0);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
    REQUIRE_EQUALS(d[3], 1.0);
}

// Complex tests

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/complex", "fast_dyn_matrix::complex", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_dyn_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_dyn_matrix<Z, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_dyn_matrix<Z, 2, 2> d;
    d = 2.5 * ((a >> b) / (a + c)) / (1.5 * scale(a, b) / c);

    REQUIRE_EQUALS_APPROX(d[0], 10.0);
    REQUIRE_EQUALS_APPROX(d[1], 5.0);
    REQUIRE_EQUALS_APPROX(d[2], 0.68627);
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/special_1", "", Z, float, double) {
    etl::fast_dyn_matrix<Z, 2, 2> a(3.3);
    etl::fast_matrix<Z, 2, 2> b(4.4);

    a = b;

    REQUIRE_EQUALS(a.size(), 4UL);

    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE_EQUALS(a[i], Z(4.4));
    }

    a = 3.3;

    b = a;

    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE_EQUALS(b[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_dyn_matrix/special_2", "", Z, float, double) {
    etl::fast_dyn_matrix<Z, 4, 3> a(3.3);
    etl::fast_matrix<Z, 2, 6> b(4.4);

    a = b;

    REQUIRE_EQUALS(a.size(), 12UL);

    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE_EQUALS(a[i], Z(4.4));
    }

    a = 3.3;

    b = a;

    for (size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS(b[i], Z(3.3));
    }
}
