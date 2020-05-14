//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

// Init tests

TEMPLATE_TEST_CASE_2("fast_matrix/init_1", "fast_matrix::fast_matrix(T)", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> test_matrix(3.3);

    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    REQUIRE_EQUALS(test_matrix.template dim<0>(), 2UL);
    REQUIRE_EQUALS(test_matrix.template dim<1>(), 3UL);
    REQUIRE_EQUALS(test_matrix.dim(0), 2UL);
    REQUIRE_EQUALS(test_matrix.dim(1), 3UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_matrix/init_2", "fast_matrix::operator=(T)", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix;

    test_matrix = 3.3;

    REQUIRE_EQUALS(test_matrix.size(), 4UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_matrix/init_3", "fast_matrix::fast_matrix(initializer_list)", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {1.0, 3.0, 5.0, 2.0};

    REQUIRE_EQUALS(test_matrix.size(), 4UL);

    REQUIRE_EQUALS(test_matrix[0], 1.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 5.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/init_4", "fast_matrix::fast_matrix(T)", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 4> test_matrix(3.3);

    REQUIRE_EQUALS(test_matrix.size(), 24UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_matrix/init_5", "fast_matrix::operator=(T)", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 4> test_matrix;

    test_matrix = 3.3;

    REQUIRE_EQUALS(test_matrix.size(), 24UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_matrix/init_6", "fast_matrix::operator=(T)", Z, float, double) {
    etl::fast_matrix<Z, 5> test_matrix(3.3);

    REQUIRE_EQUALS(test_matrix.size(), 5UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], Z(3.3));
        REQUIRE_EQUALS(test_matrix(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("fast_matrix/dim_0", "fast_matrix::fast_matrix(T)", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 4, 5, 6, 7> test_matrix(3.3);

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

TEMPLATE_TEST_CASE_2("fast_matrix/access", "fast_matrix::operator()", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

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

TEMPLATE_TEST_CASE_2("fast_matrix/add_scalar_1", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 + test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_scalar_2", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix + 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_scalar_3", "fast_matrix::operator+=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix += 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_scalar_4", "fast_matrix::operator+=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0, 1.0, 1.0, 1.0, 1.0};

    test_matrix += 1.0;

    REQUIRE_EQUALS(test_matrix[0], 0.0);
    REQUIRE_EQUALS(test_matrix[1], 3.0);
    REQUIRE_EQUALS(test_matrix[2], 6.5);
    REQUIRE_EQUALS(test_matrix[7], 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_1", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_2", "fast_matrix::operator+=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_3", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 2> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1.5);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], 9.0);
    REQUIRE_EQUALS(c[7], 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/add_4", "fast_matrix::operator+=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a[0], 1.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 9.0);
    REQUIRE_EQUALS(a[7], 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_scalar_1", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = 1.0 - test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 2.0);
    REQUIRE_EQUALS(test_matrix[1], -1.0);
    REQUIRE_EQUALS(test_matrix[2], -4.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_scalar_2", "fast_matrix::operator+", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix = test_matrix - 1.0;

    REQUIRE_EQUALS(test_matrix[0], -2.0);
    REQUIRE_EQUALS(test_matrix[1], 1.0);
    REQUIRE_EQUALS(test_matrix[2], 4.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_scalar_3", "fast_matrix::operator+=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.5, 1.0};

    test_matrix -= 1.0;

    REQUIRE_EQUALS(test_matrix[0], -2.0);
    REQUIRE_EQUALS(test_matrix[1], 1.0);
    REQUIRE_EQUALS(test_matrix[2], 4.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_1", "fast_matrix::operator-", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c;
    c = a - b;

    REQUIRE_EQUALS(c[0], -3.5);
    REQUIRE_EQUALS(c[1], -1.0);
    REQUIRE_EQUALS(c[2], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_2", "fast_matrix::operator-=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a -= b;

    REQUIRE_EQUALS(a[0], -3.5);
    REQUIRE_EQUALS(a[1], -1.0);
    REQUIRE_EQUALS(a[2], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul_scalar_1", "fast_matrix::operator*", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 * test_matrix;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul_scalar_2", "fast_matrix::operator*", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix * 2.5;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul_scalar_3", "fast_matrix::operator*=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix *= 2.5;

    REQUIRE_EQUALS(test_matrix[0], -2.5);
    REQUIRE_EQUALS(test_matrix[1], 5.0);
    REQUIRE_EQUALS(test_matrix[2], 12.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul/1", "fast_matrix::operator*=", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {-1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> b = {2.5, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> c;

    c = a >> b;

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
    REQUIRE_EQUALS(c[3], 1.0);
    REQUIRE_EQUALS(c[4], 4.0);
    REQUIRE_EQUALS(c[5], 9.0);
    REQUIRE_EQUALS(c[6], 16.0);
    REQUIRE_EQUALS(c[7], 25.0);
    REQUIRE_EQUALS(c[8], 36.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul/2", "fast_matrix::operator*=", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {-1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> b = {2.5, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    a = a >> b;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 6.0);
    REQUIRE_EQUALS(a[2], 20.0);
    REQUIRE_EQUALS(a[3], 1.0);
    REQUIRE_EQUALS(a[4], 4.0);
    REQUIRE_EQUALS(a[5], 9.0);
    REQUIRE_EQUALS(a[6], 16.0);
    REQUIRE_EQUALS(a[7], 25.0);
    REQUIRE_EQUALS(a[8], 36.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul/3", "fast_matrix::operator*=", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {-1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> b = {2.5, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    a >>= b;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 6.0);
    REQUIRE_EQUALS(a[2], 20.0);
    REQUIRE_EQUALS(a[3], 1.0);
    REQUIRE_EQUALS(a[4], 4.0);
    REQUIRE_EQUALS(a[5], 9.0);
    REQUIRE_EQUALS(a[6], 16.0);
    REQUIRE_EQUALS(a[7], 25.0);
    REQUIRE_EQUALS(a[8], 36.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul/4", "fast_matrix::operator*=", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {-1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    a >>= a;

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 4.0);
    REQUIRE_EQUALS(a[2], 25.0);
    REQUIRE_EQUALS(a[3], 1.0);
    REQUIRE_EQUALS(a[4], 4.0);
    REQUIRE_EQUALS(a[5], 9.0);
    REQUIRE_EQUALS(a[6], 16.0);
    REQUIRE_EQUALS(a[7], 25.0);
    REQUIRE_EQUALS(a[8], 36.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul_1", "fast_matrix::operator*", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c;
    c = scale(a, b);

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/mul_3", "fast_matrix::operator*", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c;
    c = a.scale(b);

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
    REQUIRE_EQUALS(c[3], 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/div_scalar_1", "fast_matrix::operator/", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = test_matrix / 2.5;

    REQUIRE_EQUALS_APPROX(test_matrix[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/div_scalar_2", "fast_matrix::operator/", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix = 2.5 / test_matrix;

    REQUIRE_EQUALS_APPROX(test_matrix[0], 2.5 / -1.0);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.5 / 2.0);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 2.5 / 5.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/div_scalar_3", "fast_matrix::operator/=", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix = {-1.0, 2.0, 5.0, 1.0};

    test_matrix /= 2.5;

    REQUIRE_EQUALS_APPROX(test_matrix[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[1], 2.0 / 2.5);
    REQUIRE_EQUALS_APPROX(test_matrix[2], 5.0 / 2.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/div_1", "fast_matrix::operator/", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> c;
    c = a / b;

    REQUIRE_EQUALS_APPROX(c[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(c[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(c[2], 5.0 / 4.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/div_2", "fast_matrix::operator/", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a /= b;

    REQUIRE_EQUALS_APPROX(a[0], -1.0 / 2.5);
    REQUIRE_EQUALS_APPROX(a[1], 2.0 / 3.0);
    REQUIRE_EQUALS_APPROX(a[2], 5.0 / 4.0);
}

ETL_TEST_CASE("fast_matrix/mod_scalar_1", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = test_matrix % 2;

    REQUIRE_EQUALS(test_matrix[0], -1 % 2);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 5 % 2);
}

ETL_TEST_CASE("fast_matrix/mod_scalar_2", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix = 2 % test_matrix;

    REQUIRE_EQUALS(test_matrix[0], 2 % -1);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 2 % 5);
}

ETL_TEST_CASE("fast_matrix/mod_scalar_3", "fast_matrix::operator%=") {
    etl::fast_matrix<int, 2, 2> test_matrix = {-1, 2, 5, 1};

    test_matrix %= 2;

    REQUIRE_EQUALS(test_matrix[0], -1 % 2);
    REQUIRE_EQUALS(test_matrix[1], 2 % 2);
    REQUIRE_EQUALS(test_matrix[2], 5 % 2);
}

ETL_TEST_CASE("fast_matrix/mod_1", "fast_matrix::operator%") {
    etl::fast_matrix<int, 2, 2> a = {-1, 2, 5, 1};
    etl::fast_matrix<int, 2, 2> b = {2, 3, 4, 1};

    etl::fast_matrix<int, 2, 2> c;
    c = a % b;

    REQUIRE_EQUALS(c[0], -1 % 2);
    REQUIRE_EQUALS(c[1], 2 % 3);
    REQUIRE_EQUALS(c[2], 5 % 4);
}

ETL_TEST_CASE("fast_matrix/mod_2", "fast_matrix::operator%=") {
    etl::fast_matrix<int, 2, 2> a = {-1, 2, 5, 1};
    etl::fast_matrix<int, 2, 2> b = {2, 3, 4, 1};

    a %= b;

    REQUIRE_EQUALS(a[0], -1 % 2);
    REQUIRE_EQUALS(a[1], 2 % 3);
    REQUIRE_EQUALS(a[2], 5 % 4);
}

// Unary operator tests

TEMPLATE_TEST_CASE_2("fast_matrix/minus_1", "fast_matrix::minus", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = -a;

    REQUIRE_EQUALS(d(0, 0), 1.0);
    REQUIRE_EQUALS(d(0, 1), -2.0);
    REQUIRE_EQUALS(d(1, 0), -5.0);
    REQUIRE_EQUALS(d(1, 1), -1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/plus_1", "fast_matrix::plus", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {-1.0, 2.0, 5.0, 1.0, 0.0, 3.3, 2.2, -1.4};

    etl::fast_matrix<Z, 2, 4> d;
    d = +a;

    REQUIRE_EQUALS(d(0, 0), Z(-1.0));
    REQUIRE_EQUALS(d(0, 1), Z(2.0));
    REQUIRE_EQUALS(d(0, 2), Z(5.0));
    REQUIRE_EQUALS(d(0, 3), Z(1.0));
    REQUIRE_EQUALS(d(1, 0), Z(0.0));
    REQUIRE_EQUALS(d(1, 1), Z(3.3));
    REQUIRE_EQUALS(d(1, 2), Z(2.2));
    REQUIRE_EQUALS(d(1, 3), Z(-1.4));
}

// Complex tests

TEMPLATE_TEST_CASE_2("fast_matrix/complex", "fast_matrix::complex", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2> c = {1.2, -3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = 2.5 * ((a >> b) / (a + c)) / (1.5 * (a >> b) / c);

    REQUIRE_EQUALS_APPROX(d[0], 10.0);
    REQUIRE_EQUALS_APPROX(d[1], 5.0);
    REQUIRE_EQUALS_APPROX(d[2], 0.68627);
}

TEMPLATE_TEST_CASE_2("fast_matrix/complex_2", "fast_matrix::complex", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {1.1, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, -3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2> c = {2.2, 3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = 2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c);

    REQUIRE_EQUALS_APPROX(d[0], 46.39429);
    REQUIRE_EQUALS_APPROX(d[1], 9.13499);
    REQUIRE_EQUALS_APPROX(d[2], 5.8273);
}

TEMPLATE_TEST_CASE_2("fast_matrix/complex_3", "fast_matrix::complex", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = 2.5 / (a >> b);

    REQUIRE_EQUALS_APPROX(d[0], -1.0);
    REQUIRE_EQUALS_APPROX(d[1], 0.416666);
    REQUIRE_EQUALS_APPROX(d[2], 0.125);
}

TEMPLATE_TEST_CASE_2("fast_matrix/complex_4", "fast_matrix::complex", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.5, -3.0, 4.0, 1.0, 2.5, -3.0, 4.0, 1.0};
    etl::fast_matrix<Z, 2, 2, 2> c = {2.2, 3.0, 3.5, 1.0, 2.2, 3.0, 3.5, 1.0};

    etl::fast_matrix<Z, 2, 2, 2> d;
    d = 2.5 * ((a >> b) / (log(a) >> abs(c))) / (1.5 * scale(a, sign(b)) / c) + 2.111 / log(c);

    REQUIRE_EQUALS_APPROX(d[0], 46.39429);
    REQUIRE_EQUALS_APPROX(d[1], 9.13499);
    REQUIRE_EQUALS_APPROX(d[2], 5.8273);
    REQUIRE_EQUALS_APPROX(d[4], 46.39429);
    REQUIRE_EQUALS_APPROX(d[5], 9.13499);
    REQUIRE_EQUALS_APPROX(d[6], 5.8273);
}

// is_finite tests

TEMPLATE_TEST_CASE_2("fast_matrix/is_finite_1", "fast_matrix::is_finite", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    REQUIRE_DIRECT(a.is_finite());
}

TEMPLATE_TEST_CASE_2("fast_matrix/is_finite_2", "fast_matrix::is_finite", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, NAN, 5.0, 1.0};

    REQUIRE_DIRECT(!a.is_finite());
}

TEMPLATE_TEST_CASE_2("fast_matrix/is_finite_3", "fast_matrix::is_finite", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 1.0, INFINITY, 1.0};

    REQUIRE_DIRECT(!a.is_finite());
}

// scale tests

TEMPLATE_TEST_CASE_2("fast_matrix/scale_1", "", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    a *= 2.5;

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 12.5);
    REQUIRE_EQUALS(a[3], 2.5);
}

TEMPLATE_TEST_CASE_2("fast_matrix/scale_2", "", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 2.0, 3.0, -1.2};

    a *= b;

    REQUIRE_EQUALS(a[0], Z(-2.5));
    REQUIRE_EQUALS(a[1], Z(4.0));
    REQUIRE_EQUALS(a[2], Z(15.0));
    REQUIRE_EQUALS(a[3], Z(-1.2));
}

TEMPLATE_TEST_CASE_2("fast_matrix/scale_3", "", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b = {2.5, 2.0, 3.0, -1.2};

    a.scale_inplace(b);

    REQUIRE_EQUALS(a[0], Z(-2.5));
    REQUIRE_EQUALS(a[1], Z(4.0));
    REQUIRE_EQUALS(a[2], Z(15.0));
    REQUIRE_EQUALS(a[3], Z(-1.2));
}

TEMPLATE_TEST_CASE_2("fast_matrix/scale_4", "", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    a.scale_inplace(2.5);

    REQUIRE_EQUALS(a[0], -2.5);
    REQUIRE_EQUALS(a[1], 5.0);
    REQUIRE_EQUALS(a[2], 12.5);
    REQUIRE_EQUALS(a[3], 2.5);
}

// swap tests

TEMPLATE_TEST_CASE_2("fast_matrix/swap_1", "", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {1.0, 3.3, 4.4, 9, 10.1, -1.1};

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

TEMPLATE_TEST_CASE_2("fast_matrix/swap_2", "", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a = {-1.0, 2.0, 5.0, 1.0, 1.1, 1.9};
    etl::fast_matrix<Z, 3, 2> b = {1.0, 3.3, 4.4, 9, 10.1, -1.1};

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

TEMPLATE_TEST_CASE_2("fast_matrix/binary_sub", "[binary][sub]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 2> a = {-1.0, 2.0, 5.5, 1.0};
    etl::fast_matrix<Z, 2, 1, 2> b = {-1.0, 2.0, 5.5, 1.0};
    etl::fast_matrix<Z, 2> c;

    REQUIRE_EQUALS(a(0, 0, 0), -1.0);

    c = (a + b)(1)(0);

    REQUIRE_EQUALS(c[0], 11.0);
    REQUIRE_EQUALS(c[1], 2.0);

    REQUIRE_EQUALS((a + b)(1)(0)(0), 11.0);
    REQUIRE_EQUALS((a + b)(1)(0)(1), 2.0);
}

TEMPLATE_TEST_CASE_2("one_if_max_sub/1", "[fast]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 3.0, 5.0, 2.0, 7, 1.0};
    etl::fast_matrix<Z, 2, 3> b;

    b = one_if_max_sub(a);

    REQUIRE_EQUALS(b(0, 0), Z(0.0));
    REQUIRE_EQUALS(b(0, 1), Z(0.0));
    REQUIRE_EQUALS(b(0, 2), Z(1.0));

    REQUIRE_EQUALS(b(1, 0), Z(0.0));
    REQUIRE_EQUALS(b(1, 1), Z(1.0));
    REQUIRE_EQUALS(b(1, 2), Z(0.0));
}

TEMPLATE_TEST_CASE_2("one_if_max_sub/2", "[fast]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a = {1.0, 3.0, 5.0, 2.0, 7, 1.0};
    etl::fast_matrix<Z, 3, 2> b;

    b = one_if_max_sub(a);

    REQUIRE_EQUALS(b(0, 0), Z(0.0));
    REQUIRE_EQUALS(b(0, 1), Z(1.0));

    REQUIRE_EQUALS(b(1, 0), Z(1.0));
    REQUIRE_EQUALS(b(1, 1), Z(0.0));

    REQUIRE_EQUALS(b(2, 0), Z(1.0));
    REQUIRE_EQUALS(b(2, 1), Z(0.0));
}

ETL_TEST_CASE("fast_matrix/mixed/0", "fast_matrix::operator*=") {
    etl::fast_matrix<float, 3, 3> a = {-1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<double, 3, 3> b = {2.5, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<float, 3, 3> c;

    c = a >> b;

    REQUIRE_EQUALS(c[0], -2.5);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], 20.0);
    REQUIRE_EQUALS(c[3], 1.0);
    REQUIRE_EQUALS(c[4], 4.0);
    REQUIRE_EQUALS(c[5], 9.0);
    REQUIRE_EQUALS(c[6], 16.0);
    REQUIRE_EQUALS(c[7], 25.0);
    REQUIRE_EQUALS(c[8], 36.0);
}
