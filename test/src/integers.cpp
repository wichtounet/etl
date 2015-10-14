//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// Init tests

TEMPLATE_TEST_CASE_2("integers/init_1", "[integers][fast]", Z, int, long) {
    etl::fast_vector<Z, 4> test_vector(3);

    REQUIRE(test_vector.size() == 4);

    for (std::size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE(test_vector[i] == 3);
    }
}

TEMPLATE_TEST_CASE_2("integers/init_2", "[integers][fast]", Z, int, long) {
    etl::fast_vector<Z, 4> test_vector;

    test_vector = 3;

    REQUIRE(test_vector.size() == 4);

    for (std::size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE(test_vector[i] == 3);
    }
}

TEMPLATE_TEST_CASE_2("integeres/init_3", "[integers][fast]", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {1, 2, 3};

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1);
    REQUIRE(test_vector[1] == 2);
    REQUIRE(test_vector[2] == 3);
}

TEMPLATE_TEST_CASE_2("integers/init_4", "[integers][dyn]", Z, int, long) {
    etl::dyn_vector<Z> test_vector(4UL, 3);

    REQUIRE(test_vector.size() == 4);

    for (std::size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE(test_vector[i] == 3);
        REQUIRE(test_vector(i) == 3);
    }
}

TEMPLATE_TEST_CASE_2("integers/init_5", "[integers][dyn]", Z, int, long) {
    etl::dyn_vector<Z> test_vector(4);

    test_vector = 3;

    REQUIRE(test_vector.size() == 4);

    for (std::size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE(test_vector[i] == 3);
        REQUIRE(test_vector(i) == 3);
    }
}

TEMPLATE_TEST_CASE_2("integers/init_6", "[integers][dyn]", Z, int, long) {
    etl::dyn_vector<Z> test_vector({1, 2, 3});

    REQUIRE(test_vector.size() == 3);

    REQUIRE(test_vector[0] == 1);
    REQUIRE(test_vector[1] == 2);
    REQUIRE(test_vector[2] == 3);
}

// Binary operators test

TEMPLATE_TEST_CASE_2("integers/add_scalar_1", "fast_vector::operator+", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 1 + test_vector;

    REQUIRE(test_vector[0] == 0);
    REQUIRE(test_vector[1] == 3);
    REQUIRE(test_vector[2] == 6);
}

TEMPLATE_TEST_CASE_2("integers/add_scalar_2", "fast_vector::operator+", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector + 1;

    REQUIRE(test_vector[0] == 0);
    REQUIRE(test_vector[1] == 3);
    REQUIRE(test_vector[2] == 6);
}

TEMPLATE_TEST_CASE_2("integers/add_scalar_3", "fast_vector::operator+=", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector += 1;

    REQUIRE(test_vector[0] == 0);
    REQUIRE(test_vector[1] == 3);
    REQUIRE(test_vector[2] == 6);
}

TEMPLATE_TEST_CASE_2("integers/add_1", "fast_vector::operator+", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c(a + b);

    REQUIRE(c[0] == 1);
    REQUIRE(c[1] == 5);
    REQUIRE(c[2] == 9);
}

TEMPLATE_TEST_CASE_2("integers/add_2", "fast_vector::operator+=", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a += b;

    REQUIRE(a[0] == 1);
    REQUIRE(a[1] == 5);
    REQUIRE(a[2] == 9);
}

TEMPLATE_TEST_CASE_2("integers/sub_scalar_1", "fast_vector::operator+", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 1 - test_vector;

    REQUIRE(test_vector[0] == 2);
    REQUIRE(test_vector[1] == -1);
    REQUIRE(test_vector[2] == -4);
}

TEMPLATE_TEST_CASE_2("integers/sub_scalar_2", "fast_vector::operator+", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector - 1;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 1);
    REQUIRE(test_vector[2] == 4);
}

TEMPLATE_TEST_CASE_2("integers/sub_scalar_3", "fast_vector::operator+=", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector -= 1;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 1);
    REQUIRE(test_vector[2] == 4);
}

TEMPLATE_TEST_CASE_2("integers/sub_1", "fast_vector::operator-", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c(a - b);

    REQUIRE(c[0] == -3);
    REQUIRE(c[1] == -1);
    REQUIRE(c[2] == 1);
}

TEMPLATE_TEST_CASE_2("integers/sub_2", "fast_vector::operator-=", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a -= b;

    REQUIRE(a[0] == -3);
    REQUIRE(a[1] == -1);
    REQUIRE(a[2] == 1);
}

TEMPLATE_TEST_CASE_2("integers/mul_scalar_1", "fast_vector::operator*", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 2 * test_vector;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 4);
    REQUIRE(test_vector[2] == 10);
}

TEMPLATE_TEST_CASE_2("integers/mul_scalar_2", "fast_vector::operator*", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector * 2;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 4);
    REQUIRE(test_vector[2] == 10);
}

TEMPLATE_TEST_CASE_2("integers/mul_scalar_3", "fast_vector::operator*=", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector *= 2;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 4);
    REQUIRE(test_vector[2] == 10);
}

TEMPLATE_TEST_CASE_2("integers/mul_1", "fast_vector::operator*", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c(a >> b);

    REQUIRE(c[0] == -2);
    REQUIRE(c[1] == 6);
    REQUIRE(c[2] == 20);
}

TEMPLATE_TEST_CASE_2("integers/mul_2", "fast_vector::operator*=", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a *= b;

    REQUIRE(a[0] == -2);
    REQUIRE(a[1] == 6);
    REQUIRE(a[2] == 20);
}

TEMPLATE_TEST_CASE_2("integers/mul_3", "fast_vector::operator*", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c(a >> b);

    REQUIRE(c[0] == -2);
    REQUIRE(c[1] == 6);
    REQUIRE(c[2] == 20);
}

TEMPLATE_TEST_CASE_2("integers/div_scalar_1", "fast_vector::operator/", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector / 2;

    REQUIRE(test_vector[0] == 0);
    REQUIRE(test_vector[1] == 1);
    REQUIRE(test_vector[2] == 2);
}

TEMPLATE_TEST_CASE_2("integers/div_scalar_2", "fast_vector::operator/", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 2 / test_vector;

    REQUIRE(test_vector[0] == -2);
    REQUIRE(test_vector[1] == 1);
    REQUIRE(test_vector[2] == 0);
}

TEMPLATE_TEST_CASE_2("integers/div_scalar_3", "fast_vector::operator/=", Z, int, long) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector /= 2;

    REQUIRE(test_vector[0] == 0);
    REQUIRE(test_vector[1] == 1);
    REQUIRE(test_vector[2] == 2);
}

TEMPLATE_TEST_CASE_2("integers/div_1", "fast_vector::operator/", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c(a / b);

    REQUIRE(c[0] == 0);
    REQUIRE(c[1] == 0);
    REQUIRE(c[2] == 1);
}

TEMPLATE_TEST_CASE_2("integers/div_2", "fast_vector::operator/=", Z, int, long) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a /= b;

    REQUIRE(a[0] == 0);
    REQUIRE(a[1] == 0);
    REQUIRE(a[2] == 1);
}
