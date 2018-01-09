//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// Init tests

TEMPLATE_TEST_CASE_4("integers/init_1", "[integers][fast]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 4> test_vector(3);

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], 3);
    }
}

TEMPLATE_TEST_CASE_4("integers/init_2", "[integers][fast]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 4> test_vector;

    test_vector = 3;

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], 3);
    }
}

TEMPLATE_TEST_CASE_4("integeres/init_3", "[integers][fast]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {1, 2, 3};

    REQUIRE_EQUALS(test_vector.size(), 3UL);

    REQUIRE_EQUALS(test_vector[0], 1);
    REQUIRE_EQUALS(test_vector[1], 2);
    REQUIRE_EQUALS(test_vector[2], 3);
}

TEMPLATE_TEST_CASE_4("integers/init_4", "[integers][dyn]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::dyn_vector<Z> test_vector(4UL, 3);

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], 3);
        REQUIRE_EQUALS(test_vector(i), 3);
    }
}

TEMPLATE_TEST_CASE_4("integers/init_5", "[integers][dyn]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::dyn_vector<Z> test_vector(4);

    test_vector = 3;

    REQUIRE_EQUALS(test_vector.size(), 4UL);

    for (size_t i = 0; i < test_vector.size(); ++i) {
        REQUIRE_EQUALS(test_vector[i], 3);
        REQUIRE_EQUALS(test_vector(i), 3);
    }
}

TEMPLATE_TEST_CASE_4("integers/init_6", "[integers][dyn]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::dyn_vector<Z> test_vector({1, 2, 3});

    REQUIRE_EQUALS(test_vector.size(), 3UL);

    REQUIRE_EQUALS(test_vector[0], 1);
    REQUIRE_EQUALS(test_vector[1], 2);
    REQUIRE_EQUALS(test_vector[2], 3);
}

// Binary operators test

TEMPLATE_TEST_CASE_4("integers/add_scalar_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 1 + test_vector;

    REQUIRE_EQUALS(test_vector[0], 0);
    REQUIRE_EQUALS(test_vector[1], 3);
    REQUIRE_EQUALS(test_vector[2], 6);
}

TEMPLATE_TEST_CASE_4("integers/add_scalar_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector + 1;

    REQUIRE_EQUALS(test_vector[0], 0);
    REQUIRE_EQUALS(test_vector[1], 3);
    REQUIRE_EQUALS(test_vector[2], 6);
}

TEMPLATE_TEST_CASE_4("integers/add_scalar_3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector += 1;

    REQUIRE_EQUALS(test_vector[0], 0);
    REQUIRE_EQUALS(test_vector[1], 3);
    REQUIRE_EQUALS(test_vector[2], 6);
}

TEMPLATE_TEST_CASE_4("integers/add_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c;
    c = a + b;

    REQUIRE_EQUALS(c[0], 1);
    REQUIRE_EQUALS(c[1], 5);
    REQUIRE_EQUALS(c[2], 9);
}

TEMPLATE_TEST_CASE_4("integers/add_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a += b;

    REQUIRE_EQUALS(a[0], 1);
    REQUIRE_EQUALS(a[1], 5);
    REQUIRE_EQUALS(a[2], 9);
}

TEMPLATE_TEST_CASE_4("integers/add/3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 131> a;
    etl::fast_vector<Z, 131> b;

    a = 1;
    b = 2;

    a += b;

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS(a[i], Z(3));
    }
}

TEMPLATE_TEST_CASE_4("integers/sub_scalar_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 1 - test_vector;

    REQUIRE_EQUALS(test_vector[0], 2);
    REQUIRE_EQUALS(test_vector[1], -1);
    REQUIRE_EQUALS(test_vector[2], -4);
}

TEMPLATE_TEST_CASE_4("integers/sub_scalar_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector - 1;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 1);
    REQUIRE_EQUALS(test_vector[2], 4);
}

TEMPLATE_TEST_CASE_4("integers/sub_scalar_3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector -= 1;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 1);
    REQUIRE_EQUALS(test_vector[2], 4);
}

TEMPLATE_TEST_CASE_4("integers/sub_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c;
    c = a - b;

    REQUIRE_EQUALS(c[0], -3);
    REQUIRE_EQUALS(c[1], -1);
    REQUIRE_EQUALS(c[2], 1);
}

TEMPLATE_TEST_CASE_4("integers/sub_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a -= b;

    REQUIRE_EQUALS(a[0], -3);
    REQUIRE_EQUALS(a[1], -1);
    REQUIRE_EQUALS(a[2], 1);
}

TEMPLATE_TEST_CASE_4("integers/sub/3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 131> a;
    etl::fast_vector<Z, 131> b;

    a = 4;
    b = 3;

    a -= b;

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS(a[i], Z(1));
    }
}

TEMPLATE_TEST_CASE_4("integers/mul_scalar_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 2 * test_vector;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 4);
    REQUIRE_EQUALS(test_vector[2], 10);
}

TEMPLATE_TEST_CASE_4("integers/mul_scalar_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector * 2;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 4);
    REQUIRE_EQUALS(test_vector[2], 10);
}

TEMPLATE_TEST_CASE_4("integers/mul_scalar_3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector *= 2;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 4);
    REQUIRE_EQUALS(test_vector[2], 10);
}

TEMPLATE_TEST_CASE_4("integers/mul_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c;
    c = a >> b;

    REQUIRE_EQUALS(c[0], -2);
    REQUIRE_EQUALS(c[1], 6);
    REQUIRE_EQUALS(c[2], 20);
}

TEMPLATE_TEST_CASE_4("integers/mul_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a *= b;

    REQUIRE_EQUALS(a[0], -2);
    REQUIRE_EQUALS(a[1], 6);
    REQUIRE_EQUALS(a[2], 20);
}

TEMPLATE_TEST_CASE_4("integers/mul_3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c;
    c = a >> b;

    REQUIRE_EQUALS(c[0], -2);
    REQUIRE_EQUALS(c[1], 6);
    REQUIRE_EQUALS(c[2], 20);
}

TEMPLATE_TEST_CASE_4("integers/mul/4", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 131> a;
    etl::fast_vector<Z, 131> b;

    a = 4;
    b = 3;

    a *= b;

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS(a[i], Z(12));
    }
}

TEMPLATE_TEST_CASE_4("integers/div_scalar_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = test_vector / 2;

    REQUIRE_EQUALS(test_vector[0], 0);
    REQUIRE_EQUALS(test_vector[1], 1);
    REQUIRE_EQUALS(test_vector[2], 2);
}

TEMPLATE_TEST_CASE_4("integers/div_scalar_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector = 2 / test_vector;

    REQUIRE_EQUALS(test_vector[0], -2);
    REQUIRE_EQUALS(test_vector[1], 1);
    REQUIRE_EQUALS(test_vector[2], 0);
}

TEMPLATE_TEST_CASE_4("integers/div_scalar_3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> test_vector = {-1, 2, 5};

    test_vector /= 2;

    REQUIRE_EQUALS(test_vector[0], 0);
    REQUIRE_EQUALS(test_vector[1], 1);
    REQUIRE_EQUALS(test_vector[2], 2);
}

TEMPLATE_TEST_CASE_4("integers/div_1", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    etl::fast_vector<Z, 3> c;
    c = a / b;

    REQUIRE_EQUALS(c[0], 0);
    REQUIRE_EQUALS(c[1], 0);
    REQUIRE_EQUALS(c[2], 1);
}

TEMPLATE_TEST_CASE_4("integers/div_2", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 3> a = {-1, 2, 5};
    etl::fast_vector<Z, 3> b = {2, 3, 4};

    a /= b;

    REQUIRE_EQUALS(a[0], 0);
    REQUIRE_EQUALS(a[1], 0);
    REQUIRE_EQUALS(a[2], 1);
}

TEMPLATE_TEST_CASE_4("integers/div/3", "[integers]", Z, int8_t, int16_t, int32_t, int64_t) {
    etl::fast_vector<Z, 131> a;
    etl::fast_vector<Z, 131> b;

    a = 12;
    b = 3;

    a /= b;

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS(a[i], Z(4));
    }
}
