//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("unmanaged/dyn_1", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_vector(8);

    etl::custom_dyn_matrix<Z, 1> vector(raw_vector.data(), 8);

    vector = 3.3;

    for (size_t i = 0; i < vector.size(); ++i) {
        REQUIRE_EQUALS(vector[i], Z(3.3));
        REQUIRE_EQUALS(vector(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("unmanaged/dyn_2", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_a(4);
    std::vector<Z> raw_b(4);
    std::vector<Z> raw_c(4);

    raw_a[0] = 0.5;
    raw_a[1] = 1.5;
    raw_a[2] = -0.5;
    raw_a[3] = 3.5;

    const auto* data_a = raw_a.data();

    etl::custom_dyn_matrix<const Z, 1> v_a(data_a, 4UL);
    etl::custom_dyn_matrix<Z, 1> v_b(raw_b.data(), 4UL);
    etl::custom_dyn_matrix<Z, 1> v_c(raw_c.data(), 4UL);

    v_b[0] = -0.5;
    v_b[1] = 4.5;
    v_b[2] = -2.3;
    v_b[3] = 1.5;

    v_c = v_a + v_b;

    REQUIRE_EQUALS(raw_a[0], Z(0.5));
    REQUIRE_EQUALS(raw_a[1], Z(1.5));
    REQUIRE_EQUALS(raw_a[2], Z(-0.5));
    REQUIRE_EQUALS(raw_a[3], Z(3.5));

    REQUIRE_EQUALS(raw_b[0], Z(-0.5));
    REQUIRE_EQUALS(raw_b[1], Z(4.5));
    REQUIRE_EQUALS(raw_b[2], Z(-2.3));
    REQUIRE_EQUALS(raw_b[3], Z(1.5));

    REQUIRE_EQUALS(raw_c[0], Z(0.0));
    REQUIRE_EQUALS(raw_c[1], Z(6.0));
    REQUIRE_EQUALS(raw_c[2], Z(-2.8));
    REQUIRE_EQUALS(raw_c[3], Z(5.0));
}

TEMPLATE_TEST_CASE_2("unmanaged/fast_1", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_vector(8);

    etl::custom_fast_matrix<Z, 8> vector(raw_vector.data());

    vector = 3.3;

    for (size_t i = 0; i < vector.size(); ++i) {
        REQUIRE_EQUALS(vector[i], Z(3.3));
        REQUIRE_EQUALS(vector(i), Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("unmanaged/fast_2", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_a(4);
    std::vector<Z> raw_b(4);
    std::vector<Z> raw_c(4);

    raw_a[0] = 0.5;
    raw_a[1] = 1.5;
    raw_a[2] = -0.5;
    raw_a[3] = 3.5;

    const auto* data_a = raw_a.data();

    etl::custom_fast_matrix<const Z, 4> v_a(data_a);
    etl::custom_fast_matrix<Z, 4> v_b(raw_b.data());
    etl::custom_fast_matrix<Z, 4> v_c(raw_c.data());

    v_b[0] = -0.5;
    v_b[1] = 4.5;
    v_b[2] = -2.3;
    v_b[3] = 1.5;

    v_c = v_a >> v_b;

    REQUIRE_EQUALS(raw_a[0], Z(0.5));
    REQUIRE_EQUALS(raw_a[1], Z(1.5));
    REQUIRE_EQUALS(raw_a[2], Z(-0.5));
    REQUIRE_EQUALS(raw_a[3], Z(3.5));

    REQUIRE_EQUALS(raw_b[0], Z(-0.5));
    REQUIRE_EQUALS(raw_b[1], Z(4.5));
    REQUIRE_EQUALS(raw_b[2], Z(-2.3));
    REQUIRE_EQUALS(raw_b[3], Z(1.5));

    REQUIRE_EQUALS(raw_c[0], Z(0.5 * -0.5));
    REQUIRE_EQUALS(raw_c[1], Z(1.5 * 4.5));
    REQUIRE_EQUALS(raw_c[2], Z(-0.5 * -2.3));
    REQUIRE_EQUALS(raw_c[3], Z(3.5 * 1.5));
}
