//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("unmanaged/dyn_1", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_vector(8);

    auto vector = etl::dyn_matrix_over(raw_vector.data(), 8);

    vector = 3.3;

    for (std::size_t i = 0; i < vector.size(); ++i) {
        REQUIRE(vector[i] == Z(3.3));
        REQUIRE(vector(i) == Z(3.3));
    }
}

TEMPLATE_TEST_CASE_2("unmanaged/dyn_2", "[dyn][unmanaged]", Z, double, float) {
    std::vector<Z> raw_a(4);
    std::vector<Z> raw_b(4);
    std::vector<Z> raw_c(4);

    auto v_a = etl::dyn_matrix_over(raw_a.data(), 4);
    auto v_b = etl::dyn_matrix_over(raw_b.data(), 4);
    auto v_c = etl::dyn_matrix_over(raw_c.data(), 4);

    v_a[0] = 0.5;
    v_a[1] = 1.5;
    v_a[2] = -0.5;
    v_a[3] = 3.5;

    v_b[0] = -0.5;
    v_b[1] = 4.5;
    v_b[2] = -2.3;
    v_b[3] = 1.5;

    v_c = v_a + v_b;

    REQUIRE(raw_a[0] == Z(0.5));
    REQUIRE(raw_a[1] == Z(1.5));
    REQUIRE(raw_a[2] == Z(-0.5));
    REQUIRE(raw_a[3] == Z(3.5));

    REQUIRE(raw_b[0] == Z(-0.5));
    REQUIRE(raw_b[1] == Z(4.5));
    REQUIRE(raw_b[2] == Z(-2.3));
    REQUIRE(raw_b[3] == Z(1.5));

    REQUIRE(raw_c[0] == Z(0.0));
    REQUIRE(raw_c[1] == Z(6.0));
    REQUIRE(raw_c[2] == Z(-2.8));
    REQUIRE(raw_c[3] == Z(5.0));
}
