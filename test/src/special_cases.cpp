//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("deep_assign/vec<mat>", "deep_assign", Z, float, double) {
    etl::fast_vector<etl::fast_matrix<Z, 2, 3>, 2> a;

    a = 0.0;

    for (auto& v : a) {
        for (auto& v2 : v) {
            REQUIRE_EQUALS(v2, 0.0);
        }
    }
}

TEMPLATE_TEST_CASE_2("deep_assign/mat<vec>", "deep_assign", Z, float, double) {
    etl::fast_matrix<etl::fast_vector<Z, 2>, 2, 3> a;

    a = 0.0;

    for (auto& v : a) {
        for (auto& v2 : v) {
            REQUIRE_EQUALS(v2, 0.0);
        }
    }
}

TEMPLATE_TEST_CASE_2("deep_assign/mat<mat>", "deep_assign", Z, float, double) {
    etl::fast_matrix<etl::fast_matrix<Z, 2, 3>, 2, 3> a;

    a = 0.0;

    for (auto& v : a) {
        for (auto& v2 : v) {
            REQUIRE_EQUALS(v2, 0.0);
        }
    }
}
