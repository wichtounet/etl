//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "dot_test.hpp"

TEMPLATE_TEST_CASE_2("cross/1", "[cross]", T, float, double) {
    etl::fast_vector<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<T, 3> b = {4.0, 5.0, 6.0};

    auto c = etl::cross(a, b);

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], -3.0);
}

TEMPLATE_TEST_CASE_2("cross/2", "[cross]", T, float, double) {
    etl::dyn_vector<T> a{1.0, 2.0, 3.0};
    etl::dyn_vector<T> b{4.0, 5.0, 6.0};

    auto c = etl::cross(a, b);

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], 6.0);
    REQUIRE_EQUALS(c[2], -3.0);
}
