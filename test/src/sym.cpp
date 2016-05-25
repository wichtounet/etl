//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("sym/fast_matrix/1", "[sym][fast]", Z, float, double) {
    etl::sym_matrix<etl::fast_matrix<Z, 2,2>> a;
    etl::sym_matrix<etl::fast_matrix<Z, 1,1>> b;
    etl::sym_matrix<etl::fast_dyn_matrix<Z, 1,1>> c(Z(0.0));
    etl::sym_matrix<etl::dyn_matrix<Z>> d(3, Z(1.0));

    REQUIRE_EQUALS(a.dimensions(), 2UL);
    REQUIRE_EQUALS(b.dimensions(), 2UL);
    REQUIRE_EQUALS(c.dimensions(), 2UL);
    REQUIRE_EQUALS(d.dimensions(), 2UL);
}

TEMPLATE_TEST_CASE_2("sym/fast_matrix/2", "[sym][fast]", Z, float, double) {
    etl::sym_matrix<etl::fast_matrix<Z, 3, 3>> a;

    REQUIRE_EQUALS(a(0, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 2), Z(0.0));

    a(2, 1) = 3.5;

    REQUIRE_EQUALS(a(2, 1), Z(3.5));
    REQUIRE_EQUALS(a(1, 2), Z(3.5));

    a(1, 1) += 5.5;

    REQUIRE_EQUALS(a(1, 1), Z(5.5));

    a(2, 0) = 1.5;

    REQUIRE_EQUALS(a(2, 0), Z(1.5));
    REQUIRE_EQUALS(a(0, 2), Z(1.5));

    a(2, 0) *= 2.0;

    REQUIRE_EQUALS(a(2, 0), Z(3.0));
    REQUIRE_EQUALS(a(0, 2), Z(3.0));
}

TEMPLATE_TEST_CASE_2("sym/fast_matrix/3", "[sym][fast]", Z, float, double) {
    etl::sym_matrix<etl::dyn_matrix<Z>> a(3UL);

    REQUIRE_EQUALS(a(0, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 2), Z(0.0));

    a(2, 1) = 3.5;

    REQUIRE_EQUALS(a(2, 1), Z(3.5));
    REQUIRE_EQUALS(a(1, 2), Z(3.5));

    a(1, 1) += 5.5;

    REQUIRE_EQUALS(a(1, 1), Z(5.5));

    a(2, 0) = 1.5;

    REQUIRE_EQUALS(a(2, 0), Z(1.5));
    REQUIRE_EQUALS(a(0, 2), Z(1.5));

    a(2, 0) *= 2.0;

    REQUIRE_EQUALS(a(2, 0), Z(3.0));
    REQUIRE_EQUALS(a(0, 2), Z(3.0));
}
