//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("elt_compare/equal/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.0, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = equal(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}
