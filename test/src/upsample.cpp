//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("upsample/2d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 4, 4> c(etl::upsample_2d<2, 2>(a));

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 1.0);
    REQUIRE_EQUALS(c(1, 0), 1.0);
    REQUIRE_EQUALS(c(1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 3), 2.0);
    REQUIRE_EQUALS(c(1, 2), 2.0);
    REQUIRE_EQUALS(c(1, 3), 2.0);

    REQUIRE_EQUALS(c(2, 0), 3.0);
    REQUIRE_EQUALS(c(2, 1), 3.0);
    REQUIRE_EQUALS(c(3, 0), 3.0);
    REQUIRE_EQUALS(c(3, 1), 3.0);

    REQUIRE_EQUALS(c(2, 2), 4.0);
    REQUIRE_EQUALS(c(2, 3), 4.0);
    REQUIRE_EQUALS(c(3, 2), 4.0);
    REQUIRE_EQUALS(c(3, 3), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/3d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 1, 4, 4> c(etl::upsample_3d<1, 2, 2>(a));

    REQUIRE_EQUALS(c(0, 0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 0, 1), 1.0);
    REQUIRE_EQUALS(c(0, 1, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 0, 3), 2.0);
    REQUIRE_EQUALS(c(0, 1, 2), 2.0);
    REQUIRE_EQUALS(c(0, 1, 3), 2.0);

    REQUIRE_EQUALS(c(0, 2, 0), 3.0);
    REQUIRE_EQUALS(c(0, 2, 1), 3.0);
    REQUIRE_EQUALS(c(0, 3, 0), 3.0);
    REQUIRE_EQUALS(c(0, 3, 1), 3.0);

    REQUIRE_EQUALS(c(0, 2, 2), 4.0);
    REQUIRE_EQUALS(c(0, 2, 3), 4.0);
    REQUIRE_EQUALS(c(0, 3, 2), 4.0);
    REQUIRE_EQUALS(c(0, 3, 3), 4.0);
}
