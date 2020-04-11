//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

ETL_TEST_CASE("elt_logical/and/1", "[compare]") {
    etl::fast_matrix<bool, 2, 2> a{false, false, true, true};
    etl::fast_matrix<bool, 2, 2> b{false, true, true, false};
    etl::fast_matrix<bool, 2, 2> c;

    c = logical_and(a, b);

    REQUIRE_EQUALS(c(0, 0), false);
    REQUIRE_EQUALS(c(0, 1), false);
    REQUIRE_EQUALS(c(1, 0), true);
    REQUIRE_EQUALS(c(1, 1), false);
}

ETL_TEST_CASE("elt_logical/or/1", "[compare]") {
    etl::fast_matrix<bool, 2, 2> a{false, false, true, true};
    etl::fast_matrix<bool, 2, 2> b{false, true, true, false};
    etl::fast_matrix<bool, 2, 2> c;

    c = logical_or(a, b);

    REQUIRE_EQUALS(c(0, 0), false);
    REQUIRE_EQUALS(c(0, 1), true);
    REQUIRE_EQUALS(c(1, 0), true);
    REQUIRE_EQUALS(c(1, 1), true);
}

ETL_TEST_CASE("elt_logical/xor/1", "[compare]") {
    etl::fast_dyn_matrix<bool, 2, 2> a{false, false, true, true};
    etl::fast_dyn_matrix<bool, 2, 2> b{false, true, true, false};
    etl::fast_dyn_matrix<bool, 2, 2> c;

    c = logical_xor(a, b);

    REQUIRE_EQUALS(c(0, 0), false);
    REQUIRE_EQUALS(c(0, 1), true);
    REQUIRE_EQUALS(c(1, 0), false);
    REQUIRE_EQUALS(c(1, 1), true);
}
