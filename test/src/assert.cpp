//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#undef NDEBUG
#define CPP_UTILS_ASSERT_EXCEPTION

#include "etl/etl_light.hpp"
#include "catch.hpp"

TEST_CASE("dyn_vector/assert", "assert") {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0, 1.0};

    REQUIRE_THROWS(a + b);
}
