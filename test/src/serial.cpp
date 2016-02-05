//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEST_CASE("serial/1", "[fast][serial]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = serial(a + a);

    REQUIRE(b[0] == 2.0);
}
