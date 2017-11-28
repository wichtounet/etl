//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#ifndef ETL_CUDA

ETL_TEST_CASE("optimize/6", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(0.0 * a + 0.0 * a);

    REQUIRE_EQUALS(b[0], 0.0);
}

ETL_TEST_CASE("optimize/7", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(0.0 * a + 0.0 * a + 1.0 * a);

    REQUIRE_EQUALS(b[0], 1.0);
}

ETL_TEST_CASE("optimize/8", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(0.0 * a + 1.0 * a + 1.0 * (a - 0));

    REQUIRE_EQUALS(b[0], 2.0);
}

ETL_TEST_CASE("optimize/10", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(+((-(a * 1.0)) * 1.0));

    REQUIRE_EQUALS(b[0], -1.0);
}

#endif
