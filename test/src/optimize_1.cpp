//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#ifndef ETL_CUDA

ETL_TEST_CASE("optimize/1", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(a + a);

    REQUIRE_EQUALS(b[0], 2.0);
}

ETL_TEST_CASE("optimize/2", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(a * 1.0);

    REQUIRE_EQUALS(b[0], 1.0);
}

ETL_TEST_CASE("optimize/3", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(a + a * 1.0);

    REQUIRE_EQUALS(b[0], 2.0);
}

ETL_TEST_CASE("optimize/4", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(a + 1.0 * a);

    REQUIRE_EQUALS(b[0], 2.0);
}

ETL_TEST_CASE("optimize/5", "[fast][optimizer]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    b = opt(a * 1.0 + 1.0 * a);

    REQUIRE_EQUALS(b[0], 2.0);
}

#endif
