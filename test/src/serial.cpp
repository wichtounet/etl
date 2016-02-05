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

TEST_CASE("serial_section/1", "[fast][serial]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    REQUIRE(!etl::local_context().serial);

    SERIAL_SECTION {
        REQUIRE(etl::local_context().serial);
        b = a + a;
        b += a + a;
    }

    REQUIRE(!etl::local_context().serial);

    REQUIRE(b[0] == 4.0);
}

TEST_CASE("serial_section/2", "[fast][serial]") {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b;

    REQUIRE(!etl::local_context().serial);

    etl::local_context().serial = true;

    SERIAL_SECTION {
        REQUIRE(etl::local_context().serial);
        b = a + a;
        b += a + a;

        etl::local_context().serial = false;
    }

    REQUIRE(etl::local_context().serial);

    REQUIRE(b[0] == 4.0);
}
