//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("serial/1", "[fast][serial]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = serial(a + a);

    REQUIRE_EQUALS(b[0], 2.0);
}

TEMPLATE_TEST_CASE_2("serial/2", "[dyn][serial][sum]", Z, float, double) {
    etl::dyn_matrix<Z> a(500, 500);

    a = 12.0;

    Z sum = 0.0;

    SERIAL_SECTION {
        sum = etl::sum(a);
    }

    REQUIRE_EQUALS(sum, 12.0 * etl::size(a));
}

TEMPLATE_TEST_CASE_2("serial/3", "[fast][serial]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = serial(a + a);
    b += serial(a + a);

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("serial_section/1", "[fast][serial]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    REQUIRE_DIRECT(!etl::local_context().serial);

    SERIAL_SECTION {
        REQUIRE_DIRECT(etl::local_context().serial);
        b = a + a;
        b += a + a;
    }

    REQUIRE_DIRECT(!etl::local_context().serial);

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("serial_section/2", "[fast][serial]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    REQUIRE_DIRECT(!etl::local_context().serial);

    etl::local_context().serial = true;

    SERIAL_SECTION {
        REQUIRE_DIRECT(etl::local_context().serial);
        b = a + a;
        b += a + a;

        etl::local_context().serial = false;
    }

    REQUIRE_DIRECT(etl::local_context().serial);

    etl::local_context().serial = false;

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("serial_section/3", "[fast][serial]", Z, float, double) {
    REQUIRE_DIRECT(!etl::local_context().serial);

    SERIAL_SECTION {
        REQUIRE_DIRECT(etl::local_context().serial);

        etl::local_context().serial = false;

        SERIAL_SECTION {
            REQUIRE_DIRECT(etl::local_context().serial);
        }

        REQUIRE_DIRECT(!etl::local_context().serial);
    }

    REQUIRE_DIRECT(!etl::local_context().serial);
}
