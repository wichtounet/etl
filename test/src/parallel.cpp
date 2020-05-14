//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("parallel/1", "[fast][parallel]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = parallel(a + a);

    REQUIRE_EQUALS(b[0], 2.0);
}

TEMPLATE_TEST_CASE_2("parallel/2", "[dyn][parallel][sum]", Z, float, double) {
    etl::dyn_matrix<Z> a(500, 500);

    a = 12.0;

    Z sum = 0.0;

    PARALLEL_SECTION {
        sum = etl::sum(a);
    }

    REQUIRE_EQUALS(sum, 12.0 * etl::size(a));
}

TEMPLATE_TEST_CASE_2("parallel/3", "[fast][parallel]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = parallel(a + a);
    b += parallel(a + a);

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("parallel_section/1", "[fast][parallel]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    REQUIRE_DIRECT(!etl::local_context().parallel);

    PARALLEL_SECTION {
        REQUIRE_DIRECT(etl::local_context().parallel);
        b = a + a;
        b += a + a;
    }

    REQUIRE_DIRECT(!etl::local_context().parallel);

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("parallel_section/2", "[fast][parallel]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    REQUIRE_DIRECT(!etl::local_context().parallel);

    etl::local_context().parallel = true;

    PARALLEL_SECTION {
        REQUIRE_DIRECT(etl::local_context().parallel);
        b = a + a;
        b += a + a;

        etl::local_context().parallel = false;
    }

    REQUIRE_DIRECT(etl::local_context().parallel);

    etl::local_context().parallel = false;

    REQUIRE_EQUALS(b[0], 4.0);
}

TEMPLATE_TEST_CASE_2("parallel_section/3", "[fast][parallel]", Z, float, double) {
    REQUIRE_DIRECT(!etl::local_context().parallel);

    PARALLEL_SECTION {
        REQUIRE_DIRECT(etl::local_context().parallel);

        etl::local_context().parallel = false;

        PARALLEL_SECTION {
            REQUIRE_DIRECT(etl::local_context().parallel);
        }

        REQUIRE_DIRECT(!etl::local_context().parallel);
    }

    REQUIRE_DIRECT(!etl::local_context().parallel);
}
