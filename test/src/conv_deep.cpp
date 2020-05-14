//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

// convolution_deep_full

TEMPLATE_TEST_CASE_2("convolution_3d/full_1", "convolution_deep_full", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    etl::fast_matrix<Z, 2, 2, 2> b = {2.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0};
    etl::fast_matrix<Z, 2, 3, 3> c;

    *etl::conv_2d_full_deep(a, b, c);

    REQUIRE_EQUALS_APPROX_E(c(0, 0, 0), 2.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 1), 5.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 2), 2.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 1, 0), 6.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 1), 8.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 2), 3.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 2, 0), 1.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2, 1), 2.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2, 2), 1.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0, 0), 5.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 0, 1), 16.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 0, 2), 12.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 1, 0), 12.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1, 1), 38.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1, 2), 28.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 2, 0), 7.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2, 1), 22.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2, 2), 16.0, base_eps / 10);
}

TEMPLATE_TEST_CASE_2("convolution_4d/full_1", "convolution_deep_full", Z, float, double) {
    etl::fast_matrix<Z, 1, 2, 2, 2> a = {1.0, 2.0, 3.0, 2.0, 5.0, 6.0, 7.0, 8.0};
    etl::fast_matrix<Z, 1, 2, 2, 2> b = {2.0, 1.0, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0};
    etl::fast_matrix<Z, 1, 2, 3, 3> c;

    c = etl::conv_2d_full_deep(a, b);

    REQUIRE_EQUALS_APPROX_E(c(0, 0, 0, 0), 2.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 0, 1), 5.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 0, 2), 2.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 0, 1, 0), 6.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 1, 1), 8.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 1, 2), 3.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 0, 2, 0), 1.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 2, 1), 2.5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 0, 2, 2), 1.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 1, 0, 0), 5.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 0, 1), 16.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 0, 2), 12.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 1, 1, 0), 12.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 1, 1), 38.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 1, 2), 28.0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(0, 1, 2, 0), 7.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 2, 1), 22.0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1, 2, 2), 16.0, base_eps / 10);
}
