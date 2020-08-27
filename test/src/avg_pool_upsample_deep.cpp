//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/avg2/deep/1", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::dyn_matrix<Z, 3> input(5, 9, 9);
    input = etl::uniform_generator<Z>(g, -1000.0, 1000.0);

    etl::dyn_matrix<Z, 3> errors(5, 3, 3);
    errors = etl::uniform_generator<Z>(g, -1000.0, 1000.0);

    etl::dyn_matrix<Z, 3> output(5, 3, 3);
    output = etl::avg_pool_2d(input, 3, 3);

    etl::dyn_matrix<Z, 3> c1(5, 9, 9);
    etl::dyn_matrix<Z, 3> c2(5, 9, 9);

    c1 = etl::avg_pool_derivative_2d(input, output, 3, 3) >> etl::upsample_2d(errors, 3, 3);
    c2 = etl::avg_pool_upsample_2d(input, output, errors, 3, 3);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/avg2/deep/1", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 3, 8, 4> input;
    input = etl::uniform_generator<Z>(g, -1000.0, 1000.0);

    etl::fast_matrix<Z, 3, 4, 4> errors;
    errors = etl::uniform_generator<Z>(g, -1000.0, 1000.0);

    etl::fast_matrix<Z, 3, 4, 4> output;
    output = etl::avg_pool_2d<2, 1>(input);

    etl::fast_matrix<Z, 3, 8, 4> c1;
    etl::fast_matrix<Z, 3, 8, 4> c2;

    c1 = etl::avg_pool_derivative_2d<2, 1>(input, output) >> etl::upsample_2d<2, 1>(errors);
    c2 = etl::avg_pool_upsample_2d<2, 1>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps_etl));
}
