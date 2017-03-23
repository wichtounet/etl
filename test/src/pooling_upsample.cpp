//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pool_upsample/max2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 2> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 4, 4> c1;
    etl::fast_matrix<Z, 4, 4> c2;

    c1 = etl::max_pool_derivative_2d<2, 2>(input, output) >> etl::upsample_2d<2, 2>(errors);
    c2 = etl::max_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 8, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 4, 2> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 4, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 8, 4> c1;
    etl::fast_matrix<Z, 8, 4> c2;

    c1 = etl::max_pool_derivative_2d<2, 2>(input, output) >> etl::upsample_2d<2, 2>(errors);
    c2 = etl::max_pool_upsample_2d<2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 8, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 4, 4> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 4, 4> output;
    output = etl::max_pool_2d<2, 1>(input);

    etl::fast_matrix<Z, 8, 4> c1;
    etl::fast_matrix<Z, 8, 4> c2;

    c1 = etl::max_pool_derivative_2d<2, 1>(input, output) >> etl::upsample_2d<2, 1>(errors);
    c2 = etl::max_pool_upsample_2d<2, 1>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max2/deep/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 8, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 3, 4, 4> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 3, 4, 4> output;
    output = etl::max_pool_2d<2, 1>(input);

    etl::fast_matrix<Z, 3, 8, 4> c1;
    etl::fast_matrix<Z, 3, 8, 4> c2;

    c1 = etl::max_pool_derivative_2d<2, 1>(input, output) >> etl::upsample_2d<2, 1>(errors);
    c2 = etl::max_pool_upsample_2d<2, 1>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 1, 2, 2> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 1, 2, 2> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 2, 4, 4> c1;
    etl::fast_matrix<Z, 2, 4, 4> c2;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    c2 = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 8, 8> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 4, 4> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 4, 4> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 4, 8, 8> c1;
    etl::fast_matrix<Z, 4, 8, 8> c2;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    c2 = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 8, 8> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 8, 8> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 8, 8> output;
    output = etl::max_pool_3d<2, 1, 1>(input);

    etl::fast_matrix<Z, 4, 8, 8> c1;
    etl::fast_matrix<Z, 4, 8, 8> c2;

    c1 = etl::max_pool_derivative_3d<2, 1, 1>(input, output) >> etl::upsample_3d<2, 1, 1>(errors);
    c2 = etl::max_pool_upsample_3d<2, 1, 1>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}

TEMPLATE_TEST_CASE_2("pool_upsample/deep/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 4, 4> input;
    input = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 1, 2, 2> errors;
    errors = etl::uniform_generator<Z>(-1000.0, 1000.0);

    etl::fast_matrix<Z, 2, 1, 2, 2> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 2, 2, 4, 4> c1;
    etl::fast_matrix<Z, 2, 2, 4, 4> c2;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    c2 = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, c2, base_eps));
}
