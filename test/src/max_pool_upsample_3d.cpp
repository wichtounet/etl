//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max3/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> input(2, 4, 4);
    input = etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> errors(2, 2, 2);
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> output(2, 2, 2);
    output = etl::max_pool_3d(input, 1, 2, 2);

    etl::dyn_matrix<Z, 3> c1(2, 4, 4);
    etl::dyn_matrix<Z, 3> result(2, 4, 4);

    c1 = etl::max_pool_derivative_3d(input, output, 1, 2, 2) >> etl::upsample_3d(errors, 1, 2, 2);
    result = etl::max_pool_upsample_3d(input, output, errors, 1, 2, 2);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max3/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> input(2, 6, 9);
    input = etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> errors(1, 3, 3);
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> output(1, 3, 3);
    output = etl::max_pool_3d(input, 2, 2, 3);

    etl::dyn_matrix<Z, 3> c1(2, 6, 9);
    etl::dyn_matrix<Z, 3> result(2, 6, 9);

    c1 = etl::max_pool_derivative_3d(input, output, 2, 2, 3) >> etl::upsample_3d(errors, 2, 2, 3);
    result = etl::max_pool_upsample_3d(input, output, errors, 2, 2, 3);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max3/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> input(3, 6, 9);
    input = etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> errors(1, 6, 3);
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 3> output(1, 6, 3);
    output = etl::max_pool_3d(input, 3, 1, 3);

    etl::dyn_matrix<Z, 3> c1(3, 6, 9);
    etl::dyn_matrix<Z, 3> result(3, 6, 9);

    c1 = etl::max_pool_derivative_3d(input, output, 3, 1, 3) >> etl::upsample_3d(errors, 3, 1, 3);
    result = etl::max_pool_upsample_3d(input, output, errors, 3, 1, 3);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/dyn/max3/deep/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 4> input(4, 3, 6, 9);
    input = etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 4> errors(4, 1, 6, 3);
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 4> output(4, 1, 6, 3);
    output = etl::max_pool_3d(input, 3, 1, 3);

    etl::dyn_matrix<Z, 4> c1(4, 3, 6, 9);
    etl::dyn_matrix<Z, 4> result(4, 3, 6, 9);

    c1 = etl::max_pool_derivative_3d(input, output, 3, 1, 3) >> etl::upsample_3d(errors, 3, 1, 3);
    result = etl::max_pool_upsample_3d(input, output, errors, 3, 1, 3);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> input;
    input = etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 1, 2, 2> errors;
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 1, 2, 2> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 2, 4, 4> c1;
    etl::fast_matrix<Z, 2, 4, 4> result;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    result = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 8, 8> input;
    input = etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 4, 4> errors;
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 4, 4> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 4, 8, 8> c1;
    etl::fast_matrix<Z, 4, 8, 8> result;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    result = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/max3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 8, 8> input;
    input = etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 8, 8> errors;
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 8, 8> output;
    output = etl::max_pool_3d<2, 1, 1>(input);

    etl::fast_matrix<Z, 4, 8, 8> c1;
    etl::fast_matrix<Z, 4, 8, 8> result;

    c1 = etl::max_pool_derivative_3d<2, 1, 1>(input, output) >> etl::upsample_3d<2, 1, 1>(errors);
    result = etl::max_pool_upsample_3d<2, 1, 1>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("pool_upsample/deep/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 4, 4> input;
    input = etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 1, 2, 2> errors;
    errors = 100.0 * etl::sequence_generator<Z>(1.0);

    etl::fast_matrix<Z, 2, 1, 2, 2> output;
    output = etl::max_pool_3d<2, 2, 2>(input);

    etl::fast_matrix<Z, 2, 2, 4, 4> c1;
    etl::fast_matrix<Z, 2, 2, 4, 4> result;

    c1 = etl::max_pool_derivative_3d<2, 2, 2>(input, output) >> etl::upsample_3d<2, 2, 2>(errors);
    result = etl::max_pool_upsample_3d<2, 2, 2>(input, output, errors);

    REQUIRE_DIRECT(approx_equals(c1, result, base_eps_etl));
}
