//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("max_pool_derivative/fast/0", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d<2, 2>(input, output);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 0.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/fast/1", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({6.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 14.0, 11.0, 12.0, 13.0, 14.0, 16.0, 16.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d<2, 2>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d<2, 2>(input, output);

    REQUIRE(result(0, 0) == 1.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 0.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 1.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 0.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 0.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/fast/2", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::max_pool_2d<2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d<2, 2, 1, 1>(input, output);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 1.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 1.0);
    REQUIRE(result(2, 2) == 1.0);
    REQUIRE(result(2, 3) == 1.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/fast/3", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::max_pool_2d<2, 2, 2, 2, 1, 1>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d<2, 2, 2, 2, 1, 1>(input, output);

    REQUIRE(result(0, 0) == 1.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 1.0);
    REQUIRE(result(0, 3) == 1.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 0.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 0.0);

    REQUIRE(result(2, 0) == 1.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 1.0);
    REQUIRE(result(2, 3) == 1.0);

    REQUIRE(result(3, 0) == 1.0);
    REQUIRE(result(3, 1) == 0.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/fast/4", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 4, 4> output;
    output = etl::max_pool_2d<2, 2, 2, 2, 2, 2>(input);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d<2, 2, 2, 2, 2, 2>(input, output);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 0.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/dyn/0", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d(input, 2, 2);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d(input, output, 2, 2);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 0.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/dyn/1", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({6.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 14.0, 11.0, 12.0, 13.0, 14.0, 16.0, 16.0});

    etl::fast_matrix<Z, 2, 2> output;
    output = etl::max_pool_2d(input, 2, 2);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d(input, output, 2, 2);

    REQUIRE(result(0, 0) == 1.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 0.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 1.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 0.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 0.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/dyn/2", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::max_pool_2d(input, 2, 2, 1, 1);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d(input, output, 2, 2, 1, 1);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 1.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 1.0);
    REQUIRE(result(2, 2) == 1.0);
    REQUIRE(result(2, 3) == 1.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/dyn/3", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 3, 3> output;
    output = etl::max_pool_2d(input, 2, 2, 2, 2, 1, 1);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d(input, output, 2, 2, 2, 2, 1, 1);

    REQUIRE(result(0, 0) == 1.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 1.0);
    REQUIRE(result(0, 3) == 1.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 0.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 0.0);

    REQUIRE(result(2, 0) == 1.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 1.0);
    REQUIRE(result(2, 3) == 1.0);

    REQUIRE(result(3, 0) == 1.0);
    REQUIRE(result(3, 1) == 0.0);
    REQUIRE(result(3, 2) == 1.0);
    REQUIRE(result(3, 3) == 1.0);
}

TEMPLATE_TEST_CASE_2("max_pool_derivative/dyn/4", "[pooling]", Z, float, double) {
    std::random_device rd;
    etl::random_engine g(rd());

    etl::fast_matrix<Z, 4, 4> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});

    etl::fast_matrix<Z, 4, 4> output;
    output = etl::max_pool_2d(input, 2, 2, 2, 2, 2, 2);

    etl::fast_matrix<Z, 4, 4> result;
    result = etl::max_pool_derivative_2d(input, output, 2, 2, 2, 2, 2, 2);

    REQUIRE(result(0, 0) == 0.0);
    REQUIRE(result(0, 1) == 0.0);
    REQUIRE(result(0, 2) == 0.0);
    REQUIRE(result(0, 3) == 0.0);

    REQUIRE(result(1, 0) == 0.0);
    REQUIRE(result(1, 1) == 1.0);
    REQUIRE(result(1, 2) == 0.0);
    REQUIRE(result(1, 3) == 1.0);

    REQUIRE(result(2, 0) == 0.0);
    REQUIRE(result(2, 1) == 0.0);
    REQUIRE(result(2, 2) == 0.0);
    REQUIRE(result(2, 3) == 0.0);

    REQUIRE(result(3, 0) == 0.0);
    REQUIRE(result(3, 1) == 1.0);
    REQUIRE(result(3, 2) == 0.0);
    REQUIRE(result(3, 3) == 1.0);
}
