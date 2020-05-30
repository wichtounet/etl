//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

// These are simply compilation tests to avoid regression in noise functions

TEMPLATE_TEST_CASE_2("logistic_noise/0", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    d = logistic_noise(a);
}

TEMPLATE_TEST_CASE_2("logistic_noise/1", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    etl::random_engine g(666);
    d = logistic_noise(g, a);
}

TEMPLATE_TEST_CASE_2("logistic_noise/2", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    d = state_logistic_noise(a);
}

TEMPLATE_TEST_CASE_2("logistic_noise/3", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    etl::random_engine g(666);
    d = state_logistic_noise(g, a);
}

TEMPLATE_TEST_CASE_2("logistic_noise/4", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    auto states = std::make_shared<void*>();
    d = state_logistic_noise(a, states);
}

TEMPLATE_TEST_CASE_2("logistic_noise/5", "[logistic_noise]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> d;

    auto states = std::make_shared<void*>();
    etl::random_engine g(666);
    d = state_logistic_noise(g, a, states);
}
