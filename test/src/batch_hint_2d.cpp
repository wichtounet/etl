//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define ETL_RELAXED
#include "test.hpp"

TEMPLATE_TEST_CASE_2("batch_hint/A/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == input(b, i) * gamma(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == Z(42) + input(b, i) * gamma(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == Z(42) * input(b, i) * gamma(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == input(b, i) * gamma(i) + beta(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == Z(42) + (input(b, i) * gamma(i) + beta(i)));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/dyn/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == input(b, i) * gamma(i) + beta(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == gamma(i) * (input(b, i) - beta(i)));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3> input;
    etl::fast_matrix<Z, 2, 3> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(output(b, i) == Z(42) * gamma(i) * (input(b, i) - beta(i)));
        }
    }
}
