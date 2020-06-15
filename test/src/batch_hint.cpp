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
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == input(b, i, m, n) * Z(i + 1));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == 42 + input(b, i, m, n) * Z(i + 1));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output -= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == 42 - input(b, i, m, n) * Z(i + 1));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == 42 * input(b, i, m, n) * Z(i + 1));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output /= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == 42 / (input(b, i, m, n) * Z(i + 1)));
                }
            }
        }
    }
}
