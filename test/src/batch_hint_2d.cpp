//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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

TEMPLATE_TEST_CASE_2("batch_hint/A/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 109> gamma;
    etl::fast_matrix<Z, 2, 109> input;
    etl::fast_matrix<Z, 2, 109> output;

    gamma = etl::sequence_generator(2.0);
    input = etl::sequence_generator(1.0);

    output = batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 109; ++i) {
            REQUIRE(output(b, i) == gamma(i) * input(b, i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/4", "[batch_hint]", Z, float, double) {
    constexpr size_t N = 4 * 16 + 2 * 16 + 16 + 4 + 3 + 2 + 1;

    etl::fast_matrix<Z, N> gamma;
    etl::fast_matrix<Z, 3, N> input;
    etl::fast_matrix<Z, 3, N> output;

    gamma = etl::sequence_generator(2.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> input);

    for (size_t b = 0; b < 3; ++b) {
        for (size_t i = 0; i < 109; ++i) {
            REQUIRE(output(b, i) == Z(42) + gamma(i) * input(b, i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 109> gamma;
    etl::fast_matrix<Z, 3, 109> input;
    etl::fast_matrix<Z, 3, 109> output;

    gamma = etl::sequence_generator(2.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output -= batch_hint(gamma >> input);

    for (size_t b = 0; b < 3; ++b) {
        for (size_t i = 0; i < 109; ++i) {
            REQUIRE(output(b, i) == Z(42) - gamma(i) * input(b, i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/6", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 109> gamma;
    etl::fast_matrix<Z, 3, 109> input;
    etl::fast_matrix<Z, 3, 109> output;

    gamma = etl::sequence_generator(2.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> input);

    for (size_t b = 0; b < 3; ++b) {
        for (size_t i = 0; i < 109; ++i) {
            REQUIRE(output(b, i) == Z(42) * gamma(i) * input(b, i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/7", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 109> gamma;
    etl::fast_matrix<Z, 3, 109> input;
    etl::fast_matrix<Z, 3, 109> output;

    gamma = etl::sequence_generator(2.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output /= batch_hint(gamma >> input);

    for (size_t b = 0; b < 3; ++b) {
        for (size_t i = 0; i < 109; ++i) {
            REQUIRE(output(b, i) == Z(42) / (gamma(i) * input(b, i)));
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

TEMPLATE_TEST_CASE_2("batch_hint/B/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 99> gamma;
    etl::fast_matrix<Z, 99> beta;
    etl::fast_matrix<Z, 9, 99> input;
    etl::fast_matrix<Z, 9, 99> output;

    gamma  = etl::sequence_generator(1.0);
    beta   = etl::sequence_generator(2.0);
    input  = etl::sequence_generator(3.0);
    output = 42;

    output = batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 99; ++i) {
            REQUIRE(output(b, i) == input(b, i) * gamma(i) + beta(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 99> gamma;
    etl::fast_matrix<Z, 99> beta;
    etl::fast_matrix<Z, 9, 99> input;
    etl::fast_matrix<Z, 9, 99> output;

    gamma  = etl::sequence_generator(1.0);
    beta   = etl::sequence_generator(2.0);
    input  = etl::sequence_generator(3.0);
    output = 42;

    output += batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 99; ++i) {
            REQUIRE(output(b, i) == Z(42) + input(b, i) * gamma(i) + beta(i));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 99> gamma;
    etl::fast_matrix<Z, 99> beta;
    etl::fast_matrix<Z, 9, 99> input;
    etl::fast_matrix<Z, 9, 99> output;

    gamma  = etl::sequence_generator(1.0);
    beta   = etl::sequence_generator(2.0);
    input  = etl::sequence_generator(3.0);
    output = 42;

    output *= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 99; ++i) {
            REQUIRE(output(b, i) == Z(42) * (input(b, i) * gamma(i) + beta(i)));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 99> gamma;
    etl::fast_matrix<Z, 99> beta;
    etl::fast_matrix<Z, 9, 99> input;
    etl::fast_matrix<Z, 9, 99> output;

    gamma  = etl::sequence_generator(1.0);
    beta   = etl::sequence_generator(2.0);
    input  = etl::sequence_generator(3.0);
    output = 42;

    output -= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 99; ++i) {
            REQUIRE(output(b, i) == Z(42) - (input(b, i) * gamma(i) + beta(i)));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/6", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 99> gamma;
    etl::fast_matrix<Z, 99> beta;
    etl::fast_matrix<Z, 9, 99> input;
    etl::fast_matrix<Z, 9, 99> output;

    gamma  = etl::sequence_generator(1.0);
    beta   = etl::sequence_generator(2.0);
    input  = etl::sequence_generator(3.0);
    output = 42;

    output /= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 99; ++i) {
            REQUIRE(output(b, i) == Z(42) / (input(b, i) * gamma(i) + beta(i)));
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

TEMPLATE_TEST_CASE_2("batch_hint/C/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 97> gamma;
    etl::fast_matrix<Z, 97> beta;
    etl::fast_matrix<Z, 7, 97> input;
    etl::fast_matrix<Z, 7, 97> output;

    gamma = etl::sequence_generator(1.0);
    beta = etl::sequence_generator(1.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 97; ++i) {
            REQUIRE(output(b, i) == gamma(i) * (input(b, i) - beta(i)));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 97> gamma;
    etl::fast_matrix<Z, 97> beta;
    etl::fast_matrix<Z, 7, 97> input;
    etl::fast_matrix<Z, 7, 97> output;

    gamma = etl::sequence_generator(1.0);
    beta = etl::sequence_generator(1.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 97; ++i) {
            REQUIRE(output(b, i) == Z(42) + (gamma(i) * (input(b, i) - beta(i))));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 97> gamma;
    etl::fast_matrix<Z, 97> beta;
    etl::fast_matrix<Z, 7, 97> input;
    etl::fast_matrix<Z, 7, 97> output;

    gamma = etl::sequence_generator(1.0);
    beta = etl::sequence_generator(1.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output -= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 97; ++i) {
            REQUIRE(output(b, i) == Z(42) - (gamma(i) * (input(b, i) - beta(i))));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 97> gamma;
    etl::fast_matrix<Z, 97> beta;
    etl::fast_matrix<Z, 7, 97> input;
    etl::fast_matrix<Z, 7, 97> output;

    gamma = etl::sequence_generator(1.0);
    beta = etl::sequence_generator(1.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 97; ++i) {
            REQUIRE(output(b, i) == Z(42) * (gamma(i) * (input(b, i) - beta(i))));
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/6", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 97> gamma;
    etl::fast_matrix<Z, 97> beta;
    etl::fast_matrix<Z, 7, 97> input;
    etl::fast_matrix<Z, 7, 97> output;

    gamma = etl::sequence_generator(1.0);
    beta = etl::sequence_generator(1.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output /= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 7; ++b) {
        for (size_t i = 0; i < 97; ++i) {
            REQUIRE(output(b, i) == Z(42) / (gamma(i) * (input(b, i) - beta(i))));
        }
    }
}
