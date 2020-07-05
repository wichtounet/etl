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

TEMPLATE_TEST_CASE_2("batch_hint/A/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 2, 7, 9, 9> input;
    etl::fast_matrix<Z, 2, 7, 9, 9> output;

    gamma = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);

    output = batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 9; ++n) {
                    REQUIRE(output(b, i, m, n) == gamma(i) * input(b, i, m, n));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/6", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 2, 7, 11, 9> input;
    etl::fast_matrix<Z, 2, 7, 11, 9> output;

    gamma = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 9; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) + gamma(i) * input(b, i, m, n));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/7", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 2, 7, 11, 9> input;
    etl::fast_matrix<Z, 2, 7, 11, 9> output;

    gamma = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output -= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 9; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) - gamma(i) * input(b, i, m, n));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/8", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 2, 7, 11, 9> input;
    etl::fast_matrix<Z, 2, 7, 11, 9> output;

    gamma = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 9; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) * gamma(i) * input(b, i, m, n));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/9", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 2, 7, 11, 9> input;
    etl::fast_matrix<Z, 2, 7, 11, 9> output;

    gamma = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output /= batch_hint(gamma >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 9; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) / (gamma(i) * input(b, i, m, n)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/A/dyn/0", "[batch_hint]", Z, float, double) {
    etl::dyn_matrix<Z, 1> gamma(3);
    etl::dyn_matrix<Z, 4> input(2, 3, 2, 2);
    etl::dyn_matrix<Z, 4> output(2, 3, 2, 2);

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

TEMPLATE_TEST_CASE_2("batch_hint/A/dyn/1", "[batch_hint]", Z, float, double) {
    etl::dyn_matrix<Z, 1> gamma(3);
    etl::dyn_matrix<Z, 4> input(2, 3, 2, 2);
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

TEMPLATE_TEST_CASE_2("batch_hint/B/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

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
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == input(b, i, m, n) * Z(i + 1) + beta(i));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

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
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) + (input(b, i, m, n) * Z(i + 1) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 9, 7, 9, 11> input;
    etl::fast_matrix<Z, 9, 7, 9, 11> output;

    gamma = etl::sequence_generator(10.0);
    beta = etl::sequence_generator(20.0);
    input = etl::sequence_generator(1.0);
    output = 666;

    output = batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == (input(b, i, m, n) * gamma(i) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 9, 7, 9, 11> input;
    etl::fast_matrix<Z, 9, 7, 9, 11> output;

    gamma = etl::sequence_generator(10.0);
    beta = etl::sequence_generator(20.0);
    input = etl::sequence_generator(1.0);
    output = 666;

    output += batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(666) + (input(b, i, m, n) * gamma(i) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 9, 7, 9, 11> input;
    etl::fast_matrix<Z, 9, 7, 9, 11> output;

    gamma = etl::sequence_generator(10.0);
    beta = etl::sequence_generator(20.0);
    input = etl::sequence_generator(1.0);
    output = 666;

    output -= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(666) - (input(b, i, m, n) * gamma(i) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 9, 7, 9, 11> input;
    etl::fast_matrix<Z, 9, 7, 9, 11> output;

    gamma = etl::sequence_generator(10.0);
    beta = etl::sequence_generator(20.0);
    input = etl::sequence_generator(1.0);
    output = 666;

    output *= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(666) * (input(b, i, m, n) * gamma(i) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/B/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 9, 7, 9, 11> input;
    etl::fast_matrix<Z, 9, 7, 9, 11> output;

    gamma = etl::sequence_generator(10.0);
    beta = etl::sequence_generator(20.0);
    input = etl::sequence_generator(1.0);
    output = 666;

    output /= batch_hint((gamma >> input) + beta);

    for (size_t b = 0; b < 9; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(666) / (input(b, i, m, n) * gamma(i) + beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

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
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == gamma(i) * (input(b, i, m, n) - beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) + (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/dyn/0", "[batch_hint]", Z, float, double) {
    etl::dyn_matrix<Z, 1> gamma(3);
    etl::dyn_matrix<Z, 1> beta(3);
    etl::dyn_matrix<Z, 4> input(2, 3, 2, 2);
    etl::dyn_matrix<Z, 4> output(2, 3, 2, 2);

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) + (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/mixed/0", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint((gamma + gamma) >> input);

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == input(b, i, m, n) * (gamma(i) + gamma(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/mixed/1", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint((gamma + gamma) >> (input * 2.0));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == 2.0 * input(b, i, m, n) * (gamma(i) + gamma(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/mixed/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 3> gamma;
    etl::fast_matrix<Z, 3> beta;
    etl::fast_matrix<Z, 2, 3, 2, 2> input;
    etl::fast_matrix<Z, 2, 3, 2, 2> output;

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(((gamma - beta) >> input) + (gamma >> beta));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE(output(b, i, m, n) == (input(b, i, m, n) * (gamma(i) - beta(i))) + (gamma(i) * beta(i)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/dyn/0", "[batch_hint]", Z, float, double) {
    etl::dyn_matrix<Z, 1> gamma(3);
    etl::dyn_matrix<Z, 1> beta(3);
    etl::dyn_matrix<Z, 4> input(2, 3, 2, 2);
    etl::dyn_matrix<Z, 4> output(2, 3, 2, 2);

    gamma[0] = Z(1);
    gamma[1] = Z(2);
    gamma[2] = Z(3);

    beta[0] = Z(10);
    beta[1] = Z(20);
    beta[2] = Z(30);

    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint((2.0 * gamma) >> ((1.1 * input) - (1.2 * beta)));

    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t m = 0; m < 2; ++m) {
                for (size_t n = 0; n < 2; ++n) {
                    REQUIRE_EQUALS_APPROX(output(b, i, m, n), Z(42) + (2.0 * gamma(i) * ((1.1 * input(b, i, m, n)) - (1.2 * beta(i)))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/2", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output = batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/3", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output += batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) + (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/4", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output -= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) - (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/5", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output *= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) * (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_2("batch_hint/C/6", "[batch_hint]", Z, float, double) {
    etl::fast_matrix<Z, 7> gamma;
    etl::fast_matrix<Z, 7> beta;
    etl::fast_matrix<Z, 5, 7, 9, 11> input;
    etl::fast_matrix<Z, 5, 7, 9, 11> output;

    gamma = etl::sequence_generator(9.0);
    beta = etl::sequence_generator(3.0);
    input = etl::sequence_generator(1.0);
    output = 42;

    output /= batch_hint(gamma >> (input - beta));

    for (size_t b = 0; b < 5; ++b) {
        for (size_t i = 0; i < 7; ++i) {
            for (size_t m = 0; m < 9; ++m) {
                for (size_t n = 0; n < 11; ++n) {
                    REQUIRE(output(b, i, m, n) == Z(42) / (gamma(i) * (input(b, i, m, n) - beta(i))));
                }
            }
        }
    }
}
