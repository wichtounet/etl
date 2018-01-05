//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

/// sequence_generator

TEMPLATE_TEST_CASE_2("sequence/fast_vector_1", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::sequence_generator<Z>();

    REQUIRE_EQUALS(b[0], 0.0);
    REQUIRE_EQUALS(b[1], 1.0);
    REQUIRE_EQUALS(b[2], 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_vector_2", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::sequence_generator<Z>(99);

    REQUIRE_EQUALS(b[0], 99.0);
    REQUIRE_EQUALS(b[1], 100.0);
    REQUIRE_EQUALS(b[2], 101.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_vector_3", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::sequence_generator<Z>(99);

    REQUIRE_EQUALS(b[0], 99.0);
    REQUIRE_EQUALS(b[1], 100.0);
    REQUIRE_EQUALS(b[2], 101.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_vector_4", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = 0.5 * etl::sequence_generator<Z>(99.0);

    REQUIRE_EQUALS(b[0], 49.5);
    REQUIRE_EQUALS(b[1], 50.0);
    REQUIRE_EQUALS(b[2], 50.5);
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_1", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = etl::sequence_generator<Z>();

    REQUIRE_EQUALS(b(0, 0), 0.0);
    REQUIRE_EQUALS(b(0, 1), 1.0);
    REQUIRE_EQUALS(b(1, 0), 2.0);
    REQUIRE_EQUALS(b(1, 1), 3.0);
    REQUIRE_EQUALS(b(2, 0), 4.0);
    REQUIRE_EQUALS(b(2, 1), 5.0);
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_2", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = 0.1 * etl::sequence_generator<Z>();

    REQUIRE_DIRECT(etl::decay_traits<decltype(0.1 * etl::sequence_generator())>::is_generator);

    REQUIRE_EQUALS_APPROX(b(0, 0), 0.0);
    REQUIRE_EQUALS_APPROX(b(0, 1), 0.1);
    REQUIRE_EQUALS_APPROX(b(1, 0), 0.2);
    REQUIRE_EQUALS_APPROX(b(1, 1), 0.3);
    REQUIRE_EQUALS_APPROX(b(2, 0), 0.4);
    REQUIRE_EQUALS_APPROX(b(2, 1), 0.5);
}

TEMPLATE_TEST_CASE_2("sequence/fast_matrix_3", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b(1.0);

    b = 0.1 * etl::sequence_generator<Z>() + b;

    REQUIRE_EQUALS_APPROX(b(0, 0), 1.0);
    REQUIRE_EQUALS_APPROX(b(0, 1), 1.1);
    REQUIRE_EQUALS_APPROX(b(1, 0), 1.2);
    REQUIRE_EQUALS_APPROX(b(1, 1), 1.3);
    REQUIRE_EQUALS_APPROX(b(2, 0), 1.4);
    REQUIRE_EQUALS_APPROX(b(2, 1), 1.5);
}

TEMPLATE_TEST_CASE_2("sequence/dyn_vector_1", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::sequence_generator<Z>();

    REQUIRE_EQUALS(b[0], 0.0);
    REQUIRE_EQUALS(b[1], 1.0);
    REQUIRE_EQUALS(b[2], 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/dyn_vector_2", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::sequence_generator<Z>();

    REQUIRE_EQUALS(b[0], 0.0);
    REQUIRE_EQUALS(b[1], 1.0);
    REQUIRE_EQUALS(b[2], 2.0);
}

TEMPLATE_TEST_CASE_2("sequence/dyn_matrix_1", "generator", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::sequence_generator<Z>() >> 2.0;

    REQUIRE_EQUALS(b(0, 0), 0.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(1, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1), 6.0);
    REQUIRE_EQUALS(b(2, 0), 8.0);
    REQUIRE_EQUALS(b(2, 1), 10.0);
}

/// normal_generator

//Simply ensures that it compiles

TEMPLATE_TEST_CASE_2("normal/fast_vector_1", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("normal/fast_matrix_1", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = etl::normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("normal/dyn_vector_1", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::normal_generator<Z>() >> etl::normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("normal/dyn_matrix_1", "generator", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 5);

    b = etl::normal_generator<Z>();
}

/// truncated_normal_generator

//Simply ensures that it compiles

TEMPLATE_TEST_CASE_2("truncated_normal/fast_vector_1", "generator", Z, float, double) {
    etl::fast_vector<Z, 3> b;

    b = etl::truncated_normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("truncated_normal/fast_matrix_1", "generator", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> b;

    b = etl::truncated_normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("truncated_normal/dyn_vector_1", "generator", Z, float, double) {
    etl::dyn_vector<Z> b(3);

    b = etl::truncated_normal_generator<Z>();
}

TEMPLATE_TEST_CASE_2("truncated_normal/dyn_matrix_1", "generator", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::truncated_normal_generator<Z>();
}

/// uniform_generator

TEMPLATE_TEST_CASE_2("generators/uniform/1", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::uniform_generator<Z>(-2.0, +2.0);

    for (auto value : b) {
        REQUIRE_DIRECT(value >= -2.0);
        REQUIRE_DIRECT(value <= +2.0);
    }
}

TEMPLATE_TEST_CASE_2("generators/uniform/2", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::uniform_generator<Z>(5.5, 8.0);

    for (auto value : b) {
        REQUIRE_DIRECT(value >= 5.5);
        REQUIRE_DIRECT(value <= 8.0);
    }
}

TEMPLATE_TEST_CASE_2("generators/uniform/3", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::uniform_generator<Z>(1.0, 2.0) >> 4.0;

    for (auto value : b) {
        REQUIRE_DIRECT(value >= 4.0);
        REQUIRE_DIRECT(value <= 8.0);
    }
}

/// dropout_mask

TEMPLATE_TEST_CASE_2("generators/dropout_mask/1", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::dropout_mask<Z>(0.2);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }
}

TEMPLATE_TEST_CASE_2("generators/dropout_mask/2", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::dropout_mask<Z>(0.5);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }
}

/// state_dropout_mask

TEMPLATE_TEST_CASE_2("generators/state_dropout_mask/1", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::state_dropout_mask<Z>(0.2);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }
}

TEMPLATE_TEST_CASE_2("generators/state_dropout_mask/2", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::state_dropout_mask<Z>(0.5);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }
}

TEMPLATE_TEST_CASE_2("generators/state_dropout_mask/3", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    auto dropout = etl::state_dropout_mask<Z>(0.5);

    b = dropout;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }

    b = dropout;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0));
    }
}

TEMPLATE_TEST_CASE_2("generators/state_dropout_mask/4", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    auto dropout = etl::state_dropout_mask<Z>(0.5);

    b = (dropout * 0.5f) + 1.0f;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(1.0) || value == Z(1.5));
    }
}

/// inverted_dropout_mask

TEMPLATE_TEST_CASE_2("generators/inverted_dropout_mask/1", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::inverted_dropout_mask<Z>(0.2);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0 / 0.8));
    }
}

TEMPLATE_TEST_CASE_2("generators/inverted_dropout_mask/2", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::inverted_dropout_mask<Z>(0.5);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.0 / 0.5));
    }
}

TEMPLATE_TEST_CASE_2("generators/inverted_dropout_mask/3", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::inverted_dropout_mask<Z>(0.5) >> 2.0;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(2.0 / 0.5));
    }
}

/// state_inverted_dropout_mask

TEMPLATE_TEST_CASE_2("generators/state_inverted_dropout_mask/1", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::state_inverted_dropout_mask<Z>(0.2);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(1.25));
    }
}

TEMPLATE_TEST_CASE_2("generators/state_inverted_dropout_mask/2", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    b = etl::state_inverted_dropout_mask<Z>(0.5);

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(2.0));
    }
}

TEMPLATE_TEST_CASE_2("generators/state_inverted_dropout_mask/3", "uniform", Z, float, double) {
    etl::dyn_matrix<Z> b(3, 2);

    auto dropout = etl::state_inverted_dropout_mask<Z>(0.5);

    b = dropout;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(2.0));
    }

    b = dropout;

    for (auto value : b) {
        REQUIRE_DIRECT(value == Z(0.0) || value == Z(2.0));
    }
}
