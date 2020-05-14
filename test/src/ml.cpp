//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("ml/sigmoid/backward/1", "[ml]", Z, double, float) {
    etl::dyn_vector<Z> x  = {-1.0, 2.0, 0.0, 0.5};
    etl::dyn_vector<Z> dy = {1.0, 2.0, 3.0, 0.5};

    etl::dyn_vector<Z> y;
    y = etl::sigmoid(x);

    etl::dyn_vector<Z> dx;
    dx = etl::ml::sigmoid_backward(y, dy);

    REQUIRE_EQUALS_APPROX(dx[0], Z(0.196611));
    REQUIRE_EQUALS_APPROX(dx[1], Z(0.209987));
    REQUIRE_EQUALS_APPROX(dx[2], Z(0.75));
    REQUIRE_EQUALS_APPROX(dx[3], Z(0.1175));
}

TEMPLATE_TEST_CASE_2("ml/relu/backward/1", "[ml]", Z, double, float) {
    etl::dyn_vector<Z> x  = {-1.0, 2.0, 0.0, 0.5};
    etl::dyn_vector<Z> dy = {1.0, 2.0, 3.0, 0.5};

    etl::dyn_vector<Z> y;
    y = etl::relu(x);

    etl::dyn_vector<Z> dx;
    dx = etl::ml::relu_backward(y, dy);

    REQUIRE_EQUALS_APPROX(dx[0], Z(0.0));
    REQUIRE_EQUALS_APPROX(dx[1], Z(2.0));
    REQUIRE_EQUALS_APPROX(dx[2], Z(0.0));
    REQUIRE_EQUALS_APPROX(dx[3], Z(0.5));
}

TEMPLATE_TEST_CASE_2("ml/cce/loss/1", "[ml]", Z, double, float) {
    etl::dyn_vector<Z> o(137);
    etl::dyn_vector<Z> l(137);

    for (size_t i = 0; i < 137; ++i) {
        o[i] = 0.1 * (i + 1);
        l[i] = (i + 1);
    }

    auto loss = etl::ml::cce_loss(o, l, Z(1.1));

    REQUIRE_EQUALS_APPROX(loss, Z(22055.71671));
}

TEMPLATE_TEST_CASE_2("ml/cce/error/1", "[ml]", Z, double, float) {
    etl::dyn_matrix<Z> o(137, 8);
    etl::dyn_matrix<Z> l(137, 8);

    for (size_t i = 0; i < 137 * 8; ++i) {
        o[i] = (i + 1);
        l[i] = ((i + 1) % 9);
    }

    auto error = etl::ml::cce_error(o, l, Z(1.0 / 137));

    REQUIRE_EQUALS_APPROX(error, Z(0.76642));
}

TEMPLATE_TEST_CASE_2("ml/cce/error/2", "[ml]", Z, double, float) {
    etl::dyn_matrix<Z> o(128, 9);
    etl::dyn_matrix<Z> l(128, 9);

    for (size_t i = 0; i < 128 * 9; ++i) {
        o[i] = (i + 1);
        l[i] = ((i + 1) % 11);
    }

    auto error = etl::ml::cce_error(o, l, Z(1.0 / 128));

    REQUIRE_EQUALS_APPROX(error, Z(0.71875));
}

TEMPLATE_TEST_CASE_2("ml/bce/loss/1", "[ml]", Z, double, float) {
    etl::dyn_vector<Z> o(137);
    etl::dyn_vector<Z> l(137);

    for (size_t i = 0; i < 137; ++i) {
        o[i] = 0.9 / (i + 1);
        l[i] = (i + 1);
    }

    auto loss = etl::ml::bce_loss(o, l, Z(1.1));

    REQUIRE_EQUALS_APPROX(loss, Z(-46962.205));
}

TEMPLATE_TEST_CASE_2("ml/bce/error/1", "[ml]", Z, double, float) {
    etl::dyn_matrix<Z> o(137, 8);
    etl::dyn_matrix<Z> l(137, 8);

    for (size_t i = 0; i < 137 * 8; ++i) {
        o[i] = (i + 1);
        l[i] = ((i + 1) % 9);
    }

    auto error = etl::ml::bce_error(o, l, Z(1.0 / 137));

    REQUIRE_EQUALS_APPROX(error, Z(4356.0f));
}

TEMPLATE_TEST_CASE_2("ml/bce/error/2", "[ml]", Z, double, float) {
    etl::dyn_matrix<Z> o(128, 9);
    etl::dyn_matrix<Z> l(128, 9);

    for (size_t i = 0; i < 128 * 9; ++i) {
        o[i] = (i + 1);
        l[i] = ((i + 1) % 11);
    }

    auto error = etl::ml::bce_error(o, l, Z(1.0 / 128));

    REQUIRE_EQUALS_APPROX(error, Z(5143.532));
}
