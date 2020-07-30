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
    etl::dyn_matrix<Z> o(137, 1);
    etl::dyn_matrix<Z> l(137, 1);

    for (size_t i = 0; i < 137 * 1; ++i) {
        o[i] = 0.1 * (i + 1);
        l[i] = (i + 1);
    }

    auto loss = etl::ml::cce_loss(o, l, Z(1.1));
    REQUIRE_EQUALS_APPROX(loss, Z(22055.71671));

    auto both = etl::ml::cce(o, l, Z(1.1), Z(1.0 / 137));
    REQUIRE_EQUALS_APPROX(both.first, Z(loss));
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

    auto both = etl::ml::cce(o, l, Z(1.1), Z(1.0 / 137));
    REQUIRE_EQUALS_APPROX(both.second, Z(error));
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

    auto both = etl::ml::cce(o, l, Z(1.1), Z(1.0 / 128));
    REQUIRE_EQUALS_APPROX(both.second, Z(error));
}

TEMPLATE_TEST_CASE_2("ml/bce/loss/1", "[ml]", Z, double, float) {
    etl::dyn_matrix<Z> o(137, 1);
    etl::dyn_matrix<Z> l(137, 1);

    for (size_t i = 0; i < 137 * 1; ++i) {
        o[i] = 0.9 / (i + 1);
        l[i] = (i + 1);
    }

    auto loss = etl::ml::bce_loss(o, l, Z(1.1));
    REQUIRE_EQUALS_APPROX(loss, Z(-46962.205));

    auto both = etl::ml::bce(o, l, Z(1.1), Z(1.1));
    REQUIRE_EQUALS_APPROX(both.first, Z(loss));
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

    auto both = etl::ml::bce(o, l, Z(1.1), Z(1.0 / 137));
    REQUIRE_EQUALS_APPROX(both.second, Z(error));
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

    auto both = etl::ml::bce(o, l, Z(1.1), Z(1.0 / 128));
    REQUIRE_EQUALS_APPROX(both.second, Z(error));
}

TEMPLATE_TEST_CASE_2("ml/pool/max/0", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::max_pool_forward<2, 2, 2, 2, 0, 0>(a);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("ml/pool/max/1", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::max_pool_forward<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("ml/pool/max/2", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::max_pool_forward(a, 2, 2, 2, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("ml/pool/max/3", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::max_pool_forward(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("ml/pool/avg/0", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::avg_pool_forward<2, 2, 2, 2, 0, 0>(a);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("ml/pool/avg/1", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::avg_pool_forward<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("ml/pool/avg/2", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::avg_pool_forward(a, 2, 2, 2, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("ml/pool/avg/3", "[ml]", Z, double, float) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;

    b = etl::ml::avg_pool_forward(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}
