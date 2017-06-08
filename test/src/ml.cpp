//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
