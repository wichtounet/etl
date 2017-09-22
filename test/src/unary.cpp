//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("log/0", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log/1", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log/2", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {3.0, 2.0, 5.0, 1.0, 1.2, 2.2, 3.2, 4.2};

    etl::fast_matrix<Z, 2, 2, 2> d;

    d = log(a);

    REQUIRE_EQUALS_APPROX(d[0], std::log(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::log(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::log(Z(1.2)));
    REQUIRE_EQUALS_APPROX(d[5], std::log(Z(2.2)));
    REQUIRE_EQUALS_APPROX(d[6], std::log(Z(3.2)));
    REQUIRE_EQUALS_APPROX(d[7], std::log(Z(4.2)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/sqrt_1", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/sqrt_2", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/sqrt_3", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], std::sqrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invsqrt_1", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = invsqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invsqrt_2", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invsqrt_3", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::sqrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/cbrt_1", "fast_matrix::cbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/cbrt_2", "fast_matrix::cbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/cbrt_3", "fast_matrix::cbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invcbrt_1", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = invcbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invcbrt_2", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invcbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/invcbrt_3", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invcbrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/abs", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(a);

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 2.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sign", "fast_matrix::sign", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sign(a);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_unary", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(sign(a));

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_binary_1", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_binary_2", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(a) + a;

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("sigmoid/forward/0", "[fast][ml]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sigmoid(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::logistic_sigmoid(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::logistic_sigmoid(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("sigmoid/forward/1", "[fast][ml]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    // Inplace should work!
    a = sigmoid(a);

    REQUIRE_EQUALS_APPROX(a[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(a[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(a[2], etl::math::logistic_sigmoid(Z(0.0)));
    REQUIRE_EQUALS_APPROX(a[3], etl::math::logistic_sigmoid(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/softplus", "fast_matrix::softplus", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = softplus(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::softplus(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::softplus(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::softplus(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::softplus(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("exp/0", "[exp]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::exp(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("exp/1", "[exp]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.1, 6.1};

    etl::fast_matrix<Z, 2, 2, 2> d;
    d = exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::exp(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::exp(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[5], std::exp(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[6], std::exp(Z(5.1)));
    REQUIRE_EQUALS_APPROX(d[7], std::exp(Z(6.1)));
}

constexpr bool binary(double a) {
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2("fast_matrix/bernoulli", "fast_matrix::bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = etl::bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}

TEMPLATE_TEST_CASE_2("fast_matrix/r_bernoulli", "fast_matrix::r_bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = etl::r_bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}
