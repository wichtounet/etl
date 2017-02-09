//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

// Trigonometric tests

TEMPLATE_TEST_CASE_2("trigo/tan", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::tan(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::tan(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/sin", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::sin(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::sin(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/cos", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::cos(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::cos(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/tanh", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::tanh(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::tanh(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/sinh", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::sinh(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::sinh(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("trigo/cosh", "dyn_matrix::dyn_matrix(T)", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1.0, 2.0, -1.0, -0.5, 0.6, 0.1));
    etl::dyn_matrix<Z> b(etl::cosh(a));

    for (std::size_t i = 0; i < b.size(); ++i) {
        REQUIRE_EQUALS_APPROX(b[i], std::cosh(a[i]));
    }
}
