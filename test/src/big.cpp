//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

// These tests are made to test operations using enough operations
// to make sure thresholds are reached

TEMPLATE_TEST_CASE_2("big/add", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);

    c = a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(a[i] + b[i]));
    }
}

TEMPLATE_TEST_CASE_2("big/sub", "[big][sub]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);

    c = a - b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(a[i] - b[i]));
    }
}

TEMPLATE_TEST_CASE_2("big/mul", "[big][sub]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);

    c = a >> b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(a[i] * b[i]));
    }
}

TEMPLATE_TEST_CASE_2("big/compound/add", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 120.0;

    c += a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], Z(120.0) + a[i] + b[i], base_eps * 10);
    }
}

TEMPLATE_TEST_CASE_2("big/compound/sub", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 1200.0;

    c -= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0 - (a[i] + b[i])));
    }
}

TEMPLATE_TEST_CASE_2("big/compound/mul", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 1200.0;

    c *= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0 * (a[i] + b[i])));
    }
}

TEMPLATE_TEST_CASE_2("big/compound/div", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> c(etl::parallel_threshold + 100, 1UL);

    a = etl::uniform_generator(1000.0, 5000.0);
    b = etl::uniform_generator(1000.0, 5000.0);
    c = 1200.0;

    c /= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0 / (a[i] + b[i])));
    }
}

TEMPLATE_TEST_CASE_2("big/sum/1", "[big][sum]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::sum_parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::sum_parallel_threshold + 100, 1UL);

    a = 1.0;
    b = 2.5;

    REQUIRE_EQUALS(etl::sum(a), 1.0 * (etl::sum_parallel_threshold + 100));
    REQUIRE_EQUALS(etl::sum(b), 2.5 * (etl::sum_parallel_threshold + 100));
}

TEMPLATE_TEST_CASE_2("big/sum/2", "[big][sum]", Z, double, float) {
    etl::dyn_matrix<Z> a(etl::sum_parallel_threshold + 100, 1UL);
    etl::dyn_matrix<Z> b(etl::sum_parallel_threshold + 100, 1UL);

    a = 1.0;
    b = 2.5;

    REQUIRE_EQUALS(etl::asum(a), 1.0 * (etl::sum_parallel_threshold + 100));
    REQUIRE_EQUALS(etl::asum(b), 2.5 * (etl::sum_parallel_threshold + 100));
}

TEMPLATE_TEST_CASE_2("big/exp", "[big][exp]", Z, double, float) {
    etl::dyn_matrix<Z> a(1024UL, 2UL);
    etl::dyn_matrix<Z> c(1024UL, 2UL);

    a = etl::uniform_generator(10.0, 50.0);
    c = etl::exp(a);

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], std::exp(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("big/log", "[big][log]", Z, double, float) {
    etl::dyn_matrix<Z> a(1024UL, 2UL);
    etl::dyn_matrix<Z> c(1024UL, 2UL);

    a = etl::uniform_generator(10.0, 50.0);
    c = etl::log(a);

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], std::log(a[i]));
    }
}

TEMPLATE_TEST_CASE_2("big/cos", "[big][cos]", Z, double, float) {
    etl::dyn_matrix<Z> a(128UL, 2UL);
    etl::dyn_matrix<Z> c(128UL, 2UL);

    a = etl::uniform_generator(0.0, 360.0);
    c = etl::cos(a);

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], std::cos(a[i]), base_eps * 10);
    }
}

TEMPLATE_TEST_CASE_2("big/sin", "[big][sin]", Z, double, float) {
    etl::dyn_matrix<Z> a(128UL, 2UL);
    etl::dyn_matrix<Z> c(128UL, 2UL);

    a = etl::uniform_generator(0.0, 360.0);
    c = etl::sin(a);

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX_E(c[i], std::sin(a[i]), base_eps * 10);
    }
}

TEMPLATE_TEST_CASE_2("big/relu_derivative", "[big][add]", Z, double, float) {
    etl::dyn_matrix<Z> a(1024, 1UL);
    etl::dyn_matrix<Z> c(1024, 1UL);

    a = etl::uniform_generator(0.0, 5000.0);
    c = relu_derivative(a);

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], (a[i] > 0.0 ? Z(1.0) : Z(0.0)));
    }
}
