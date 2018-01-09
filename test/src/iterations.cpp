//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("iterable/fast_matrix", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> test_matrix(5.5);

    for (auto& v : test_matrix) {
        REQUIRE_EQUALS(v, 5.5);
    }
}

TEMPLATE_TEST_CASE_2("iterable/dyn_matrix", "iterable", Z, float, double) {
    etl::dyn_matrix<Z> test_matrix(2, 2, 5.5);

    for (auto& v : test_matrix) {
        REQUIRE_EQUALS(v, 5.5);
    }
}

TEMPLATE_TEST_CASE_2("iterable/binary_expr", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(5.5);

    auto expr = a + a;

    for (auto v : expr) {
        REQUIRE_EQUALS(v, 11.0);
    }
}

TEMPLATE_TEST_CASE_2("iterable/unary_expr", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(5.5);

    auto expr = -a;

    for (auto v : expr) {
        REQUIRE_EQUALS(v, -5.5);
    }
}

TEMPLATE_TEST_CASE_2("iterable/identity", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(5.5);

    auto expr = a(0);

    for (auto v : expr) {
        REQUIRE_EQUALS(v, 5.5);
    }
}

TEMPLATE_TEST_CASE_2("iterable/identity_2", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(5.5);

    auto expr = sub(a + a, 0);

    for (auto v : expr) {
        REQUIRE_EQUALS(v, 11.0);
    }
}

TEMPLATE_TEST_CASE_2("iterable/stable_transform_expr", "iterable", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(5.5);

    auto expr = mean_l(a);

    for (auto v : expr) {
        REQUIRE_EQUALS(v, 5.5);
    }
}

TEMPLATE_TEST_CASE_2("iterator/binary_expr", "iterator", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1, 2, 3, 4});

    auto expr = a + a;

    auto it  = expr.begin();
    auto end = expr.end();

    REQUIRE_EQUALS(std::distance(it, end), 4);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*it, 2.0);
    REQUIRE_EQUALS(it[3], 8.0);
    REQUIRE_EQUALS(*std::next(it), 4.0);

    ++it;

    REQUIRE_EQUALS(*it, 4.0);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*std::prev(it), 2.0);

    it += 2;

    REQUIRE_EQUALS(*it, 8.0);
    REQUIRE_DIRECT(it != end);

    auto temp = it + 1;
    REQUIRE_EQUALS(*(it - 1), 6.0);
    REQUIRE_EQUALS(temp, end);

    REQUIRE_EQUALS(end, expr.end());
    REQUIRE_EQUALS(end, std::end(expr));

    it = std::begin(expr);
    std::advance(it, 3);

    REQUIRE_EQUALS(*it, 8.0);

    REQUIRE_EQUALS(std::accumulate(std::begin(expr), std::end(expr), 0.0), 20.0);
}

TEMPLATE_TEST_CASE_2("iterator/const identity", "iterator", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a({1, 2, 3, 4, 1, 2, 3, 4});

    const auto expr = a(0);

    for (auto& v : expr) {
        REQUIRE_DIRECT(v > 0);
    }

    auto it  = expr.begin();
    auto end = expr.end();

    REQUIRE_EQUALS(std::distance(it, end), 4);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*it, 1.0);
    REQUIRE_EQUALS(it[3], 4.0);
    REQUIRE_EQUALS(*std::next(it), 2.0);

    ++it;

    REQUIRE_EQUALS(*it, 2.0);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*std::prev(it), 1.0);

    it += 2;

    REQUIRE_EQUALS(*it, 4.0);
    REQUIRE_DIRECT(it != end);

    auto temp = it + 1;
    REQUIRE_EQUALS(*(it - 1), 3.0);
    REQUIRE_EQUALS(temp, end);

    REQUIRE_EQUALS(end, expr.end());
    REQUIRE_EQUALS(end, std::end(expr));

    it = std::begin(expr);
    std::advance(it, 3);

    REQUIRE_EQUALS(*it, 4.0);

    REQUIRE_EQUALS(std::accumulate(std::begin(expr), std::end(expr), 0.0), 10.0);
}

TEMPLATE_TEST_CASE_2("iterator/identity", "iterator", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a({1, 2, 3, 4, 1, 2, 3, 4});

    auto expr = a(0);

    for (auto& v : expr) {
        ++v;
        REQUIRE_DIRECT(v > 0);
    }

    auto it  = expr.begin();
    auto end = expr.end();

    REQUIRE_EQUALS(std::distance(it, end), 4);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*it, 2.0);
    REQUIRE_EQUALS(it[3], 5.0);
    REQUIRE_EQUALS(*std::next(it), 3.0);

    ++it;

    REQUIRE_EQUALS(*it, 3.0);
    REQUIRE_DIRECT(it != end);
    REQUIRE_EQUALS(*std::prev(it), 2.0);

    it += 2;

    REQUIRE_EQUALS(*it, 5.0);
    REQUIRE_DIRECT(it != end);

    auto temp = it + 1;
    REQUIRE_EQUALS(*(it - 1), 4.0);
    REQUIRE_EQUALS(temp, end);

    REQUIRE_EQUALS(end, expr.end());
    REQUIRE_EQUALS(end, std::end(expr));

    it = std::begin(expr);
    std::advance(it, 3);

    REQUIRE_EQUALS(*it, 5.0);

    REQUIRE_EQUALS(std::accumulate(std::begin(expr), std::end(expr), 0.0), 14.0);
}
