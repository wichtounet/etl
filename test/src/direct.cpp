//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("direct_access/traits", "is_dma", Z, double, float) {
    etl::fast_matrix<Z, 3, 2> a;

    using expr_1 = etl::fast_matrix<Z, 3, 2>;
    using expr_2 = etl::dyn_matrix<Z, 3>;

    REQUIRE_DIRECT(etl::is_dma<expr_1>);
    REQUIRE_DIRECT(etl::is_dma<expr_2>);

    using expr_3 = decltype(a + a);
    using expr_4 = decltype(etl::abs(a));

    REQUIRE_DIRECT(!etl::is_dma<expr_3>);
    REQUIRE_DIRECT(!etl::is_dma<expr_4>);

    using expr_5  = decltype(a(1));
    using expr_6  = decltype(etl::reshape<2, 3>(a));
    using expr_7  = decltype(etl::reshape<2, 3>(a + a));
    using expr_8  = decltype(etl::reshape(a, 2, 3));
    using expr_9  = decltype(etl::reshape(a + a, 2, 3));
    using expr_10 = decltype(etl::row(a, 1));

    REQUIRE_DIRECT(etl::is_dma<expr_5>);
    REQUIRE_DIRECT(etl::is_dma<expr_6>);
    REQUIRE_DIRECT(!etl::is_dma<expr_7>);
    REQUIRE_DIRECT(etl::is_dma<expr_8>);
    REQUIRE_DIRECT(!etl::is_dma<expr_9>);
    REQUIRE_DIRECT(etl::is_dma<expr_10>);
}

TEMPLATE_TEST_CASE_2("direct_access/fast_matrix", "direct_access", Z, double, float) {
    etl::fast_matrix<Z, 5, 5> test_matrix;
    test_matrix = etl::magic<5>();

    REQUIRE_EQUALS(test_matrix.size(), 25UL);

    auto it  = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}

TEMPLATE_TEST_CASE_2("direct_access/dyn_matrix", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix;
    test_matrix = etl::magic(5);

    REQUIRE_EQUALS(test_matrix.size(), 25UL);

    auto it  = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE_EQUALS(test_matrix[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}

TEMPLATE_TEST_CASE_2("direct_access/sub_view", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix;
    test_matrix = etl::magic(5);

    auto v = test_matrix(1);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE_EQUALS(v[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}

TEMPLATE_TEST_CASE_2("direct_access/reshape", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix;
    test_matrix = etl::magic(6);

    auto v = etl::reshape<3, 12>(test_matrix);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE_EQUALS(v[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}

TEMPLATE_TEST_CASE_2("direct_access/reshape_dyn", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix;
    test_matrix = etl::magic(6);

    auto v = etl::reshape(test_matrix, 3, 12);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE_EQUALS(v[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}

TEMPLATE_TEST_CASE_2("direct_access/dim_view", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix;
    test_matrix = etl::magic(6);

    auto v = row(test_matrix, 2);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE_DIRECT(std::is_pointer<decltype(it)>::value);
    REQUIRE_DIRECT(std::is_pointer<decltype(end)>::value);

    for (size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE_EQUALS(v[i], *it);
        REQUIRE_DIRECT(it != end);
        ++it;
    }

    REQUIRE_EQUALS(it, end);
}
