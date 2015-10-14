//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("direct_access/traits", "has_direct_access", Z, double, float) {
    etl::fast_matrix<Z, 3, 2> a;

    using expr_1 = etl::fast_matrix<Z, 3, 2>;
    using expr_2 = etl::dyn_matrix<Z, 3>;

    REQUIRE(etl::has_direct_access<expr_1>::value);
    REQUIRE(etl::has_direct_access<expr_2>::value);

    using expr_3 = decltype(a + a);
    using expr_4 = decltype(etl::abs(a));

    REQUIRE(!etl::has_direct_access<expr_3>::value);
    REQUIRE(!etl::has_direct_access<expr_4>::value);

    using expr_5  = decltype(a(1));
    using expr_6  = decltype(etl::reshape<2, 3>(a));
    using expr_7  = decltype(etl::reshape<2, 3>(a + a));
    using expr_8  = decltype(etl::reshape(a, 2, 3));
    using expr_9  = decltype(etl::reshape(a + a, 2, 3));
    using expr_10 = decltype(etl::row(a, 1));

    REQUIRE(etl::has_direct_access<expr_5>::value);
    REQUIRE(etl::has_direct_access<expr_6>::value);
    REQUIRE(!etl::has_direct_access<expr_7>::value);
    REQUIRE(etl::has_direct_access<expr_8>::value);
    REQUIRE(!etl::has_direct_access<expr_9>::value);
    REQUIRE(etl::has_direct_access<expr_10>::value);
}

TEMPLATE_TEST_CASE_2("direct_access/fast_matrix", "direct_access", Z, double, float) {
    etl::fast_matrix<Z, 5, 5> test_matrix{etl::magic<5>()};

    REQUIRE(test_matrix.size() == 25);

    auto it  = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE(test_matrix[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEMPLATE_TEST_CASE_2("direct_access/dyn_matrix", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix{etl::magic(5)};

    REQUIRE(test_matrix.size() == 25);

    auto it  = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < test_matrix.size(); ++i) {
        REQUIRE(test_matrix[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEMPLATE_TEST_CASE_2("direct_access/sub_view", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix{etl::magic(5)};

    auto v = test_matrix(1);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE(v[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEMPLATE_TEST_CASE_2("direct_access/reshape", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix{etl::magic(6)};

    auto v = etl::reshape<3, 12>(test_matrix);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE(v[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEMPLATE_TEST_CASE_2("direct_access/reshape_dyn", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix{etl::magic(6)};

    auto v = etl::reshape(test_matrix, 3, 12);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE(v[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEMPLATE_TEST_CASE_2("direct_access/dim_view", "direct_access", Z, double, float) {
    etl::dyn_matrix<Z, 2> test_matrix{etl::magic(6)};

    auto v = row(test_matrix, 2);

    auto it  = v.memory_start();
    auto end = v.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for (std::size_t i = 0; i < etl::size(v); ++i) {
        REQUIRE(v[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}
