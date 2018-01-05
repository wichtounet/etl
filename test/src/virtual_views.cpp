//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "etl/stop.hpp"

/// magic(n)

TEMPLATE_TEST_CASE_2("magic/dyn_matrix_1", "magic(1)", Z, float, double) {
    auto m = etl::s(etl::magic<Z>(1));

    REQUIRE_DIRECT(!etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 1UL);
    REQUIRE_DIRECT((std::is_same<typename decltype(m)::value_type, Z>::value));
    REQUIRE_DIRECT(etl::is_etl_expr<decltype(m)>);

    REQUIRE_DIRECT(!etl::decay_traits<decltype(etl::magic<Z>(1))>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(etl::magic<Z>(1))>::size(etl::magic<Z>(1)), 1UL);
    REQUIRE_DIRECT((std::is_same<typename decltype(etl::magic<Z>(1))::value_type, Z>::value));
    REQUIRE_DIRECT(etl::is_etl_expr<decltype(etl::magic<Z>(1))>);

    REQUIRE_EQUALS(etl::size(m), 1UL);

    REQUIRE_EQUALS(m[0], 1);
    REQUIRE_EQUALS(m(0, 0), 1);
}

TEMPLATE_TEST_CASE_2("magic/dyn_matrix_2", "magic(2)", Z, float, double) {
    auto m = etl::s(etl::magic<Z>(2));

    REQUIRE_DIRECT(!etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 4UL);
    REQUIRE_EQUALS(etl::size(m), 4UL);

    REQUIRE_EQUALS(m[0], 1);
    REQUIRE_EQUALS(m[1], 3);
    REQUIRE_EQUALS(m[2], 4);
    REQUIRE_EQUALS(m[3], 2);
    REQUIRE_EQUALS(m(0, 0), 1);
    REQUIRE_EQUALS(m(0, 1), 3);
    REQUIRE_EQUALS(m(1, 0), 4);
    REQUIRE_EQUALS(m(1, 1), 2);
}

TEMPLATE_TEST_CASE_2("magic/dyn_matrix_3", "magic(3)", Z, float, double) {
    auto m = etl::s(etl::magic<Z>(3));

    REQUIRE_DIRECT(!etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 9UL);
    REQUIRE_EQUALS(etl::size(m), 9UL);

    REQUIRE_EQUALS(m[0], 8);
    REQUIRE_EQUALS(m(0, 0), 8);
    REQUIRE_EQUALS(m(0, 1), 1);
    REQUIRE_EQUALS(m(0, 2), 6);
    REQUIRE_EQUALS(m(1, 0), 3);
    REQUIRE_EQUALS(m(1, 1), 5);
    REQUIRE_EQUALS(m(1, 2), 7);
    REQUIRE_EQUALS(m(2, 0), 4);
    REQUIRE_EQUALS(m(2, 1), 9);
    REQUIRE_EQUALS(m(2, 2), 2);
}

TEMPLATE_TEST_CASE_2("magic/dyn_matrix_4", "magic(4)", Z, float, double) {
    auto m = etl::s(etl::magic<Z>(4));

    REQUIRE_DIRECT(!etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 16UL);
    REQUIRE_EQUALS(etl::size(m), 16UL);

    REQUIRE_EQUALS(m[4], 3);
    REQUIRE_EQUALS(m(2, 0), 8);
    REQUIRE_EQUALS(m(2, 1), 10);
    REQUIRE_EQUALS(m(2, 2), 16);
    REQUIRE_EQUALS(m(2, 3), 2);
}

/// magic<N>

TEMPLATE_TEST_CASE_2("fast_magic/dyn_matrix_1", "fast_magic(1)", Z, float, double) {
    auto m = etl::s(etl::magic<1, Z>());

    REQUIRE_DIRECT(etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 1UL);
    REQUIRE_DIRECT((std::is_same<typename decltype(m)::value_type, Z>::value));
    REQUIRE_DIRECT(etl::is_etl_expr<decltype(m)>);

    REQUIRE_DIRECT(!etl::decay_traits<decltype(etl::magic<Z>(1))>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(etl::magic<Z>(1))>::size(etl::magic<Z>(1)), 1UL);
    REQUIRE_DIRECT((std::is_same<typename decltype(etl::magic<Z>(1))::value_type, Z>::value));
    REQUIRE_DIRECT(etl::is_etl_expr<decltype(etl::magic<Z>(1))>);

    REQUIRE_EQUALS(etl::size(m), 1UL);

    REQUIRE_EQUALS(m[0], 1);
    REQUIRE_EQUALS(m(0, 0), 1);
}

TEMPLATE_TEST_CASE_2("fast_magic/dyn_matrix_2", "fast_magic(2)", Z, float, double) {
    auto m = etl::s(etl::magic<2, Z>());

    REQUIRE_DIRECT(etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 4UL);
    REQUIRE_EQUALS(etl::size(m), 4UL);

    REQUIRE_EQUALS(m[0], 1);
    REQUIRE_EQUALS(m[1], 3);
    REQUIRE_EQUALS(m[2], 4);
    REQUIRE_EQUALS(m[3], 2);
    REQUIRE_EQUALS(m(0, 0), 1);
    REQUIRE_EQUALS(m(0, 1), 3);
    REQUIRE_EQUALS(m(1, 0), 4);
    REQUIRE_EQUALS(m(1, 1), 2);
}

TEMPLATE_TEST_CASE_2("fast_magic/dyn_matrix_3", "fast_magic(3)", Z, float, double) {
    auto m = etl::s(etl::magic<3, Z>());

    REQUIRE_DIRECT(etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 9UL);
    REQUIRE_EQUALS(etl::size(m), 9UL);

    REQUIRE_EQUALS(m[0], 8);
    REQUIRE_EQUALS(m(0, 0), 8);
    REQUIRE_EQUALS(m(0, 1), 1);
    REQUIRE_EQUALS(m(0, 2), 6);
    REQUIRE_EQUALS(m(1, 0), 3);
    REQUIRE_EQUALS(m(1, 1), 5);
    REQUIRE_EQUALS(m(1, 2), 7);
    REQUIRE_EQUALS(m(2, 0), 4);
    REQUIRE_EQUALS(m(2, 1), 9);
    REQUIRE_EQUALS(m(2, 2), 2);
}

TEMPLATE_TEST_CASE_2("fast_magic/dyn_matrix_4", "fast_magic(4)", Z, float, double) {
    auto m = etl::s(etl::magic<4, Z>());

    REQUIRE_DIRECT(etl::decay_traits<decltype(m)>::is_fast);
    REQUIRE_EQUALS(etl::decay_traits<decltype(m)>::size(m), 16UL);
    REQUIRE_EQUALS(etl::size(m), 16UL);

    REQUIRE_EQUALS(m[4], 3);
    REQUIRE_EQUALS(m(2, 0), 8);
    REQUIRE_EQUALS(m(2, 1), 10);
    REQUIRE_EQUALS(m(2, 2), 16);
    REQUIRE_EQUALS(m(2, 3), 2);
}
