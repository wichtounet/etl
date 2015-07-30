//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "etl/stop.hpp"

///{{{ magic(n)

TEMPLATE_TEST_CASE_2( "magic/dyn_matrix_1", "magic(1)", Z, float, double ) {
    auto m = etl::s(etl::magic<Z>(1));

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 1);
    REQUIRE((std::is_same<typename decltype(m)::value_type, Z>::value));
    REQUIRE(etl::is_etl_expr<decltype(m)>::value);

    REQUIRE(etl::decay_traits<decltype(etl::magic<Z>(1))>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(etl::magic<Z>(1))>::size(etl::magic<Z>(1)) == 1);
    REQUIRE((std::is_same<typename decltype(etl::magic<Z>(1))::value_type, Z>::value));
    REQUIRE(etl::is_etl_expr<decltype(etl::magic<Z>(1))>::value);

    REQUIRE(etl::size(m) == 1);

    REQUIRE(m[0] == 1);
    REQUIRE(m(0,0) == 1);
}

TEMPLATE_TEST_CASE_2( "magic/dyn_matrix_2", "magic(2)", Z, float, double ) {
    auto m = etl::s(etl::magic<Z>(2));

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 4);
    REQUIRE(etl::size(m) == 4);

    REQUIRE(m[0] == 1);
    REQUIRE(m[1] == 3);
    REQUIRE(m[2] == 4);
    REQUIRE(m[3] == 2);
    REQUIRE(m(0,0) == 1);
    REQUIRE(m(0,1) == 3);
    REQUIRE(m(1,0) == 4);
    REQUIRE(m(1,1) == 2);
}

TEMPLATE_TEST_CASE_2( "magic/dyn_matrix_3", "magic(3)", Z, float, double ) {
    auto m = etl::s(etl::magic<Z>(3));

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 9);
    REQUIRE(etl::size(m) == 9);

    REQUIRE(m[0] == 8);
    REQUIRE(m(0,0) == 8);
    REQUIRE(m(0,1) == 1);
    REQUIRE(m(0,2) == 6);
    REQUIRE(m(1,0) == 3);
    REQUIRE(m(1,1) == 5);
    REQUIRE(m(1,2) == 7);
    REQUIRE(m(2,0) == 4);
    REQUIRE(m(2,1) == 9);
    REQUIRE(m(2,2) == 2);
}

TEMPLATE_TEST_CASE_2( "magic/dyn_matrix_4", "magic(4)", Z, float, double ) {
    auto m = etl::s(etl::magic<Z>(4));

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 16);
    REQUIRE(etl::size(m) == 16);

    REQUIRE(m[4] == 3);
    REQUIRE(m(2,0) == 8);
    REQUIRE(m(2,1) == 10);
    REQUIRE(m(2,2) == 16);
    REQUIRE(m(2,3) == 2);
}

///}}}

///{{{ magic<N>

TEMPLATE_TEST_CASE_2( "fast_magic/dyn_matrix_1", "fast_magic(1)", Z, float, double ) {
    auto m = etl::s(etl::magic<1, Z>());

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == true);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 1);
    REQUIRE((std::is_same<typename decltype(m)::value_type, Z>::value));
    REQUIRE(etl::is_etl_expr<decltype(m)>::value);

    REQUIRE(etl::decay_traits<decltype(etl::magic<Z>(1))>::is_fast == false);
    REQUIRE(etl::decay_traits<decltype(etl::magic<Z>(1))>::size(etl::magic<Z>(1)) == 1);
    REQUIRE((std::is_same<typename decltype(etl::magic<Z>(1))::value_type, Z>::value));
    REQUIRE(etl::is_etl_expr<decltype(etl::magic<Z>(1))>::value);

    REQUIRE(etl::size(m) == 1);

    REQUIRE(m[0] == 1);
    REQUIRE(m(0,0) == 1);
}

TEMPLATE_TEST_CASE_2( "fast_magic/dyn_matrix_2", "fast_magic(2)", Z, float, double ) {
    auto m = etl::s(etl::magic<2, Z>());

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == true);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 4);
    REQUIRE(etl::size(m) == 4);

    REQUIRE(m[0] == 1);
    REQUIRE(m[1] == 3);
    REQUIRE(m[2] == 4);
    REQUIRE(m[3] == 2);
    REQUIRE(m(0,0) == 1);
    REQUIRE(m(0,1) == 3);
    REQUIRE(m(1,0) == 4);
    REQUIRE(m(1,1) == 2);
}

TEMPLATE_TEST_CASE_2( "fast_magic/dyn_matrix_3", "fast_magic(3)", Z, float, double ) {
    auto m = etl::s(etl::magic<3, Z>());

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == true);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 9);
    REQUIRE(etl::size(m) == 9);

    REQUIRE(m[0] == 8);
    REQUIRE(m(0,0) == 8);
    REQUIRE(m(0,1) == 1);
    REQUIRE(m(0,2) == 6);
    REQUIRE(m(1,0) == 3);
    REQUIRE(m(1,1) == 5);
    REQUIRE(m(1,2) == 7);
    REQUIRE(m(2,0) == 4);
    REQUIRE(m(2,1) == 9);
    REQUIRE(m(2,2) == 2);
}

TEMPLATE_TEST_CASE_2( "fast_magic/dyn_matrix_4", "fast_magic(4)", Z, float, double ) {
    auto m = etl::s(etl::magic<4, Z>());

    REQUIRE(etl::decay_traits<decltype(m)>::is_fast == true);
    REQUIRE(etl::decay_traits<decltype(m)>::size(m) == 16);
    REQUIRE(etl::size(m) == 16);

    REQUIRE(m[4] == 3);
    REQUIRE(m(2,0) == 8);
    REQUIRE(m(2,1) == 10);
    REQUIRE(m(2,2) == 16);
    REQUIRE(m(2,3) == 2);
}

///}}}
