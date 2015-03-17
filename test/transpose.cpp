//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

TEMPLATE_TEST_CASE_2( "transpose/fast_matrix_1", "transpose", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 2, 3> b(transpose(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 3.0);
    REQUIRE(b(0,2) == 0.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(1,2) == -1);
}

TEMPLATE_TEST_CASE_2( "transpose/dyn_matrix_1", "transpose", Z, float, double ) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(transpose(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 3.0);
    REQUIRE(b(0,2) == 0.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(1,2) == -1);
}

TEMPLATE_TEST_CASE_2( "transpose/fast_matrix_2", "transpose", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b(transpose(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 0.5);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == 0.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == -1);
}

TEMPLATE_TEST_CASE_2( "transpose/dyn_matrix_2", "transpose", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 3, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(transpose(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 0.5);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == 0.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == -1);
}
