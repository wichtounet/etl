//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"

//{{{ Tests for rep

TEST_CASE( "rep/fast_matrix_1", "rep" ) {
    etl::fast_matrix<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<double, 3, 3> b(etl::rep<3>(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

TEST_CASE( "rep/fast_matrix_2", "rep" ) {
    etl::fast_matrix<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<double, 3, 3> b;

    b = etl::rep<3>(a);

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

//TODO Add The equivalent for dyn_versions

//}}}

//{{{ Tests for sum_r

TEST_CASE( "sum_r/fast_matrix_1", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b(etl::sum_r(a));

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEST_CASE( "sum_r/fast_matrix_2", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::sum_r(a);

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEST_CASE( "sum_r/fast_matrix_3", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::sum_r(a) * 2.5;

    REQUIRE(b(0) == 25.0);
    REQUIRE(b(1) == 5.0);
    REQUIRE(b(2) == 55.0);
}

TEST_CASE( "sum_r/fast_matrix_4", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = (etl::sum_r(a) - etl::sum_r(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

//}}}