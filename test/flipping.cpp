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
#include "etl/print.hpp"

//{{{ hflip

TEST_CASE( "hflip/fast_vector", "hflip" ) {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b(hflip(a));

    REQUIRE(b[0] == 3.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 1.0);
}

TEST_CASE( "hflip/dyn_vector", "hflip" ) {
    etl::dyn_vector<double> a({1.0, -2.0, 3.0});
    etl::dyn_vector<double> b(hflip(a));

    REQUIRE(b[0] == 3.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 1.0);
}

TEST_CASE( "hflip/fast_matrix", "hflip" ) {
    etl::fast_matrix<double, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<double, 3, 2> b(hflip(a));

    REQUIRE(b(0,0) == -2.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -1.0);
    REQUIRE(b(2,1) == 0.0);
}

TEST_CASE( "hflip/dyn_matrix", "hflip" ) {
    etl::dyn_matrix<double> a(3,2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<double> b(hflip(a));

    REQUIRE(b(0,0) == -2.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -1.0);
    REQUIRE(b(2,1) == 0.0);
}

//}}}

//{{{ vflip

TEST_CASE( "vflip/fast_vector", "vflip" ) {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b(vflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEST_CASE( "vflip/dyn_vector", "vflip" ) {
    etl::dyn_vector<double> a({1.0, -2.0, 3.0});
    etl::dyn_vector<double> b(vflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEST_CASE( "vflip/fast_matrix", "vflip" ) {
    etl::fast_matrix<double, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<double, 3, 2> b(vflip(a));

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == -1.0);
    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(2,0) == 1.0);
    REQUIRE(b(2,1) == -2.0);
}

TEST_CASE( "vflip/dyn_matrix", "vflip" ) {
    etl::dyn_matrix<double> a(3,2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<double> b(vflip(a));

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == -1.0);
    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(2,0) == 1.0);
    REQUIRE(b(2,1) == -2.0);
}

//}}}

//{{{ fflip

TEST_CASE( "fflip/fast_vector", "fflip" ) {
    etl::fast_vector<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<double, 3> b(fflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEST_CASE( "fflip/dyn_vector", "fflip" ) {
    etl::dyn_vector<double> a({1.0, -2.0, 3.0});
    etl::dyn_vector<double> b(fflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEST_CASE( "fflip/fast_matrix", "fflip" ) {
    etl::fast_matrix<double, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<double, 3, 2> b(fflip(a));

    REQUIRE(b(0,0) == -1.0);
    REQUIRE(b(0,1) == 0.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -2.0);
    REQUIRE(b(2,1) == 1.0);
}

TEST_CASE( "fflip/dyn_matrix", "fflip" ) {
    etl::dyn_matrix<double> a(3,2, std::initializer_list<double>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<double> b(fflip(a));

    REQUIRE(b(0,0) == -1.0);
    REQUIRE(b(0,1) == 0.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -2.0);
    REQUIRE(b(2,1) == 1.0);
}

//}}}