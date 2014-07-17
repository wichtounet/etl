//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"

TEST_CASE( "deep_assign/vec<mat>", "deep_assign" ) {
    etl::fast_vector<etl::fast_matrix<double, 2, 3>, 2> a;

    a = 0.0;

    for(auto& v : a){
        for(auto& v2 : v){
            REQUIRE(v2 == 0.0);
        }
    }
}

TEST_CASE( "deep_assign/mat<vec>", "deep_assign" ) {
    etl::fast_matrix<etl::fast_vector<double, 2>, 2, 3> a;

    a = 0.0;

    for(auto& v : a){
        for(auto& v2 : v){
            REQUIRE(v2 == 0.0);
        }
    }
}

TEST_CASE( "deep_assign/mat<mat>", "deep_assign" ) {
    etl::fast_matrix<etl::fast_matrix<double, 2, 3>, 2, 3> a;

    a = 0.0;

    for(auto& v : a){
        for(auto& v2 : v){
            REQUIRE(v2 == 0.0);
        }
    }
}