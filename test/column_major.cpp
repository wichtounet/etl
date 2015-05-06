//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <cmath>

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

TEMPLATE_TEST_CASE_2( "column_major/1", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 2, 3> test_matrix(0);

    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix.template dim<0>() == 2);
    REQUIRE(test_matrix.template dim<1>() == 3);
    REQUIRE(test_matrix.dim(0) == 2);
    REQUIRE(test_matrix.dim(1) == 3);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0) == 1);
    REQUIRE(test_matrix(0, 1) == 3);
    REQUIRE(test_matrix(0, 2) == 5);
    REQUIRE(test_matrix(1, 0) == 2);
    REQUIRE(test_matrix(1, 1) == 4);
    REQUIRE(test_matrix(1, 2) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/2", "[dyn][cm]", Z, int, long ) {
    etl::dyn_matrix_cm<Z> test_matrix(2, 3);

    test_matrix = 0;

    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix.template dim<0>() == 2);
    REQUIRE(test_matrix.template dim<1>() == 3);
    REQUIRE(test_matrix.dim(0) == 2);
    REQUIRE(test_matrix.dim(1) == 3);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0) == 1);
    REQUIRE(test_matrix(0, 1) == 3);
    REQUIRE(test_matrix(0, 2) == 5);
    REQUIRE(test_matrix(1, 0) == 2);
    REQUIRE(test_matrix(1, 1) == 4);
    REQUIRE(test_matrix(1, 2) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/3", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 2, 3, 4> test_matrix;

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0, 0) == 1);
    REQUIRE(test_matrix(0, 0, 1) == 7);
    REQUIRE(test_matrix(0, 0, 2) == 13);
    REQUIRE(test_matrix(0, 0, 3) == 19);
    REQUIRE(test_matrix(0, 1, 0) == 3);
    REQUIRE(test_matrix(0, 1, 1) == 9);
    REQUIRE(test_matrix(0, 1, 2) == 15);
    REQUIRE(test_matrix(0, 1, 3) == 21);
    REQUIRE(test_matrix(0, 2, 0) == 5);
    REQUIRE(test_matrix(0, 2, 1) == 11);
    REQUIRE(test_matrix(0, 2, 2) == 17);
    REQUIRE(test_matrix(0, 2, 3) == 23);
    REQUIRE(test_matrix(1, 0, 0) == 2);
    REQUIRE(test_matrix(1, 0, 1) == 8);
    REQUIRE(test_matrix(1, 0, 2) == 14);
    REQUIRE(test_matrix(1, 0, 3) == 20);
    REQUIRE(test_matrix(1, 1, 0) == 4);
    REQUIRE(test_matrix(1, 1, 1) == 10);
    REQUIRE(test_matrix(1, 1, 2) == 16);
    REQUIRE(test_matrix(1, 1, 3) == 22);
    REQUIRE(test_matrix(1, 2, 0) == 6);
    REQUIRE(test_matrix(1, 2, 1) == 12);
    REQUIRE(test_matrix(1, 2, 2) == 18);
    REQUIRE(test_matrix(1, 2, 3) == 24);
}

TEMPLATE_TEST_CASE_2( "column_major/4", "[dyn][cm]", Z, int, long ) {
    etl::dyn_matrix_cm<Z, 3> test_matrix(2, 3, 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0, 0) == 1);
    REQUIRE(test_matrix(0, 0, 1) == 7);
    REQUIRE(test_matrix(0, 0, 2) == 13);
    REQUIRE(test_matrix(0, 0, 3) == 19);
    REQUIRE(test_matrix(0, 1, 0) == 3);
    REQUIRE(test_matrix(0, 1, 1) == 9);
    REQUIRE(test_matrix(0, 1, 2) == 15);
    REQUIRE(test_matrix(0, 1, 3) == 21);
    REQUIRE(test_matrix(0, 2, 0) == 5);
    REQUIRE(test_matrix(0, 2, 1) == 11);
    REQUIRE(test_matrix(0, 2, 2) == 17);
    REQUIRE(test_matrix(0, 2, 3) == 23);
    REQUIRE(test_matrix(1, 0, 0) == 2);
    REQUIRE(test_matrix(1, 0, 1) == 8);
    REQUIRE(test_matrix(1, 0, 2) == 14);
    REQUIRE(test_matrix(1, 0, 3) == 20);
    REQUIRE(test_matrix(1, 1, 0) == 4);
    REQUIRE(test_matrix(1, 1, 1) == 10);
    REQUIRE(test_matrix(1, 1, 2) == 16);
    REQUIRE(test_matrix(1, 1, 3) == 22);
    REQUIRE(test_matrix(1, 2, 0) == 6);
    REQUIRE(test_matrix(1, 2, 1) == 12);
    REQUIRE(test_matrix(1, 2, 2) == 18);
    REQUIRE(test_matrix(1, 2, 3) == 24);
}
