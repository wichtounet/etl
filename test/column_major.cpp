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
