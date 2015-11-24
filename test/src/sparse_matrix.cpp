//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

// Init tests

TEMPLATE_TEST_CASE_2("sparse_matrix/init/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> test_matrix;

    REQUIRE(test_matrix.rows() == 0);
    REQUIRE(test_matrix.columns() == 0);
    REQUIRE(test_matrix.size() == 0);
}
