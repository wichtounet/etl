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
    etl::sparse_matrix<Z> a;

    REQUIRE(a.rows() == 0);
    REQUIRE(a.columns() == 0);
    REQUIRE(a.size() == 0);
    REQUIRE(a.non_zeros() == 0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/2", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 6);
    REQUIRE(a.non_zeros() == 3);

    REQUIRE(a(0, 0) == Z(1.0));
    REQUIRE(a(0, 1) == Z(0.0));
    REQUIRE(a(1, 0) == Z(0.0));
    REQUIRE(a(1, 1) == Z(2.0));
    REQUIRE(a(2, 0) == Z(3.0));
    REQUIRE(a(2, 1) == Z(0.0));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/3", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, etl::values(0.0, 1.2, 0.0, 2.0, 0.0, 0.01));

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 6);
    REQUIRE(a.non_zeros() == 3);

    REQUIRE(a(0, 0) == Z(0.0));
    REQUIRE(a(0, 1) == Z(1.2));
    REQUIRE(a(1, 0) == Z(0.0));
    REQUIRE(a(1, 1) == Z(2.0));
    REQUIRE(a(2, 0) == Z(0.0));
    REQUIRE(a(2, 1) == Z(0.01));
}
