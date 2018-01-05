//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef NDEBUG

#include "test_light.hpp"

ETL_TEST_CASE("assert/nothrow/1", "[assert]") {
#ifdef CPP_UTILS_ASSERT_EXCEPTION
    REQUIRE_DIRECT(!etl::assert_nothrow);
#else
    REQUIRE_DIRECT(etl::assert_nothrow);
#endif
}

#ifdef CPP_UTILS_ASSERT_EXCEPTION
ETL_TEST_CASE("assert/sizes/1", "[assert]") {
    etl::dyn_vector<double> a = {-1.0, 2.0, 5.0};
    etl::dyn_vector<double> b = {2.5, 3.0, 4.0, 1.0};

    REQUIRE_THROWS(a + b);
}

ETL_TEST_CASE("assert/dim/1", "[assert]") {
    etl::fast_matrix<double, 2, 3, 4> matrix;

    REQUIRE_NOTHROW(matrix(1, 1, 1));
    REQUIRE_THROWS(matrix(3, 2, 1));
    REQUIRE_THROWS(matrix(2, 2, 1));
    REQUIRE_THROWS(matrix(1, 5, 1));
    REQUIRE_THROWS(matrix(1, 1, 5));
    REQUIRE_THROWS(matrix(1, 1, 4));
    REQUIRE_THROWS(matrix(3, 3, 4));
}

ETL_TEST_CASE("assert/dim/2", "[assert]") {
    etl::fast_dyn_matrix<double, 2, 3, 4> matrix;

    REQUIRE_NOTHROW(matrix(1, 1, 1));
    REQUIRE_THROWS(matrix(3, 2, 1));
    REQUIRE_THROWS(matrix(2, 2, 1));
    REQUIRE_THROWS(matrix(1, 5, 1));
    REQUIRE_THROWS(matrix(1, 1, 5));
    REQUIRE_THROWS(matrix(1, 1, 4));
    REQUIRE_THROWS(matrix(3, 3, 4));
}

ETL_TEST_CASE("assert/dim/3", "[assert]") {
    etl::dyn_matrix<double, 3> matrix(2, 3, 4);

    REQUIRE_NOTHROW(matrix(1, 1, 1));
    REQUIRE_THROWS(matrix(3, 2, 1));
    REQUIRE_THROWS(matrix(2, 2, 1));
    REQUIRE_THROWS(matrix(1, 5, 1));
    REQUIRE_THROWS(matrix(1, 1, 5));
    REQUIRE_THROWS(matrix(1, 1, 4));
    REQUIRE_THROWS(matrix(3, 3, 4));
}

ETL_TEST_CASE("assert/dim/4", "[assert]") {
    etl::sparse_matrix<double> matrix(2, 3);

    REQUIRE_NOTHROW(matrix(1, 1));
    REQUIRE_THROWS(matrix(3, 2));
    REQUIRE_THROWS(matrix(2, 2));
    REQUIRE_THROWS(matrix(1, 5));
    REQUIRE_THROWS(matrix(3, 5));
}
#endif

#endif
