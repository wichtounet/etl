//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Matrix-Vector Multiplication with mixed types

// GEMV

ETL_TEST_CASE("gemv/types/0", "[gemv]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<double, 3> b = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0), float(50));
    REQUIRE_EQUALS(c(1), float(122));
}

ETL_TEST_CASE("gemv/types/1", "[gemv]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<float, 3> b = {7, 8, 9};
    etl::fast_matrix<double, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0), double(50));
    REQUIRE_EQUALS(c(1), double(122));
}

ETL_TEST_CASE("gemv/types/2", "[gemv]") {
    etl::fast_matrix<double, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<double, 3> b = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0), float(50));
    REQUIRE_EQUALS(c(1), float(122));
}

// GEMV_T

ETL_TEST_CASE("gemv_t/types/0", "[gemv]") {
    etl::fast_matrix<float, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_vector<double, 3> b = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0), float(50));
    REQUIRE_EQUALS(c(1), float(122));
}

ETL_TEST_CASE("gemv_t/types/1", "[gemv]") {
    etl::fast_matrix<float, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_vector<float, 3> b = {7, 8, 9};
    etl::fast_matrix<double, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0), double(50));
    REQUIRE_EQUALS(c(1), double(122));
}

ETL_TEST_CASE("gemv_t/types/2", "[gemv]") {
    etl::fast_matrix<double, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_vector<double, 3> b = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0), float(50));
    REQUIRE_EQUALS(c(1), float(122));
}
