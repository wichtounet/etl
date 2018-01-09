//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Vector-Matrix Multiplication with mixed types

// GEVM

TEST_CASE("gevm/types/0", "[gevm]") {
    etl::fast_matrix<float, 3, 2> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<double, 3> b    = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = b * a;

    REQUIRE_EQUALS(c(0), float(76));
    REQUIRE_EQUALS(c(1), float(100));
}

TEST_CASE("gevm/types/1", "[gevm]") {
    etl::fast_matrix<double, 3, 2> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<double, 3> b    = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = b * a;

    REQUIRE_EQUALS(c(0), float(76));
    REQUIRE_EQUALS(c(1), float(100));
}

TEST_CASE("gevm/types/2", "[gevm]") {
    etl::fast_matrix<float, 3, 2> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<float, 3> b    = {7, 8, 9};
    etl::fast_matrix<double, 2> c;

    c = b * a;

    REQUIRE_EQUALS(c(0), double(76));
    REQUIRE_EQUALS(c(1), double(100));
}

// GEVM_T

TEST_CASE("gevm_t/types/0", "[gevm]") {
    etl::fast_matrix<float, 2, 3> a = {1, 3, 5, 2, 4, 6};
    etl::fast_vector<double, 3> b    = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = b * trans(a);

    REQUIRE_EQUALS(c(0), float(76));
    REQUIRE_EQUALS(c(1), float(100));
}

TEST_CASE("gevm_t/types/1", "[gevm]") {
    etl::fast_matrix<double, 2, 3> a = {1, 3, 5, 2, 4, 6};
    etl::fast_vector<double, 3> b    = {7, 8, 9};
    etl::fast_matrix<float, 2> c;

    c = b * trans(a);

    REQUIRE_EQUALS(c(0), float(76));
    REQUIRE_EQUALS(c(1), float(100));
}

TEST_CASE("gevm_t/types/2", "[gevm]") {
    etl::fast_matrix<float, 2, 3> a = {1, 3, 5, 2, 4, 6};
    etl::fast_vector<float, 3> b    = {7, 8, 9};
    etl::fast_matrix<double, 2> c;

    c = b * trans(a);

    REQUIRE_EQUALS(c(0), double(76));
    REQUIRE_EQUALS(c(1), double(100));
}
