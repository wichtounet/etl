//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// GEMM tests with mixed types

TEST_CASE("gemm/types/1", "[gemm][type]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<double, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

TEST_CASE("gemm/types/2", "[gemm][type]") {
    etl::fast_matrix<double, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<double, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<float, 2, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0, 0), float(58));
    REQUIRE_EQUALS(c(0, 1), float(64));
    REQUIRE_EQUALS(c(1, 0), float(139));
    REQUIRE_EQUALS(c(1, 1), float(154));
}

TEST_CASE("gemm/types/3", "[gemm][type]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<float, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = a * b;

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

// GEMM_TN tests with mixed types

TEST_CASE("gemm_tn/types/1", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a  = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<double, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

TEST_CASE("gemm_tn/types/2", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a  = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<double, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<float, 2, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0, 0), float(58));
    REQUIRE_EQUALS(c(0, 1), float(64));
    REQUIRE_EQUALS(c(1, 0), float(139));
    REQUIRE_EQUALS(c(1, 1), float(154));
}

TEST_CASE("gemm_tn/types/3", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a  = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<float, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = trans(a) * b;

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

// GEMM_NT tests with mixed types

TEST_CASE("gemm_nt/types/1", "[gemm][type]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<double, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = a * trans(b);

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

TEST_CASE("gemm_nt/types/2", "[gemm][type]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<double, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<float, 2, 2> c;

    c = a * trans(b);

    REQUIRE_EQUALS(c(0, 0), float(58));
    REQUIRE_EQUALS(c(0, 1), float(64));
    REQUIRE_EQUALS(c(1, 0), float(139));
    REQUIRE_EQUALS(c(1, 1), float(154));
}

TEST_CASE("gemm_nt/types/3", "[gemm][type]") {
    etl::fast_matrix<float, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<float, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = a * trans(b);

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

// GEMM_NT tests with mixed types

TEST_CASE("gemm_tt/types/1", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<double, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = trans(a) * trans(b);

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}

TEST_CASE("gemm_tt/types/2", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<double, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<float, 2, 2> c;

    c = trans(a) * trans(b);

    REQUIRE_EQUALS(c(0, 0), float(58));
    REQUIRE_EQUALS(c(0, 1), float(64));
    REQUIRE_EQUALS(c(1, 0), float(139));
    REQUIRE_EQUALS(c(1, 1), float(154));
}

TEST_CASE("gemm_tt/types/3", "[gemm][type]") {
    etl::fast_matrix<float, 3, 2> a = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix<float, 2, 3> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix<double, 2, 2> c;

    c = trans(a) * trans(b);

    REQUIRE_EQUALS(c(0, 0), double(58));
    REQUIRE_EQUALS(c(0, 1), double(64));
    REQUIRE_EQUALS(c(1, 0), double(139));
    REQUIRE_EQUALS(c(1, 1), double(154));
}
