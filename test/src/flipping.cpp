//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"


TEMPLATE_TEST_CASE_2("flip/traits/0", "flip", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> A;

    decltype(auto) a = hflip(A);
    decltype(auto) b = vflip(A);
    decltype(auto) c = fflip(A);

    REQUIRE_EQUALS(etl::dimensions(a), 2UL);
    REQUIRE_EQUALS(etl::dimensions(b), 2UL);
    REQUIRE_EQUALS(etl::dimensions(c), 2UL);

    REQUIRE_EQUALS(etl::dim<0>(a), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(b), 2UL);
    REQUIRE_EQUALS(etl::dim<0>(c), 2UL);

    REQUIRE_EQUALS(etl::dim<1>(a), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(b), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(c), 3UL);
}

// hflip

TEMPLATE_TEST_CASE_2("hflip/fast_vector", "hflip", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;
    b = hflip(a);

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 1.0);
}

TEMPLATE_TEST_CASE_2("hflip/dyn_vector", "hflip", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b;
    b = hflip(a);

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 1.0);
}

TEMPLATE_TEST_CASE_2("hflip/fast_matrix", "hflip", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b;
    b = hflip(a);

    REQUIRE_EQUALS(b(0, 0), -2.0);
    REQUIRE_EQUALS(b(0, 1), 1.0);
    REQUIRE_EQUALS(b(1, 0), 0.5);
    REQUIRE_EQUALS(b(1, 1), 3.0);
    REQUIRE_EQUALS(b(2, 0), -1.0);
    REQUIRE_EQUALS(b(2, 1), 0.0);
}

TEMPLATE_TEST_CASE_2("hflip/dyn_matrix", "hflip", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b;
    b = hflip(a);

    REQUIRE_EQUALS(b(0, 0), -2.0);
    REQUIRE_EQUALS(b(0, 1), 1.0);
    REQUIRE_EQUALS(b(1, 0), 0.5);
    REQUIRE_EQUALS(b(1, 1), 3.0);
    REQUIRE_EQUALS(b(2, 0), -1.0);
    REQUIRE_EQUALS(b(2, 1), 0.0);
}

// vflip

TEMPLATE_TEST_CASE_2("vflip/fast_vector", "vflip", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = vflip(a);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 3.0);
}

TEMPLATE_TEST_CASE_2("vflip/dyn_vector", "vflip", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b;
    b = vflip(a);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 3.0);
}

TEMPLATE_TEST_CASE_2("vflip/fast_matrix", "vflip", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b;
    b = vflip(a);

    REQUIRE_EQUALS(b(0, 0), 0.0);
    REQUIRE_EQUALS(b(0, 1), -1.0);
    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 0.5);
    REQUIRE_EQUALS(b(2, 0), 1.0);
    REQUIRE_EQUALS(b(2, 1), -2.0);
}

TEMPLATE_TEST_CASE_2("vflip/dyn_matrix", "vflip", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b;
    b = vflip(a);

    REQUIRE_EQUALS(b(0, 0), 0.0);
    REQUIRE_EQUALS(b(0, 1), -1.0);
    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 0.5);
    REQUIRE_EQUALS(b(2, 0), 1.0);
    REQUIRE_EQUALS(b(2, 1), -2.0);
}

// fflip

TEMPLATE_TEST_CASE_2("fflip/fast_vector", "fflip", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;
    b = fflip(a);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 3.0);
}

TEMPLATE_TEST_CASE_2("fflip/dyn_vector", "fflip", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b;
    b = fflip(a);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 3.0);
}

TEMPLATE_TEST_CASE_2("fflip/fast_matrix", "fflip", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b;
    b = fflip(a);

    REQUIRE_EQUALS(b(0, 0), -1.0);
    REQUIRE_EQUALS(b(0, 1), 0.0);
    REQUIRE_EQUALS(b(1, 0), 0.5);
    REQUIRE_EQUALS(b(1, 1), 3.0);
    REQUIRE_EQUALS(b(2, 0), -2.0);
    REQUIRE_EQUALS(b(2, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("fflip/dyn_matrix", "fflip", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b;
    b = fflip(a);

    REQUIRE_EQUALS(b(0, 0), -1.0);
    REQUIRE_EQUALS(b(0, 1), 0.0);
    REQUIRE_EQUALS(b(1, 0), 0.5);
    REQUIRE_EQUALS(b(1, 1), 3.0);
    REQUIRE_EQUALS(b(2, 0), -2.0);
    REQUIRE_EQUALS(b(2, 1), 1.0);
}

// fflip_inplace

TEMPLATE_TEST_CASE_2("fflip_inplace/1", "[fflip][fast][vector][inplace]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});

    a.fflip_inplace();

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], -2.0);
    REQUIRE_EQUALS(a[2], 3.0);
}

TEMPLATE_TEST_CASE_2("fflip_inplace/2", "[fflip][dyn][vector][inplace]", Z, float, double) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});

    a.fflip_inplace();

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], -2.0);
    REQUIRE_EQUALS(a[2], 3.0);
}

TEMPLATE_TEST_CASE_2("fflip_inplace/3", "[fflip][fast][matrix][inplace]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    a.fflip_inplace();

    REQUIRE_EQUALS(a(0, 0), -1.0);
    REQUIRE_EQUALS(a(0, 1), 0.0);
    REQUIRE_EQUALS(a(1, 0), 0.5);
    REQUIRE_EQUALS(a(1, 1), 3.0);
    REQUIRE_EQUALS(a(2, 0), -2.0);
    REQUIRE_EQUALS(a(2, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("fflip_inplace/4", "[fflip][dyn][matrix][inplace]", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    a.fflip_inplace();

    REQUIRE_EQUALS(a(0, 0), -1.0);
    REQUIRE_EQUALS(a(0, 1), 0.0);
    REQUIRE_EQUALS(a(1, 0), 0.5);
    REQUIRE_EQUALS(a(1, 1), 3.0);
    REQUIRE_EQUALS(a(2, 0), -2.0);
    REQUIRE_EQUALS(a(2, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("fflip_inplace/5", "[fflip][sub][fast][matrix][inplace]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    a(1).fflip_inplace();

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), -2.0);
    REQUIRE_EQUALS(a(0, 1, 0), 3.0);
    REQUIRE_EQUALS(a(0, 1, 1), 0.5);
    REQUIRE_EQUALS(a(0, 2, 0), 0.0);
    REQUIRE_EQUALS(a(0, 2, 1), -1.0);

    REQUIRE_EQUALS(a(1, 0, 0), -1.0);
    REQUIRE_EQUALS(a(1, 0, 1), 0.0);
    REQUIRE_EQUALS(a(1, 1, 0), 0.5);
    REQUIRE_EQUALS(a(1, 1, 1), 3.0);
    REQUIRE_EQUALS(a(1, 2, 0), -2.0);
    REQUIRE_EQUALS(a(1, 2, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("fflip_inplace/6", "[fflip][fast][matrix][inplace]", Z, float, double) {
    etl::fast_matrix<Z, 31, 31> a;
    etl::fast_matrix<Z, 31, 31> b;

    a = etl::magic<Z>(31);
    b = etl::fflip(a);

    a.fflip_inplace();

    REQUIRE_EQUALS(a, b);
}
