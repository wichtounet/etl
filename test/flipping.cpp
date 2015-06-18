//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

//{{{ hflip

TEMPLATE_TEST_CASE_2( "hflip/fast_vector", "hflip", Z, float, double ) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b(hflip(a));

    REQUIRE(b[0] == 3.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 1.0);
}

TEMPLATE_TEST_CASE_2( "hflip/dyn_vector", "hflip", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b(hflip(a));

    REQUIRE(b[0] == 3.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 1.0);
}

TEMPLATE_TEST_CASE_2( "hflip/fast_matrix", "hflip", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b(hflip(a));

    REQUIRE(b(0,0) == -2.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -1.0);
    REQUIRE(b(2,1) == 0.0);
}

TEMPLATE_TEST_CASE_2( "hflip/dyn_matrix", "hflip", Z, float, double ) {
    etl::dyn_matrix<Z> a(3,2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(hflip(a));

    REQUIRE(b(0,0) == -2.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -1.0);
    REQUIRE(b(2,1) == 0.0);
}

//}}}

//{{{ vflip

TEMPLATE_TEST_CASE_2( "vflip/fast_vector", "vflip", Z, float, double ) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b(vflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "vflip/dyn_vector", "vflip", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b(vflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "vflip/fast_matrix", "vflip", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b(vflip(a));

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == -1.0);
    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(2,0) == 1.0);
    REQUIRE(b(2,1) == -2.0);
}

TEMPLATE_TEST_CASE_2( "vflip/dyn_matrix", "vflip", Z, float, double ) {
    etl::dyn_matrix<Z> a(3,2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(vflip(a));

    REQUIRE(b(0,0) == 0.0);
    REQUIRE(b(0,1) == -1.0);
    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 0.5);
    REQUIRE(b(2,0) == 1.0);
    REQUIRE(b(2,1) == -2.0);
}

//}}}

//{{{ fflip

TEMPLATE_TEST_CASE_2( "fflip/fast_vector", "fflip", Z, float, double ) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b(fflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "fflip/dyn_vector", "fflip", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});
    etl::dyn_vector<Z> b(fflip(a));

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "fflip/fast_matrix", "fflip", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<Z, 3, 2> b(fflip(a));

    REQUIRE(b(0,0) == -1.0);
    REQUIRE(b(0,1) == 0.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -2.0);
    REQUIRE(b(2,1) == 1.0);
}

TEMPLATE_TEST_CASE_2( "fflip/dyn_matrix", "fflip", Z, float, double ) {
    etl::dyn_matrix<Z> a(3,2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<Z> b(fflip(a));

    REQUIRE(b(0,0) == -1.0);
    REQUIRE(b(0,1) == 0.0);
    REQUIRE(b(1,0) == 0.5);
    REQUIRE(b(1,1) == 3.0);
    REQUIRE(b(2,0) == -2.0);
    REQUIRE(b(2,1) == 1.0);
}

//}}}

//{{{ fflip_inplace

TEMPLATE_TEST_CASE_2( "fflip_inplace/1", "[fflip][fast][vector][inplace]", Z, float, double ) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});

    a.fflip_inplace();

    REQUIRE(a[0] == 1.0);
    REQUIRE(a[1] == -2.0);
    REQUIRE(a[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "fflip_inplace/2", "[fflip][dyn][vector][inplace]", Z, float, double ) {
    etl::dyn_vector<Z> a({1.0, -2.0, 3.0});

    a.fflip_inplace();

    REQUIRE(a[0] == 1.0);
    REQUIRE(a[1] == -2.0);
    REQUIRE(a[2] == 3.0);
}

TEMPLATE_TEST_CASE_2( "fflip_inplace/3", "[fflip][fast][matrix][inplace]", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});

    a.fflip_inplace();

    REQUIRE(a(0,0) == -1.0);
    REQUIRE(a(0,1) == 0.0);
    REQUIRE(a(1,0) == 0.5);
    REQUIRE(a(1,1) == 3.0);
    REQUIRE(a(2,0) == -2.0);
    REQUIRE(a(2,1) == 1.0);
}

TEMPLATE_TEST_CASE_2( "fflip_inplace/4", "[fflip][dyn][matrix][inplace]", Z, float, double ) {
    etl::dyn_matrix<Z> a(3,2, std::initializer_list<Z>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    a.fflip_inplace();

    REQUIRE(a(0,0) == -1.0);
    REQUIRE(a(0,1) == 0.0);
    REQUIRE(a(1,0) == 0.5);
    REQUIRE(a(1,1) == 3.0);
    REQUIRE(a(2,0) == -2.0);
    REQUIRE(a(2,1) == 1.0);
}

TEMPLATE_TEST_CASE_2( "fflip_inplace/5", "[fflip][sub][fast][matrix][inplace]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    a(1).fflip_inplace();

    REQUIRE(a(0,0,0) == 1.0);
    REQUIRE(a(0,0,1) == -2.0);
    REQUIRE(a(0,1,0) == 3.0);
    REQUIRE(a(0,1,1) == 0.5);
    REQUIRE(a(0,2,0) == 0.0);
    REQUIRE(a(0,2,1) == -1.0);

    REQUIRE(a(1, 0,0) == -1.0);
    REQUIRE(a(1, 0,1) == 0.0);
    REQUIRE(a(1, 1,0) == 0.5);
    REQUIRE(a(1, 1,1) == 3.0);
    REQUIRE(a(1, 2,0) == -2.0);
    REQUIRE(a(1, 2,1) == 1.0);
}

TEMPLATE_TEST_CASE_2( "fflip_inplace/6", "[fflip][fast][matrix][inplace]", Z, float, double ) {
    etl::fast_matrix<Z, 31, 31> a(etl::magic<Z>(31));
    etl::fast_matrix<Z, 31, 31> b(etl::fflip(a));

    a.fflip_inplace();

    REQUIRE(a == b);
}

//}}}
