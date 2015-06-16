//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

#include "conv_test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

TEMPLATE_TEST_CASE_2( "convmtx/convmtx_1", "convmtx", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};

    auto c = convmtx(a, 4);

    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 1) == 2);
    REQUIRE(c(0, 2) == 3);
    REQUIRE(c(0, 3) == 0);
    REQUIRE(c(0, 4) == 0);
    REQUIRE(c(0, 5) == 0);

    REQUIRE(c(1, 0) == 0);
    REQUIRE(c(1, 1) == 1);
    REQUIRE(c(1, 2) == 2);
    REQUIRE(c(1, 3) == 3);
    REQUIRE(c(1, 4) == 0);
    REQUIRE(c(1, 5) == 0);

    REQUIRE(c(2, 0) == 0);
    REQUIRE(c(2, 1) == 0);
    REQUIRE(c(2, 2) == 1);
    REQUIRE(c(2, 3) == 2);
    REQUIRE(c(2, 4) == 3);
    REQUIRE(c(2, 5) == 0);

    REQUIRE(c(3, 0) == 0);
    REQUIRE(c(3, 1) == 0);
    REQUIRE(c(3, 2) == 0);
    REQUIRE(c(3, 3) == 1);
    REQUIRE(c(3, 4) == 2);
    REQUIRE(c(3, 5) == 3);
}

TEMPLATE_TEST_CASE_2( "convmtx/convmtx_2", "convmtx", Z, float, double ) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 4, 6> c;

    c = convmtx(a, 4);

    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 1) == 2);
    REQUIRE(c(0, 2) == 3);
    REQUIRE(c(0, 3) == 0);
    REQUIRE(c(0, 4) == 0);
    REQUIRE(c(0, 5) == 0);

    REQUIRE(c(1, 0) == 0);
    REQUIRE(c(1, 1) == 1);
    REQUIRE(c(1, 2) == 2);
    REQUIRE(c(1, 3) == 3);
    REQUIRE(c(1, 4) == 0);
    REQUIRE(c(1, 5) == 0);

    REQUIRE(c(2, 0) == 0);
    REQUIRE(c(2, 1) == 0);
    REQUIRE(c(2, 2) == 1);
    REQUIRE(c(2, 3) == 2);
    REQUIRE(c(2, 4) == 3);
    REQUIRE(c(2, 5) == 0);

    REQUIRE(c(3, 0) == 0);
    REQUIRE(c(3, 1) == 0);
    REQUIRE(c(3, 2) == 0);
    REQUIRE(c(3, 3) == 1);
    REQUIRE(c(3, 4) == 2);
    REQUIRE(c(3, 5) == 3);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_1", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 1, 1> K(etl::magic<Z>(1));
    etl::fast_matrix<Z, 3*3, 1> C1;
    etl::fast_matrix<Z, 3*3, 1> C2;

    C1 = etl::convmtx2(I, 1, 1);
    C2 = etl::convmtx2_direct<1, 1>(I);

    REQUIRE(C1(0, 0) == 8);
    REQUIRE(C1(1, 0) == 3);
    REQUIRE(C1(2, 0) == 4);
    REQUIRE(C1(3, 0) == 1);
    REQUIRE(C1(4, 0) == 5);
    REQUIRE(C1(5, 0) == 9);
    REQUIRE(C1(6, 0) == 6);
    REQUIRE(C1(7, 0) == 7);
    REQUIRE(C1(8, 0) == 2);

    REQUIRE(C1 == C2);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_2", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 2, 2> K(etl::magic<Z>(2));
    etl::fast_matrix<Z, 4*4, 4> C1;
    etl::fast_matrix<Z, 4*4, 4> C2;

    C1 = etl::convmtx2(I, 2, 2);
    C2 = etl::convmtx2_direct<2, 2>(I);

    REQUIRE(C1(0, 0) == 8);
    REQUIRE(C1(0, 1) == 0);
    REQUIRE(C1(0, 2) == 0);
    REQUIRE(C1(0, 3) == 0);

    REQUIRE(C1(1, 0) == 3);
    REQUIRE(C1(1, 1) == 8);
    REQUIRE(C1(1, 2) == 0);
    REQUIRE(C1(1, 3) == 0);

    REQUIRE(C1(2, 0) == 4);
    REQUIRE(C1(2, 1) == 3);
    REQUIRE(C1(2, 2) == 0);
    REQUIRE(C1(2, 3) == 0);

    REQUIRE(C1(3, 0) == 0);
    REQUIRE(C1(3, 1) == 4);
    REQUIRE(C1(3, 2) == 0);
    REQUIRE(C1(3, 3) == 0);

    REQUIRE(C1(10, 0) == 2);
    REQUIRE(C1(10, 1) == 7);
    REQUIRE(C1(10, 2) == 9);
    REQUIRE(C1(10, 3) == 5);

    REQUIRE(C1(15, 0) == 0);
    REQUIRE(C1(15, 1) == 0);
    REQUIRE(C1(15, 2) == 0);
    REQUIRE(C1(15, 3) == 2);

    REQUIRE(C1 == C2);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_3", "convmtx conv", Z, double, float ) {
    etl::fast_matrix<Z, 3, 3> I(etl::magic<Z>(3));
    etl::fast_matrix<Z, 2, 2> K(etl::magic<Z>(2));

    etl::fast_matrix<Z, 4*4, 4> C1(etl::convmtx2(I, 2, 2));
    etl::fast_matrix<Z, 4*4, 4> C2(etl::convmtx2_direct<2, 2>(I));

    etl::fast_matrix<Z, 16, 1> a1(etl::mul(C1, etl::reshape<4, 1>(etl::transpose(K))));
    etl::fast_matrix<Z, 4, 4> b1(etl::transpose(etl::reshape<4, 4>(a1)));

    etl::fast_matrix<Z, 16, 1> a2(etl::mul(C2, etl::reshape<4, 1>(etl::transpose(K))));
    etl::fast_matrix<Z, 4, 4> b2(etl::transpose(etl::reshape<4, 4>(a2)));

    etl::fast_matrix<Z, 4, 4> ref(etl::conv_2d_full(I, K));

    REQUIRE(ref == b1);
    REQUIRE(ref == b2);
}

TEMPLATE_TEST_CASE_2( "convmtx2/convmtx2_4", "convmtx conv", Z, double, float ) {
    etl::dyn_matrix<Z> I(etl::magic<Z>(7));
    etl::dyn_matrix<Z> K(etl::magic<Z>(5));

    etl::dyn_matrix<Z> C1(etl::convmtx2(I, 5, 5));
    etl::dyn_matrix<Z> C2(etl::convmtx2_direct<5, 5>(I));

    etl::dyn_matrix<Z> a1(etl::mul(C1, etl::reshape<25, 1>(etl::transpose(K))));
    etl::dyn_matrix<Z> b1(etl::transpose(etl::reshape(a1, 11, 11)));

    etl::dyn_matrix<Z> a2(etl::mul(C2, etl::reshape<25, 1>(etl::transpose(K))));
    etl::dyn_matrix<Z> b2(etl::transpose(etl::reshape(a2, 11, 11)));

    etl::dyn_matrix<Z> ref(etl::conv_2d_full(I, K));

    REQUIRE(ref == b1);
    REQUIRE(ref == b2);
}
