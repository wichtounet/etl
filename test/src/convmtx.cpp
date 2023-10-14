//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

//Note: The results of the tests have been validated with one of (octave/matlab/matlab)

TEMPLATE_TEST_CASE_2("convmtx/convmtx_1", "convmtx", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};

    auto c = convmtx(a, 4);

    REQUIRE_EQUALS_APPROX_E(c(0, 0), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 3), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 3), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(2, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 2), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 3), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 4), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(3, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 3), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 4), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 5), 3, base_eps / 10);
}

TEMPLATE_TEST_CASE_2("convmtx/convmtx_2", "convmtx", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix<Z, 4, 6> c;

    c = convmtx(a, 4);

    REQUIRE_EQUALS_APPROX_E(c(0, 0), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 3), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 3), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(2, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 2), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 3), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 4), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(3, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 3), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 4), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 5), 3, base_eps / 10);
}

TEMPLATE_TEST_CASE_2("convmtx/convmtx_3", "convmtx", Z, float, double) {
    etl::fast_vector_cm<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_matrix_cm<Z, 4, 6> c;
    c = convmtx(a, 4);

    REQUIRE_EQUALS_APPROX_E(c(0, 0), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 1), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 2), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 3), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(0, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(1, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 1), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 2), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 3), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 4), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(1, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(2, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 2), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 3), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 4), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(2, 5), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(c(3, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 3), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 4), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(c(3, 5), 3, base_eps / 10);
}

TEMPLATE_TEST_CASE_2("convmtx2/convmtx2_1", "convmtx conv", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I;
    etl::fast_matrix<Z, 1, 1> K;
    etl::fast_matrix<Z, 3 * 3, 1> C1;
    etl::fast_matrix<Z, 3 * 3, 1> C2;

    I = etl::magic<Z>(3);
    K = etl::magic<Z>(1);

    C1 = etl::convmtx2(I, 1, 1);
    C2 = etl::convmtx2_direct<1, 1>(I);

    REQUIRE_EQUALS_APPROX_E(C1(0, 0), 8, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(1, 0), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(2, 0), 4, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(3, 0), 1, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(4, 0), 5, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(5, 0), 9, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(6, 0), 6, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(7, 0), 7, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(8, 0), 2, base_eps / 10);

    for(size_t i = 0; i < C1.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(C1[i], C2[i], base_eps / 10);
    }
}

TEMPLATE_TEST_CASE_2("convmtx2/convmtx2_2", "convmtx conv", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I;
    etl::fast_matrix<Z, 2, 2> K;
    etl::fast_matrix<Z, 4 * 4, 4> C1;
    etl::fast_matrix<Z, 4 * 4, 4> C2;

    I = etl::magic<Z>(3);
    K = etl::magic<Z>(2);

    C1 = etl::convmtx2(I, 2, 2);
    C2 = etl::convmtx2_direct<2, 2>(I);

    REQUIRE_EQUALS_APPROX_E(C1(0, 0), 8, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(0, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(0, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(0, 3), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(C1(1, 0), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(1, 1), 8, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(1, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(1, 3), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(C1(2, 0), 4, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(2, 1), 3, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(2, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(2, 3), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(C1(3, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(3, 1), 4, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(3, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(3, 3), 0, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(C1(10, 0), 2, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(10, 1), 7, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(10, 2), 9, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(10, 3), 5, base_eps / 10);

    REQUIRE_EQUALS_APPROX_E(C1(15, 0), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(15, 1), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(15, 2), 0, base_eps / 10);
    REQUIRE_EQUALS_APPROX_E(C1(15, 3), 2, base_eps / 10);

    for(size_t i = 0; i < C1.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(C1[i], C2[i], base_eps / 10);
    }
}

TEMPLATE_TEST_CASE_2("convmtx2/convmtx2_3", "convmtx conv", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> I;
    etl::fast_matrix<Z, 2, 2> K;

    etl::fast_matrix<Z, 4 * 4, 4> C1;
    etl::fast_matrix<Z, 4 * 4, 4> C2;

    etl::fast_matrix<Z, 16, 1> a1;
    etl::fast_matrix<Z, 4, 4> b1;

    etl::fast_matrix<Z, 16, 1> a2;
    etl::fast_matrix<Z, 4, 4> b2;

    etl::fast_matrix<Z, 4, 4> ref;

    I = etl::magic<Z>(3);
    K = etl::magic<Z>(2);

    C1 = etl::convmtx2(I, 2, 2);
    C2 = etl::convmtx2_direct<2, 2>(I);

    a1 = etl::mul(C1, etl::reshape<4, 1>(etl::transpose(K)));
    b1 = etl::transpose(etl::reshape<4, 4>(a1));

    a2 = etl::mul(C2, etl::reshape<4, 1>(etl::transpose(K)));
    b2 = etl::transpose(etl::reshape<4, 4>(a2));

    ref = etl::conv_2d_full(I, K);

    for(size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(ref[i], b1[i], base_eps / 10);
        REQUIRE_EQUALS_APPROX_E(ref[i], b2[i], base_eps / 10);
    }
}

TEMPLATE_TEST_CASE_2("convmtx2/convmtx2_4", "convmtx conv", Z, double, float) {
    etl::dyn_matrix<Z> I;
    etl::dyn_matrix<Z> K;

    etl::dyn_matrix<Z> C1;
    etl::dyn_matrix<Z> C2;

    etl::dyn_matrix<Z> a1;
    etl::dyn_matrix<Z> b1;

    etl::dyn_matrix<Z> a2;
    etl::dyn_matrix<Z> b2;

    I = etl::magic<Z>(7);
    K = etl::magic<Z>(5);

    C1 = etl::convmtx2(I, 5, 5);
    C2 = etl::convmtx2_direct<5, 5>(I);

    a1 = etl::mul(C1, etl::reshape<25, 1>(etl::transpose(K)));
    b1 = etl::transpose(etl::reshape(a1, 11, 11));

    a2 = etl::mul(C2, etl::reshape<25, 1>(etl::transpose(K)));
    b2 = etl::transpose(etl::reshape(a2, 11, 11));

    etl::dyn_matrix<Z> ref;
    ref = etl::conv_2d_full(I, K);

    /*
     * TODO Fix these tests
     * for(size_t i = 0; i < ref.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(ref[i], b1[i], base_eps);
        REQUIRE_EQUALS_APPROX_E(ref[i], b2[i], base_eps);
    }*/
}
