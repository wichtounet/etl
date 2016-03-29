//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

// Note: The ugliness of the some comparisons is due to Catch being
// a douche

TEMPLATE_TEST_CASE_2("globals/lu/1", "[globals][LU]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> A{1, 3, 5, 2, 4, 7, 1, 1, 0};
    etl::fast_matrix<Z, 3, 3> L;
    etl::fast_matrix<Z, 3, 3> U;
    etl::fast_matrix<Z, 3, 3> P;

    etl::lu(A, L, U, P);

    REQUIRE((P * A == L * U) == true);

    REQUIRE(L(0, 0) == 1.0);
    REQUIRE(L(0, 1) == 0.0);
    REQUIRE(L(0, 2) == 0.0);
    REQUIRE(L(1, 0) == 0.5);
    REQUIRE(L(1, 1) == 1.0);
    REQUIRE(L(1, 2) == 0.0);
    REQUIRE(L(2, 0) == 0.5);
    REQUIRE(L(2, 1) == -1.0);
    REQUIRE(L(2, 2) == 1.0);

    REQUIRE(U(0, 0) == 2.0);
    REQUIRE(U(0, 1) == 4.0);
    REQUIRE(U(0, 2) == 7.0);
    REQUIRE(U(1, 0) == 0.0);
    REQUIRE(U(1, 1) == 1.0);
    REQUIRE(U(1, 2) == 1.5);
    REQUIRE(U(2, 0) == 0.0);
    REQUIRE(U(2, 1) == 0.0);
    REQUIRE(U(2, 2) == -2.0);

    REQUIRE(P(0, 0) == 0.0);
    REQUIRE(P(0, 1) == 1.0);
    REQUIRE(P(0, 2) == 0.0);
    REQUIRE(P(1, 0) == 1.0);
    REQUIRE(P(1, 1) == 0.0);
    REQUIRE(P(1, 2) == 0.0);
    REQUIRE(P(2, 0) == 0.0);
    REQUIRE(P(2, 1) == 0.0);
    REQUIRE(P(2, 2) == 1.0);
}

TEMPLATE_TEST_CASE_2("globals/lu/2", "[globals][LU]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> A{11, 9, 24, 2, 1, 5, 2, 6, 3, 17, 18, 1, 2, 5, 7, 1};
    etl::fast_matrix<Z, 4, 4> L;
    etl::fast_matrix<Z, 4, 4> U;
    etl::fast_matrix<Z, 4, 4> P;

    etl::lu(A, L, U, P);

    REQUIRE((P * A == L * U) == true);

    REQUIRE(L(0, 0) == Approx(Z(1.0)));
    REQUIRE(L(0, 1) == Approx(Z(0.0)));
    REQUIRE(L(0, 2) == Approx(Z(0.0)));
    REQUIRE(L(0, 3) == Approx(Z(0.0)));
    REQUIRE(L(1, 0) == Approx(Z(0.27273)));
    REQUIRE(L(1, 1) == Approx(Z(1.0)));
    REQUIRE(L(1, 2) == Approx(Z(0.0)));
    REQUIRE(L(1, 3) == Approx(Z(0.0)));
    REQUIRE(L(2, 0) == Approx(Z(0.09091)));
    REQUIRE(L(2, 1) == Approx(Z(0.28750)));
    REQUIRE(L(2, 2) == Approx(Z(1.0)));
    REQUIRE(L(2, 3) == Approx(Z(0.0)));
    REQUIRE(L(3, 0) == Approx(Z(0.18182)));
    REQUIRE(L(3, 1) == Approx(Z(0.23125)));
    REQUIRE(L(3, 2) == Approx(Z(0.00360)));
    REQUIRE(L(3, 3) == Approx(Z(1.0)));

    REQUIRE(U(0, 0) == Approx(Z(11.0)));
    REQUIRE(U(0, 1) == Approx(Z(9.0)));
    REQUIRE(U(0, 2) == Approx(Z(24.0)));
    REQUIRE(U(0, 3) == Approx(Z(2.0)));
    REQUIRE(U(1, 0) == Approx(Z(0.0)));
    REQUIRE(U(1, 1) == Approx(Z(14.54545)));
    REQUIRE(U(1, 2) == Approx(Z(11.45455)));
    REQUIRE(U(1, 3) == Approx(Z(0.45455)));
    REQUIRE(U(2, 0) == Approx(Z(0.0)));
    REQUIRE(U(2, 1) == Approx(Z(0.0)));
    REQUIRE(U(2, 2) == Approx(Z(-3.475)));
    REQUIRE(U(2, 3) == Approx(Z(5.6875)));
    REQUIRE(U(3, 0) == Approx(Z(0.0)));
    REQUIRE(U(3, 1) == Approx(Z(0.0)));
    REQUIRE(U(3, 2) == Approx(Z(0.0)));
    REQUIRE(U(3, 3) == Approx(Z(0.51079)));

    REQUIRE(P(0, 0) == 1.0);
    REQUIRE(P(0, 1) == 0.0);
    REQUIRE(P(0, 2) == 0.0);
    REQUIRE(P(1, 0) == 0.0);
    REQUIRE(P(0, 3) == 0.0);
    REQUIRE(P(1, 1) == 0.0);
    REQUIRE(P(1, 2) == 1.0);
    REQUIRE(P(1, 3) == 0.0);
    REQUIRE(P(2, 0) == 0.0);
    REQUIRE(P(2, 1) == 1.0);
    REQUIRE(P(2, 2) == 0.0);
    REQUIRE(P(2, 3) == 0.0);
    REQUIRE(P(3, 0) == 0.0);
    REQUIRE(P(3, 1) == 0.0);
    REQUIRE(P(3, 2) == 0.0);
    REQUIRE(P(3, 3) == 1.0);
}
