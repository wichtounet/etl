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

    REQUIRE_EQUALS((P * A == L * U), true);

    REQUIRE_EQUALS(L(0, 0), 1.0);
    REQUIRE_EQUALS(L(0, 1), 0.0);
    REQUIRE_EQUALS(L(0, 2), 0.0);
    REQUIRE_EQUALS(L(1, 0), 0.5);
    REQUIRE_EQUALS(L(1, 1), 1.0);
    REQUIRE_EQUALS(L(1, 2), 0.0);
    REQUIRE_EQUALS(L(2, 0), 0.5);
    REQUIRE_EQUALS(L(2, 1), -1.0);
    REQUIRE_EQUALS(L(2, 2), 1.0);

    REQUIRE_EQUALS(U(0, 0), 2.0);
    REQUIRE_EQUALS(U(0, 1), 4.0);
    REQUIRE_EQUALS(U(0, 2), 7.0);
    REQUIRE_EQUALS(U(1, 0), 0.0);
    REQUIRE_EQUALS(U(1, 1), 1.0);
    REQUIRE_EQUALS(U(1, 2), 1.5);
    REQUIRE_EQUALS(U(2, 0), 0.0);
    REQUIRE_EQUALS(U(2, 1), 0.0);
    REQUIRE_EQUALS(U(2, 2), -2.0);

    REQUIRE_EQUALS(P(0, 0), 0.0);
    REQUIRE_EQUALS(P(0, 1), 1.0);
    REQUIRE_EQUALS(P(0, 2), 0.0);
    REQUIRE_EQUALS(P(1, 0), 1.0);
    REQUIRE_EQUALS(P(1, 1), 0.0);
    REQUIRE_EQUALS(P(1, 2), 0.0);
    REQUIRE_EQUALS(P(2, 0), 0.0);
    REQUIRE_EQUALS(P(2, 1), 0.0);
    REQUIRE_EQUALS(P(2, 2), 1.0);
}

TEMPLATE_TEST_CASE_2("globals/lu/2", "[globals][LU]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> A{11, 9, 24, 2, 1, 5, 2, 6, 3, 17, 18, 1, 2, 5, 7, 1};
    etl::fast_matrix<Z, 4, 4> L;
    etl::fast_matrix<Z, 4, 4> U;
    etl::fast_matrix<Z, 4, 4> P;

    etl::lu(A, L, U, P);

    REQUIRE_EQUALS((P * A == L * U), true);

    REQUIRE_EQUALS_APPROX(L(0, 0), Z(1.0));
    REQUIRE_EQUALS_APPROX(L(0, 1), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(0, 2), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(0, 3), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(1, 0), Z(0.27273));
    REQUIRE_EQUALS_APPROX(L(1, 1), Z(1.0));
    REQUIRE_EQUALS_APPROX(L(1, 2), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(1, 3), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(2, 0), Z(0.09091));
    REQUIRE_EQUALS_APPROX(L(2, 1), Z(0.28750));
    REQUIRE_EQUALS_APPROX(L(2, 2), Z(1.0));
    REQUIRE_EQUALS_APPROX(L(2, 3), Z(0.0));
    REQUIRE_EQUALS_APPROX(L(3, 0), Z(0.18182));
    REQUIRE_EQUALS_APPROX(L(3, 1), Z(0.23125));
    REQUIRE_EQUALS_APPROX(L(3, 2), Z(0.00360));
    REQUIRE_EQUALS_APPROX(L(3, 3), Z(1.0));

    REQUIRE_EQUALS_APPROX(U(0, 0), Z(11.0));
    REQUIRE_EQUALS_APPROX(U(0, 1), Z(9.0));
    REQUIRE_EQUALS_APPROX(U(0, 2), Z(24.0));
    REQUIRE_EQUALS_APPROX(U(0, 3), Z(2.0));
    REQUIRE_EQUALS_APPROX(U(1, 0), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(1, 1), Z(14.54545));
    REQUIRE_EQUALS_APPROX(U(1, 2), Z(11.45455));
    REQUIRE_EQUALS_APPROX(U(1, 3), Z(0.45455));
    REQUIRE_EQUALS_APPROX(U(2, 0), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(2, 1), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(2, 2), Z(-3.475));
    REQUIRE_EQUALS_APPROX(U(2, 3), Z(5.6875));
    REQUIRE_EQUALS_APPROX(U(3, 0), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(3, 1), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(3, 2), Z(0.0));
    REQUIRE_EQUALS_APPROX(U(3, 3), Z(0.51079));

    REQUIRE_EQUALS(P(0, 0), 1.0);
    REQUIRE_EQUALS(P(0, 1), 0.0);
    REQUIRE_EQUALS(P(0, 2), 0.0);
    REQUIRE_EQUALS(P(1, 0), 0.0);
    REQUIRE_EQUALS(P(0, 3), 0.0);
    REQUIRE_EQUALS(P(1, 1), 0.0);
    REQUIRE_EQUALS(P(1, 2), 1.0);
    REQUIRE_EQUALS(P(1, 3), 0.0);
    REQUIRE_EQUALS(P(2, 0), 0.0);
    REQUIRE_EQUALS(P(2, 1), 1.0);
    REQUIRE_EQUALS(P(2, 2), 0.0);
    REQUIRE_EQUALS(P(2, 3), 0.0);
    REQUIRE_EQUALS(P(3, 0), 0.0);
    REQUIRE_EQUALS(P(3, 1), 0.0);
    REQUIRE_EQUALS(P(3, 2), 0.0);
    REQUIRE_EQUALS(P(3, 3), 1.0);
}
