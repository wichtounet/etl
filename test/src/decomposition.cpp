//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

/* LU */

TEMPLATE_TEST_CASE_2("globals/lu/1", "[globals][LU]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> A{1, 3, 5, 2, 4, 7, 1, 1, 0};
    etl::fast_matrix<Z, 3, 3> L;
    etl::fast_matrix<Z, 3, 3> U;
    etl::fast_matrix<Z, 3, 3> P;

    etl::lu(A, L, U, P);

    etl::fast_matrix<Z, 3, 3> PA;
    etl::fast_matrix<Z, 3, 3> LU;
    PA = P * A;
    LU = L * U;

    REQUIRE_DIRECT(approx_equals(PA, LU, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("globals/lu/2", "[globals][LU]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> A{11, 9, 24, 2, 1, 5, 2, 6, 3, 17, 18, 1, 2, 5, 7, 1};
    etl::fast_matrix<Z, 4, 4> L;
    etl::fast_matrix<Z, 4, 4> U;
    etl::fast_matrix<Z, 4, 4> P;

    etl::lu(A, L, U, P);

    etl::fast_matrix<Z, 4, 4> PA;
    etl::fast_matrix<Z, 4, 4> LU;
    PA = P * A;
    LU = L * U;

    REQUIRE_DIRECT(approx_equals(PA, LU, base_eps_etl));
}

/* QR */

TEMPLATE_TEST_CASE_2("globals/qr/1", "[globals][QR]", Z, float, double) {
    etl::fast_matrix<Z, 5, 3> A{12, -51, 4, 6, 167, -68, -4, 24, -41, -1, 1, 44, 2, 11, 3};
    etl::fast_matrix<Z, 5, 5> Q;
    etl::fast_matrix<Z, 5, 3> R;
    etl::fast_matrix<Z, 5, 3> QR;

    etl::qr(A, Q, R);

    QR = Q * R;

    // The epsilon need to be big because of the zero in the result
    // and the large difference in computation around zero
    REQUIRE_DIRECT(approx_equals(QR, A, 100 * base_eps_etl));
}
