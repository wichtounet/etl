//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/max2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);
    etl::dyn_matrix<Z, 2> c(4, 4);

    b = etl::max_pool_2d(a, 2, 2);
    c = etl::max_pool_derivative_2d(a, b, 2, 2);

    REQUIRE_EQUALS(c(0, 0), 0.0);
    REQUIRE_EQUALS(c(0, 1), 0.0);
    REQUIRE_EQUALS(c(1, 0), 0.0);
    REQUIRE_EQUALS(c(1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 2), 0.0);
    REQUIRE_EQUALS(c(0, 3), 0.0);
    REQUIRE_EQUALS(c(1, 2), 0.0);
    REQUIRE_EQUALS(c(1, 3), 1.0);

    REQUIRE_EQUALS(c(2, 0), 0.0);
    REQUIRE_EQUALS(c(2, 1), 0.0);
    REQUIRE_EQUALS(c(3, 0), 0.0);
    REQUIRE_EQUALS(c(3, 1), 1.0);

    REQUIRE_EQUALS(c(2, 2), 0.0);
    REQUIRE_EQUALS(c(2, 3), 0.0);
    REQUIRE_EQUALS(c(3, 2), 0.0);
    REQUIRE_EQUALS(c(3, 3), 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/avg2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);
    etl::dyn_matrix<Z, 2> c(4, 4);

    b = etl::avg_pool_2d(a, 2, 2);
    c = etl::avg_pool_derivative_2d(a, b, 2, 2);

    REQUIRE_EQUALS(c(0, 0), 0.25);
    REQUIRE_EQUALS(c(0, 1), 0.25);
    REQUIRE_EQUALS(c(1, 0), 0.25);
    REQUIRE_EQUALS(c(1, 1), 0.25);
}

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/max3/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<Z, 3> b(1, 2, 2);
    etl::dyn_matrix<Z, 3> c(2, 4, 4);

    b = etl::max_pool_3d(a, 2, 2, 2);
    c = etl::max_pool_derivative_3d(a, b, 2, 2, 2);

    REQUIRE_EQUALS(c(0, 0, 0), 0.0);
    REQUIRE_EQUALS(c(0, 0, 1), 0.0);
    REQUIRE_EQUALS(c(0, 1, 0), 0.0);
    REQUIRE_EQUALS(c(0, 1, 1), 0.0);

    REQUIRE_EQUALS(c(1, 0, 0), 0.0);
    REQUIRE_EQUALS(c(1, 0, 1), 0.0);
    REQUIRE_EQUALS(c(1, 1, 0), 0.0);
    REQUIRE_EQUALS(c(1, 1, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/avg3/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<Z, 3> b(1, 2, 2);
    etl::dyn_matrix<Z, 3> c(2, 4, 4);

    b = etl::avg_pool_3d(a, 2, 2, 2);
    c = etl::avg_pool_derivative_3d(a, b, 2, 2, 2);

    REQUIRE_EQUALS(c(0, 0, 0), 0.125);
    REQUIRE_EQUALS(c(0, 0, 1), 0.125);
    REQUIRE_EQUALS(c(0, 1, 0), 0.125);
    REQUIRE_EQUALS(c(0, 1, 1), 0.125);
}

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/deep/max2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> A(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 3> a(2, 4, 4);

    a(0) = A;
    a(1) = A;

    etl::dyn_matrix<Z, 3> b(2, 2, 2);
    etl::dyn_matrix<Z, 3> c(2, 4, 4);

    b = etl::max_pool_2d(a, 2, 2);
    c = etl::max_pool_derivative_2d(a, b, 2, 2);

    REQUIRE_EQUALS(c(0, 0, 0), 0.0);
    REQUIRE_EQUALS(c(0, 0, 1), 0.0);
    REQUIRE_EQUALS(c(0, 1, 0), 0.0);
    REQUIRE_EQUALS(c(0, 1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 0, 2), 0.0);
    REQUIRE_EQUALS(c(0, 0, 3), 0.0);
    REQUIRE_EQUALS(c(0, 1, 2), 0.0);
    REQUIRE_EQUALS(c(0, 1, 3), 1.0);

    REQUIRE_EQUALS(c(0, 2, 0), 0.0);
    REQUIRE_EQUALS(c(0, 2, 1), 0.0);
    REQUIRE_EQUALS(c(0, 3, 0), 0.0);
    REQUIRE_EQUALS(c(0, 3, 1), 1.0);

    REQUIRE_EQUALS(c(0, 2, 2), 0.0);
    REQUIRE_EQUALS(c(0, 2, 3), 0.0);
    REQUIRE_EQUALS(c(0, 3, 2), 0.0);
    REQUIRE_EQUALS(c(0, 3, 3), 1.0);

    REQUIRE_EQUALS(c(1, 0, 0), 0.0);
    REQUIRE_EQUALS(c(1, 0, 1), 0.0);
    REQUIRE_EQUALS(c(1, 1, 0), 0.0);
    REQUIRE_EQUALS(c(1, 1, 1), 1.0);

    REQUIRE_EQUALS(c(1, 0, 2), 0.0);
    REQUIRE_EQUALS(c(1, 0, 3), 0.0);
    REQUIRE_EQUALS(c(1, 1, 2), 0.0);
    REQUIRE_EQUALS(c(1, 1, 3), 1.0);

    REQUIRE_EQUALS(c(1, 2, 0), 0.0);
    REQUIRE_EQUALS(c(1, 2, 1), 0.0);
    REQUIRE_EQUALS(c(1, 3, 0), 0.0);
    REQUIRE_EQUALS(c(1, 3, 1), 1.0);

    REQUIRE_EQUALS(c(1, 2, 2), 0.0);
    REQUIRE_EQUALS(c(1, 2, 3), 0.0);
    REQUIRE_EQUALS(c(1, 3, 2), 0.0);
    REQUIRE_EQUALS(c(1, 3, 3), 1.0);
}

TEMPLATE_TEST_CASE_2("dyn_pool_derivative/deep/max3/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> A(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<Z, 4> a(2, 2, 4, 4);

    a(0) = A;
    a(1) = A;

    etl::dyn_matrix<Z, 4> b(2, 1, 2, 2);
    etl::dyn_matrix<Z, 4> c(2, 2, 4, 4);

    b = etl::max_pool_3d(a, 2, 2, 2);
    c = etl::max_pool_derivative_3d(a, b, 2, 2, 2);

    REQUIRE_EQUALS(c(0, 0, 0, 0), 0.0);
    REQUIRE_EQUALS(c(0, 0, 0, 1), 0.0);
    REQUIRE_EQUALS(c(0, 0, 1, 0), 0.0);
    REQUIRE_EQUALS(c(0, 0, 1, 1), 0.0);

    REQUIRE_EQUALS(c(0, 1, 0, 0), 0.0);
    REQUIRE_EQUALS(c(0, 1, 0, 1), 0.0);
    REQUIRE_EQUALS(c(0, 1, 1, 0), 0.0);
    REQUIRE_EQUALS(c(0, 1, 1, 1), 1.0);

    REQUIRE_EQUALS(c(1, 0, 0, 0), 0.0);
    REQUIRE_EQUALS(c(1, 0, 0, 1), 0.0);
    REQUIRE_EQUALS(c(1, 0, 1, 0), 0.0);
    REQUIRE_EQUALS(c(1, 0, 1, 1), 0.0);

    REQUIRE_EQUALS(c(1, 1, 0, 0), 0.0);
    REQUIRE_EQUALS(c(1, 1, 0, 1), 0.0);
    REQUIRE_EQUALS(c(1, 1, 1, 0), 0.0);
    REQUIRE_EQUALS(c(1, 1, 1, 1), 1.0);
}
