//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pooling/avg2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b;
    b = etl::avg_pool_2d<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b;
    b = etl::avg_pool_2d<4, 4>(a);

    REQUIRE_EQUALS(b(0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2> b;
    b = etl::avg_pool_2d<4, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1), 9.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 1> b;
    b = etl::avg_pool_2d<2, 4>(a);

    REQUIRE_EQUALS(b(0, 0), 4.5);
    REQUIRE_EQUALS(b(1, 0), 12.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 4, 4> a;
    a(0) = aa;
    a(1) = aa;

    etl::fast_matrix<Z, 2, 2, 2> b;
    b = etl::avg_pool_2d<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 0, 1), 5.5);
    REQUIRE_EQUALS(b(0, 1, 0), 11.5);
    REQUIRE_EQUALS(b(0, 1, 1), 13.5);

    REQUIRE_EQUALS(b(1, 0, 0), 3.5);
    REQUIRE_EQUALS(b(1, 0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2, 4, 4> a;
    a(0)(0) = aa;
    a(0)(1) = aa;
    a(1)(0) = aa;
    a(1)(1) = aa;

    etl::fast_matrix<Z, 2, 2, 2, 2> b;
    b = etl::avg_pool_2d<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 5.5);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 11.5);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 13.5);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 5.5);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 11.5);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 13.5);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 13.5);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 3.5);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 13.5);
}

// Dynamic versions

TEMPLATE_TEST_CASE_2("dyn_pooling/avg2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = etl::avg_pool_2d(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/avg2/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(1, 1);

    b = etl::avg_pool_2d(a, 4, 4);

    REQUIRE_EQUALS(b(0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/avg2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(1, 2);

    b = etl::avg_pool_2d(a, 4, 2);

    REQUIRE_EQUALS(b(0, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1), 9.5);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/avg2/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 1);

    b = etl::avg_pool_2d(a, 2, 4);

    REQUIRE_EQUALS(b(0, 0), 4.5);
    REQUIRE_EQUALS(b(1, 0), 12.5);
}

