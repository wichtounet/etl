//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pooling/max2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::max_pool_2d<2, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b(etl::max_pool_2d<4, 4>(a));

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2> b(etl::max_pool_2d<4, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 1> b(etl::max_pool_2d<2, 4>(a));

    REQUIRE_EQUALS(b(0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::avg_pool_2d<2, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 5.5);
    REQUIRE_EQUALS(b(1, 0), 11.5);
    REQUIRE_EQUALS(b(1, 1), 13.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b(etl::avg_pool_2d<4, 4>(a));

    REQUIRE_EQUALS(b(0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2> b(etl::avg_pool_2d<4, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1), 9.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 1> b(etl::avg_pool_2d<2, 4>(a));

    REQUIRE_EQUALS(b(0, 0), 4.5);
    REQUIRE_EQUALS(b(1, 0), 12.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::avg_pool_3d<2, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 11.5);
    REQUIRE_EQUALS(b(0, 0, 1), 13.5);
    REQUIRE_EQUALS(b(0, 1, 0), 19.5);
    REQUIRE_EQUALS(b(0, 1, 1), 21.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 1, 1, 2> b(etl::avg_pool_3d<2, 4, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 15.5);
    REQUIRE_EQUALS(b(0, 0, 1), 17.5);
}

TEMPLATE_TEST_CASE_2("pooling/avg3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 2, 1, 1> b(etl::avg_pool_3d<1, 4, 4>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
    REQUIRE_EQUALS(b(1, 0, 0), 24.5);
}

TEMPLATE_TEST_CASE_2("pooling/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::max_pool_3d<2, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 22.0);
    REQUIRE_EQUALS(b(0, 0, 1), 24.0);
    REQUIRE_EQUALS(b(0, 1, 0), 30.0);
    REQUIRE_EQUALS(b(0, 1, 1), 32.0);
}

TEMPLATE_TEST_CASE_2("pooling/max3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 1, 1, 2> b(etl::max_pool_3d<2, 4, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 30.0);
    REQUIRE_EQUALS(b(0, 0, 1), 32.0);
}

TEMPLATE_TEST_CASE_2("pooling/max3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<Z, 2, 1, 1> b(etl::max_pool_3d<1, 4, 4>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
    REQUIRE_EQUALS(b(1, 0, 0), 32.0);
}
