//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("pooling/stride/max2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 3, 3> b(etl::max_pool_2d<2, 2, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 2), 8.0);

    REQUIRE_EQUALS(b(1, 0), 10.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
    REQUIRE_EQUALS(b(1, 2), 12.0);

    REQUIRE_EQUALS(b(2, 0), 14.0);
    REQUIRE_EQUALS(b(2, 1), 15.0);
    REQUIRE_EQUALS(b(2, 2), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::max_pool_2d<3, 3, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 1), 12.0);

    REQUIRE_EQUALS(b(1, 0), 15.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b(etl::max_pool_2d<4, 4, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::max_pool_2d<1, 1, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg2/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 3, 3> b(etl::avg_pool_2d<2, 2, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 1), 4.5);
    REQUIRE_EQUALS(b(0, 2), 5.5);

    REQUIRE_EQUALS(b(1, 0), 7.5);
    REQUIRE_EQUALS(b(1, 1), 8.5);
    REQUIRE_EQUALS(b(1, 2), 9.5);

    REQUIRE_EQUALS(b(2, 0), 11.5);
    REQUIRE_EQUALS(b(2, 1), 12.5);
    REQUIRE_EQUALS(b(2, 2), 13.5);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::avg_pool_2d<3, 3, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);

    REQUIRE_EQUALS(b(1, 0), 10.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b(etl::avg_pool_2d<4, 4, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2> b(etl::avg_pool_2d<1, 1, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 3, 3> b(etl::max_pool_3d<1, 2, 2, 1, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 0, 2), 8.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
    REQUIRE_EQUALS(b(0, 1, 2), 12.0);

    REQUIRE_EQUALS(b(0, 2, 0), 14.0);
    REQUIRE_EQUALS(b(0, 2, 1), 15.0);
    REQUIRE_EQUALS(b(0, 2, 2), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::max_pool_3d<1, 3, 3, 1, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 0, 1), 12.0);

    REQUIRE_EQUALS(b(0, 1, 0), 15.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1, 1> b(etl::max_pool_3d<1, 4, 4, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/max3/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::max_pool_3d<1, 1, 1, 1, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 3, 3> b(etl::avg_pool_3d<1, 2, 2, 1, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 0, 1), 4.5);
    REQUIRE_EQUALS(b(0, 0, 2), 5.5);

    REQUIRE_EQUALS(b(0, 1, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1, 1), 8.5);
    REQUIRE_EQUALS(b(0, 1, 2), 9.5);

    REQUIRE_EQUALS(b(0, 2, 0), 11.5);
    REQUIRE_EQUALS(b(0, 2, 1), 12.5);
    REQUIRE_EQUALS(b(0, 2, 2), 13.5);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::avg_pool_3d<1, 3, 3, 1, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1, 1> b(etl::avg_pool_3d<1, 4, 4, 1, 1, 1>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/stride/avg3/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b(etl::avg_pool_3d<1, 1, 1, 1, 2, 2>(a));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}
