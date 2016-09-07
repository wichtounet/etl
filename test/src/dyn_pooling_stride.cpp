//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

//TODO The 3D tests are really poor

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(3, 3);

    b = etl::max_pool_2d(a, 2, 2, 1, 1);

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

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = etl::max_pool_2d(a, 3, 3, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 1), 12.0);

    REQUIRE_EQUALS(b(1, 0), 15.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(1, 1);

    b = etl::max_pool_2d(a, 4, 4, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = etl::max_pool_2d(a, 1, 1, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/5", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = etl::max_pool_2d(a, 2, 2, 2, 2, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max2/6", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> b(3, 3);

    b = etl::max_pool_2d(a, 2, 2, 1, 1, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 2), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
    REQUIRE_EQUALS(b(1, 2), 4.0);

    REQUIRE_EQUALS(b(2, 0), 3.0);
    REQUIRE_EQUALS(b(2, 1), 4.0);
    REQUIRE_EQUALS(b(2, 2), 4.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(3, 3);

    b = (etl::avg_pool_2d(a, 2, 2, 1, 1));

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

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = (etl::avg_pool_2d(a, 3, 3, 1, 1));

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);

    REQUIRE_EQUALS(b(1, 0), 10.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(1, 1);

    b = (etl::avg_pool_2d(a, 4, 4, 1, 1));

    REQUIRE_EQUALS(b(0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = (etl::avg_pool_2d(a, 1, 1, 2, 2));

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/5", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> b(2, 2);

    b = (etl::avg_pool_2d(a, 2, 2, 2, 2, 1, 1));

    REQUIRE_EQUALS(b(0, 0), 0.25);
    REQUIRE_EQUALS(b(0, 1), 0.5);

    REQUIRE_EQUALS(b(1, 0), 0.75);
    REQUIRE_EQUALS(b(1, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg2/6", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> b(3, 3);

    b = (etl::avg_pool_2d(a, 2, 2, 1, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0), 0.25);
    REQUIRE_EQUALS(b(0, 1), 0.75);
    REQUIRE_EQUALS(b(0, 2), 0.5);

    REQUIRE_EQUALS(b(1, 0), 1.0);
    REQUIRE_EQUALS(b(1, 1), 2.5);
    REQUIRE_EQUALS(b(1, 2), 1.5);

    REQUIRE_EQUALS(b(2, 0), 0.75);
    REQUIRE_EQUALS(b(2, 1), 1.75);
    REQUIRE_EQUALS(b(2, 2), 1.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 3, 3> b;

    b = (etl::max_pool_3d(a, 1, 2, 2, 1, 1, 1));

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

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b;

    b = (etl::max_pool_3d(a, 1, 3, 3, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 0, 1), 12.0);

    REQUIRE_EQUALS(b(0, 1, 0), 15.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1, 1> b;

    b = (etl::max_pool_3d(a, 1, 4, 4, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b;

    b = (etl::max_pool_3d(a, 1, 1, 1, 1, 2, 2));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 2, 2, 2> b;

    b = (etl::max_pool_3d(a, 2, 2, 2,  2, 2, 2,  1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/max3/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 3, 3, 3> b;

    b = (etl::max_pool_3d(a, 2, 2, 2,  1, 1, 1,  1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 0, 2), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);
    REQUIRE_EQUALS(b(0, 1, 2), 4.0);
    REQUIRE_EQUALS(b(0, 2, 0), 3.0);
    REQUIRE_EQUALS(b(0, 2, 1), 4.0);
    REQUIRE_EQUALS(b(0, 2, 2), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 0, 2), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 2), 8.0);
    REQUIRE_EQUALS(b(1, 2, 0), 7.0);
    REQUIRE_EQUALS(b(1, 2, 1), 8.0);
    REQUIRE_EQUALS(b(1, 2, 2), 8.0);

    REQUIRE_EQUALS(b(2, 0, 0), 5.0);
    REQUIRE_EQUALS(b(2, 0, 1), 6.0);
    REQUIRE_EQUALS(b(2, 0, 2), 6.0);
    REQUIRE_EQUALS(b(2, 1, 0), 7.0);
    REQUIRE_EQUALS(b(2, 1, 1), 8.0);
    REQUIRE_EQUALS(b(2, 1, 2), 8.0);
    REQUIRE_EQUALS(b(2, 2, 0), 7.0);
    REQUIRE_EQUALS(b(2, 2, 1), 8.0);
    REQUIRE_EQUALS(b(2, 2, 2), 8.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 3, 3> b;

    b = (etl::avg_pool_3d(a, 1, 2, 2, 1, 1, 1));

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

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b;

    b = (etl::avg_pool_3d(a, 1, 3, 3, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1, 1> b;

    b = (etl::avg_pool_3d(a, 1, 4, 4, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2, 2> b;

    b = (etl::avg_pool_3d(a, 1, 1, 1, 1, 2, 2));

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 2, 2, 2> b;

    b = (etl::avg_pool_3d(a, 2, 2, 2,  2, 2, 2, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(1, 1, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("pooling/dyn/stride/avg3/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 3, 3, 3> b;

    b = (etl::avg_pool_3d(a, 2, 2, 2, 1, 1, 1, 1, 1, 1));

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 0, 2), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 1.0);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 2.5);
    REQUIRE_EQUALS(b(0, 1, 2), 0.5 * 1.5);
    REQUIRE_EQUALS(b(0, 2, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 2, 1), 0.5 * 1.75);
    REQUIRE_EQUALS(b(0, 2, 2), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.75);
    REQUIRE_EQUALS(b(1, 0, 1), 1.75);
    REQUIRE_EQUALS(b(1, 0, 2), 1.0);
    REQUIRE_EQUALS(b(1, 1, 0), 2.0);
    REQUIRE_EQUALS(b(1, 1, 1), 4.5);
    REQUIRE_EQUALS(b(1, 1, 2), 2.5);
    REQUIRE_EQUALS(b(1, 2, 0), 1.25);
    REQUIRE_EQUALS(b(1, 2, 1), 2.75);
    REQUIRE_EQUALS(b(1, 2, 2), 1.5);
}
