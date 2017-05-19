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
    etl::fast_matrix<Z, 2, 2> b;
    b = etl::max_pool_2d<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 1> b;
    b = etl::max_pool_2d<4, 4>(a);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 1, 2> b;
    b = etl::max_pool_2d<4, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("pooling/max2/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 1> b;
    b = etl::max_pool_2d<2, 4>(a);

    REQUIRE_EQUALS(b(0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0), 16.0);
}
