//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("memory_slice/1", "[slice]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto s1 = etl::memory_slice<etl::aligned>(a, 0, 3);

    REQUIRE_EQUALS(etl::size(s1), 3UL);

    REQUIRE_EQUALS(etl::dim<0>(s1), 3UL);
    REQUIRE_EQUALS(etl::dimensions(s1), 1UL);

    REQUIRE_EQUALS(s1[0], 1);
    REQUIRE_EQUALS(s1[1], 2);
    REQUIRE_EQUALS(s1[2], 3);

    auto s2 = etl::memory_slice<etl::unaligned>(a, 1, 3);

    REQUIRE_EQUALS(etl::size(s2), 2UL);

    REQUIRE_EQUALS(etl::dim<0>(s2), 2UL);
    REQUIRE_EQUALS(etl::dimensions(s2), 1UL);

    REQUIRE_EQUALS(s2[0], 2);
    REQUIRE_EQUALS(s2[1], 3);
    REQUIRE_EQUALS(s2[2], 4);

    auto s3 = etl::memory_slice<etl::aligned>(a, 0, 6);

    REQUIRE_EQUALS(etl::size(s3), 6UL);

    REQUIRE_EQUALS(etl::dim<0>(s3), 6UL);
    REQUIRE_EQUALS(etl::dimensions(s3), 1UL);

    REQUIRE_EQUALS(s3[0], 1);
    REQUIRE_EQUALS(s3[1], 2);
    REQUIRE_EQUALS(s3[2], 3);
    REQUIRE_EQUALS(s3[3], 4);
    REQUIRE_EQUALS(s3[4], 5);
    REQUIRE_EQUALS(s3[5], 6);
}
