//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <fstream>

TEMPLATE_TEST_CASE_2("serializer/1", "[serializer]", Z, float, double) {
    {
        etl::serializer<std::ofstream> serializer("test1.tmp.etl", std::ios::binary);

        etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
        serializer << a;
    }

    etl::fast_vector<Z, 3> a;

    {
        etl::deserializer<std::ifstream> deserializer("test1.tmp.etl", std::ios::binary);
        deserializer >> a;
    }

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], -2.0);
    REQUIRE_EQUALS(a[2], 3.0);
}

TEMPLATE_TEST_CASE_2("serializer/2", "[serializer]", Z, float, double) {
    {
        etl::serializer<std::ofstream> serializer("test2.tmp.etl", std::ios::binary);

        etl::fast_vector<Z, 2> a({5.0, 3.0});
        etl::fast_vector<Z, 3> b({1.0, -2.0, 3.0});
        serializer << a << b;
    }

    etl::fast_vector<Z, 2> a;
    etl::fast_vector<Z, 3> b;

    {
        etl::deserializer<std::ifstream> deserializer("test2.tmp.etl", std::ios::binary);

        deserializer >> a >> b;
    }

    REQUIRE_EQUALS(a[0], 5.0);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 3.0);
}

TEMPLATE_TEST_CASE_2("serializer/3", "[serializer]", Z, float, double) {
    {
        etl::serializer<std::ofstream> serializer("test3.tmp.etl", std::ios::binary);

        etl::dyn_matrix<Z> a(3, 2, etl::values<Z>(1.0, 3.0, -4.0, -1.0, 0.0, 2.5));
        serializer << a;
    }

    etl::dyn_matrix<Z> a(3, 2);

    {
        etl::deserializer<std::ifstream> deserializer("test3.tmp.etl", std::ios::binary);
        deserializer >> a;
    }

    REQUIRE_EQUALS(etl::dim(a, 0), 3UL);
    REQUIRE_EQUALS(etl::dim(a, 1), 2UL);

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], -4.0);
    REQUIRE_EQUALS(a[3], -1.0);
    REQUIRE_EQUALS(a[4], 0.0);
    REQUIRE_EQUALS(a[5], 2.5);
}

TEMPLATE_TEST_CASE_2("serializer/4", "[serializer]", Z, float, double) {
    {
        etl::serializer<std::ofstream> serializer("test3.tmp.etl", std::ios::binary);

        etl::dyn_matrix<Z, 3> a(2, 3, 1, etl::values<Z>(1.0, 3.0, -4.0, -1.0, 0.0, 2.5));
        serializer << a;
    }

    etl::dyn_matrix<Z, 3> a;

    {
        etl::deserializer<std::ifstream> deserializer("test3.tmp.etl", std::ios::binary);
        deserializer >> a;
    }

    REQUIRE_EQUALS(etl::dim(a, 0), 2UL);
    REQUIRE_EQUALS(etl::dim(a, 1), 3UL);
    REQUIRE_EQUALS(etl::dim(a, 2), 1UL);

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 3.0);
    REQUIRE_EQUALS(a[2], -4.0);
    REQUIRE_EQUALS(a[3], -1.0);
    REQUIRE_EQUALS(a[4], 0.0);
    REQUIRE_EQUALS(a[5], 2.5);
}
