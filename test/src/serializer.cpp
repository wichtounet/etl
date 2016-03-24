//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <fstream>

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("serializer/3", "[serializer]", Z, float, double) {
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

    REQUIRE(a[0] == 1.0);
    REQUIRE(a[1] == -2.0);
    REQUIRE(a[2] == 3.0);
}

TEMPLATE_TEST_CASE_2("serializer/4", "[serializer]", Z, float, double) {
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

    REQUIRE(a[0] == 5.0);
    REQUIRE(a[1] == 3.0);
    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 3.0);
}
