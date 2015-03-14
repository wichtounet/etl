//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <cmath>

#include "catch.hpp"

#include "etl/etl.hpp"

TEST_CASE( "direct_access/fast_matrix", "direct_access" ) {
    etl::fast_matrix<double, 5, 5> test_matrix{etl::magic<5>()};

    REQUIRE(test_matrix.size() == 25);

    auto it = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}

TEST_CASE( "direct_access/dyn_matrix", "direct_access" ) {
    etl::dyn_matrix<double, 2> test_matrix{etl::magic(5)};

    REQUIRE(test_matrix.size() == 25);

    auto it = test_matrix.memory_start();
    auto end = test_matrix.memory_end();

    REQUIRE(std::is_pointer<decltype(it)>::value);
    REQUIRE(std::is_pointer<decltype(end)>::value);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        REQUIRE(test_matrix[i] == *it);
        REQUIRE(it != end);
        ++it;
    }

    REQUIRE(it == end);
}
