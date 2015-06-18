//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "cpp_utils/tmp.hpp"
#include "cpp_utils/assert.hpp"
#include "etl/tmp.hpp"

TEST_CASE( "tmp/sequence_equal/1", "[tmp]") {
    REQUIRE((etl::sequence_equal<std::index_sequence<2>, std::index_sequence<2>>::value));
    REQUIRE((etl::sequence_equal<std::index_sequence<>, std::index_sequence<>>::value));
    REQUIRE((etl::sequence_equal<std::index_sequence<1,2>, std::index_sequence<1,2>>::value));
    REQUIRE(!(etl::sequence_equal<std::index_sequence<1,2>, std::index_sequence<1,2,3>>::value));
    REQUIRE(!(etl::sequence_equal<std::index_sequence<1,2,3,4>, std::index_sequence<1,2,3>>::value));
}
