//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("iterator/1", "[iterator]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> M(5.5);

    REQUIRE((std::is_same_v<Z*, decltype(M.begin())>));
    REQUIRE((std::is_same_v<Z*, decltype(M.end())>));

    REQUIRE((std::is_same_v<const Z*, decltype(M.cbegin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(M.cend())>));
}

TEMPLATE_TEST_CASE_2("iterator/2", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(2,3,4);

    REQUIRE((std::is_same_v<Z*, decltype(M.begin())>));
    REQUIRE((std::is_same_v<Z*, decltype(M.end())>));

    REQUIRE((std::is_same_v<const Z*, decltype(M.cbegin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(M.cend())>));
}

TEMPLATE_TEST_CASE_2("iterator/3", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(2,3,4);

    auto A = M(0);

    REQUIRE((std::is_same_v<Z*, decltype(A.begin())>));
    REQUIRE((std::is_same_v<Z*, decltype(A.end())>));

    REQUIRE((std::is_same_v<const Z*, decltype(A.cbegin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(A.cend())>));
}

TEMPLATE_TEST_CASE_2("iterator/4", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 2> M(3,3);

    auto A = M * M;

    REQUIRE((std::is_same_v<const Z*, decltype(A.begin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(A.end())>));

    REQUIRE((std::is_same_v<const Z*, decltype(A.cbegin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(A.cend())>));
}

TEMPLATE_TEST_CASE_2("iterator/5", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(3, 3,3);

    auto A = M(0) * M(1);

    REQUIRE((std::is_same_v<const Z*, decltype(A.begin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(A.end())>));

    REQUIRE((std::is_same_v<const Z*, decltype(A.cbegin())>));
    REQUIRE((std::is_same_v<const Z*, decltype(A.cend())>));
}
