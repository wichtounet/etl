//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("iterator/1", "[iterator]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> M(5.5);

    REQUIRE((std::is_same<Z*, decltype(M.begin())>::value));
    REQUIRE((std::is_same<Z*, decltype(M.end())>::value));

    REQUIRE((std::is_same<const Z*, decltype(M.cbegin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(M.cend())>::value));
}

TEMPLATE_TEST_CASE_2("iterator/2", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(2,3,4);

    REQUIRE((std::is_same<Z*, decltype(M.begin())>::value));
    REQUIRE((std::is_same<Z*, decltype(M.end())>::value));

    REQUIRE((std::is_same<const Z*, decltype(M.cbegin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(M.cend())>::value));
}

TEMPLATE_TEST_CASE_2("iterator/3", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(2,3,4);

    auto A = M(0);

    REQUIRE((std::is_same<Z*, decltype(A.begin())>::value));
    REQUIRE((std::is_same<Z*, decltype(A.end())>::value));

    REQUIRE((std::is_same<const Z*, decltype(A.cbegin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(A.cend())>::value));
}

TEMPLATE_TEST_CASE_2("iterator/4", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 2> M(3,3);

    auto A = M * M;

    REQUIRE((std::is_same<const Z*, decltype(A.begin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(A.end())>::value));

    REQUIRE((std::is_same<const Z*, decltype(A.cbegin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(A.cend())>::value));
}

TEMPLATE_TEST_CASE_2("iterator/5", "[iterator]", Z, float, double) {
    etl::dyn_matrix<Z, 3> M(3, 3,3);

    auto A = M(0) * M(1);

    REQUIRE((std::is_same<const Z*, decltype(A.begin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(A.end())>::value));

    REQUIRE((std::is_same<const Z*, decltype(A.cbegin())>::value));
    REQUIRE((std::is_same<const Z*, decltype(A.cend())>::value));
}
