//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "bias_test.hpp"

// Tests for merge

TEMPLATE_TEST_CASE_2("merge/0", "[merge]", T, float, double) {
    etl::fast_matrix<T, 2, 2> a0({1, 2, 3, 2});
    etl::fast_matrix<T, 2, 2> a1({2, 3, 2, 6});
    etl::fast_matrix<T, 2, 2> a2({3, 2, 6, 0});
    etl::fast_matrix<T, 6, 2> c;

    merge(c, a0, 0);
    merge(c, a1, 1);
    merge(c, a2, 2);

    REQUIRE_EQUALS(c(0, 0), T(1));
    REQUIRE_EQUALS(c(0, 1), T(2));
    REQUIRE_EQUALS(c(1, 0), T(3));
    REQUIRE_EQUALS(c(1, 1), T(2));
    REQUIRE_EQUALS(c(2, 0), T(2));
    REQUIRE_EQUALS(c(2, 1), T(3));
    REQUIRE_EQUALS(c(3, 0), T(2));
    REQUIRE_EQUALS(c(3, 1), T(6));
    REQUIRE_EQUALS(c(4, 0), T(3));
    REQUIRE_EQUALS(c(4, 1), T(2));
    REQUIRE_EQUALS(c(5, 0), T(6));
    REQUIRE_EQUALS(c(5, 1), T(0));
}

TEMPLATE_TEST_CASE_2("batch_merge/0", "[batch_merge]", T, float, double) {
    etl::fast_matrix<T, 2, 2, 2> a0({1, 2, 3, 2, 0, 1, 4, 5});
    etl::fast_matrix<T, 2, 2, 2> a1({2, 3, 2, 6, -1, -2, -3, -4});
    etl::fast_matrix<T, 2, 2, 2> a2({3, 2, 6, 0, 1, 2, 3, 4});
    etl::fast_matrix<T, 2, 6, 2> c;

    batch_merge(c, a0, 0);
    batch_merge(c, a1, 1);
    batch_merge(c, a2, 2);

    REQUIRE_EQUALS(c(0, 0, 0), T(1));
    REQUIRE_EQUALS(c(0, 0, 1), T(2));
    REQUIRE_EQUALS(c(0, 1, 0), T(3));
    REQUIRE_EQUALS(c(0, 1, 1), T(2));
    REQUIRE_EQUALS(c(0, 2, 0), T(2));
    REQUIRE_EQUALS(c(0, 2, 1), T(3));
    REQUIRE_EQUALS(c(0, 3, 0), T(2));
    REQUIRE_EQUALS(c(0, 3, 1), T(6));
    REQUIRE_EQUALS(c(0, 4, 0), T(3));
    REQUIRE_EQUALS(c(0, 4, 1), T(2));
    REQUIRE_EQUALS(c(0, 5, 0), T(6));
    REQUIRE_EQUALS(c(0, 5, 1), T(0));

    REQUIRE_EQUALS(c(1, 0, 0), T(0));
    REQUIRE_EQUALS(c(1, 0, 1), T(1));
    REQUIRE_EQUALS(c(1, 1, 0), T(4));
    REQUIRE_EQUALS(c(1, 1, 1), T(5));
    REQUIRE_EQUALS(c(1, 2, 0), T(-1));
    REQUIRE_EQUALS(c(1, 2, 1), T(-2));
    REQUIRE_EQUALS(c(1, 3, 0), T(-3));
    REQUIRE_EQUALS(c(1, 3, 1), T(-4));
    REQUIRE_EQUALS(c(1, 4, 0), T(1));
    REQUIRE_EQUALS(c(1, 4, 1), T(2));
    REQUIRE_EQUALS(c(1, 5, 0), T(3));
    REQUIRE_EQUALS(c(1, 5, 1), T(4));
}

// Tests for dispatch

TEMPLATE_TEST_CASE_2("dispatch/0", "[dispatch]", T, float, double) {
    etl::fast_matrix<T, 2, 2> sa0({1, 2, 3, 2});
    etl::fast_matrix<T, 2, 2> sa1({2, 3, 2, 6});
    etl::fast_matrix<T, 2, 2> sa2({3, 2, 6, 0});
    etl::fast_matrix<T, 2, 2> a0;
    etl::fast_matrix<T, 2, 2> a1;
    etl::fast_matrix<T, 2, 2> a2;

    etl::fast_matrix<T, 6, 2> c;

    merge(c, sa0, 0);
    merge(c, sa1, 1);
    merge(c, sa2, 2);

    dispatch(a0, c, 0);
    dispatch(a1, c, 1);
    dispatch(a2, c, 2);

    for(size_t i = 0; i < 4; ++i){
        REQUIRE_EQUALS(a0[i], sa0[i]);
        REQUIRE_EQUALS(a1[i], sa1[i]);
        REQUIRE_EQUALS(a2[i], sa2[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_dispatch/0", "[batch_dispatch]", T, float, double) {
    etl::fast_matrix<T, 2, 2, 2> sa0({1, 2, 3, 2, 0, 1, 4, 5});
    etl::fast_matrix<T, 2, 2, 2> sa1({2, 3, 2, 6, -1, -2, -3, -4});
    etl::fast_matrix<T, 2, 2, 2> sa2({3, 2, 6, 0, 1, 2, 3, 4});
    etl::fast_matrix<T, 2, 2, 2> a0;
    etl::fast_matrix<T, 2, 2, 2> a1;
    etl::fast_matrix<T, 2, 2, 2> a2;

    etl::fast_matrix<T, 2, 6, 2> c;

    batch_merge(c, sa0, 0);
    batch_merge(c, sa1, 1);
    batch_merge(c, sa2, 2);

    batch_dispatch(a0, c, 0);
    batch_dispatch(a1, c, 1);
    batch_dispatch(a2, c, 2);

    for(size_t i = 0; i < 8; ++i){
        REQUIRE_EQUALS(a0[i], sa0[i]);
        REQUIRE_EQUALS(a1[i], sa1[i]);
        REQUIRE_EQUALS(a2[i], sa2[i]);
    }
}
