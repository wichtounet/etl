//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "pool_test.hpp"

#include <vector>

MP2_TEST_CASE("pooling/max2/1", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 2> b;

    Impl::template apply<2, 2, 2, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

MP2_TEST_CASE("pooling/max2/2", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1> b;

    Impl::template apply<4, 4, 4, 4, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

MP2_TEST_CASE("pooling/max2/3", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2> b;

    Impl::template apply<4, 2, 4, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1), 16.0);
}

MP2_TEST_CASE("pooling/max2/4", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 1> b;

    Impl::template apply<2, 4, 2, 4, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0), 16.0);
}

MP2_TEST_CASE("pooling/max2/5", "[pooling]") {
    etl::fast_matrix<T, 4, 4> A({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 4, 4> a;
    etl::fast_matrix<T, 2, 2, 2> b;

    a(0) = A;
    a(1) = A;

    Impl::template apply<2, 2, 2, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1), 16.0);
}

MP2_TEST_CASE("pooling/max2/6", "[pooling]") {
    etl::fast_matrix<T, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 4, 4> a;
    a(0) = aa;
    a(1) = aa;

    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::template apply<2, 2, 2, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1), 16.0);
}

MP2_TEST_CASE("pooling/max2/7", "[pooling]") {
    etl::fast_matrix<T, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 2, 4, 4> a;
    a(0)(0) = aa;
    a(0)(1) = aa;
    a(1)(0) = aa;
    a(1)(1) = aa;

    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::template apply<2, 2, 2, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 16.0);
}

// Dynamic version

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/1", "[pooling]", T, float, double) {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 2);

    b = etl::max_pool_2d(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/2", "[pooling]", T, float, double) {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(1, 1);

    b = etl::max_pool_2d(a, 4, 4);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/3", "[pooling]", T, float, double) {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(1, 2);

    b = etl::max_pool_2d(a, 4, 2);

    REQUIRE_EQUALS(b(0, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/4", "[pooling]", T, float, double) {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 1);

    b = etl::max_pool_2d(a, 2, 4);

    REQUIRE_EQUALS(b(0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0), 16.0);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 4, 4> a;
    a(0) = aa;
    a(1) = aa;

    etl::fast_matrix<Z, 2, 2, 2> b;
    b = etl::max_pool_2d(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1), 16.0);
}

TEMPLATE_TEST_CASE_2("dyn_pooling/max2/6", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<Z, 2, 2, 4, 4> a;
    a(0)(0) = aa;
    a(0)(1) = aa;
    a(1)(0) = aa;
    a(1)(1) = aa;

    etl::fast_matrix<Z, 2, 2, 2, 2> b;

    b = etl::max_pool_2d(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 16.0);
}
