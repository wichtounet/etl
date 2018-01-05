//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "pool_3d_test.hpp"

#include <vector>

AVGP3_TEST_CASE("pooling/avg3/1", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<2, 2, 2,  2, 2, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 11.5);
    REQUIRE_EQUALS(b(0, 0, 1), 13.5);
    REQUIRE_EQUALS(b(0, 1, 0), 19.5);
    REQUIRE_EQUALS(b(0, 1, 1), 21.5);
}

AVGP3_TEST_CASE("pooling/avg3/2", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 1, 1, 2> b;

    Impl::template apply<2, 4, 2,  2, 4, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 15.5);
    REQUIRE_EQUALS(b(0, 0, 1), 17.5);
}

AVGP3_TEST_CASE("pooling/avg3/3", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 2, 1, 1> b;

    Impl::template apply<1, 4, 4,  1, 4, 4,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
    REQUIRE_EQUALS(b(1, 0, 0), 24.5);
}

AVGP3_TEST_CASE("pooling/avg3/4", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::template apply<2, 2, 2,  2, 2, 2,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 1.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 1.0);
}

AVGP3_TEST_CASE("pooling/avg3/5", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 3, 3> b;

    Impl::template apply<1, 2, 2,  1, 1, 1,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 0, 1), 4.5);
    REQUIRE_EQUALS(b(0, 0, 2), 5.5);

    REQUIRE_EQUALS(b(0, 1, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1, 1), 8.5);
    REQUIRE_EQUALS(b(0, 1, 2), 9.5);

    REQUIRE_EQUALS(b(0, 2, 0), 11.5);
    REQUIRE_EQUALS(b(0, 2, 1), 12.5);
    REQUIRE_EQUALS(b(0, 2, 2), 13.5);
}

AVGP3_TEST_CASE("pooling/avg3/6", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<1, 3, 3,  1, 1, 1,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

AVGP3_TEST_CASE("pooling/avg3/7", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1, 1> b;

    Impl::template apply<1, 4, 4,  1, 1, 1,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
}

AVGP3_TEST_CASE("pooling/avg3/8", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<1, 1, 1,  1, 2, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

AVGP3_TEST_CASE("pooling/avg3/9", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::template apply<2, 2, 2,  2, 2, 2,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(1, 1, 1), 1.0);
}

AVGP3_TEST_CASE("pooling/avg3/10", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 3, 3, 3> b;

    Impl::template apply<2, 2, 2,  1, 1, 1,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 0, 2), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 1.0);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 2.5);
    REQUIRE_EQUALS(b(0, 1, 2), 0.5 * 1.5);
    REQUIRE_EQUALS(b(0, 2, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 2, 1), 0.5 * 1.75);
    REQUIRE_EQUALS(b(0, 2, 2), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.75);
    REQUIRE_EQUALS(b(1, 0, 1), 1.75);
    REQUIRE_EQUALS(b(1, 0, 2), 1.0);
    REQUIRE_EQUALS(b(1, 1, 0), 2.0);
    REQUIRE_EQUALS(b(1, 1, 1), 4.5);
    REQUIRE_EQUALS(b(1, 1, 2), 2.5);
    REQUIRE_EQUALS(b(1, 2, 0), 1.25);
    REQUIRE_EQUALS(b(1, 2, 1), 2.75);
    REQUIRE_EQUALS(b(1, 2, 2), 1.5);
}

// Dynamic versions

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/1", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(1, 2, 2);

    Impl::apply(a, b, 2, 2, 2,  2, 2, 2,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 11.5);
    REQUIRE_EQUALS(b(0, 0, 1), 13.5);
    REQUIRE_EQUALS(b(0, 1, 0), 19.5);
    REQUIRE_EQUALS(b(0, 1, 1), 21.5);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/2", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(1, 1, 2);

    Impl::apply(a, b, 2, 4, 2,  2, 4, 2,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 15.5);
    REQUIRE_EQUALS(b(0, 0, 1), 17.5);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/3", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(2, 1, 1);

    Impl::apply(a, b, 1, 4, 4,  1, 4, 4,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
    REQUIRE_EQUALS(b(1, 0, 0), 24.5);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/4", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2,  2, 2, 2,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 1.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 1.0);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/5", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 3, 3> b;

    Impl::apply(a, b, 1, 2, 2,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 3.5);
    REQUIRE_EQUALS(b(0, 0, 1), 4.5);
    REQUIRE_EQUALS(b(0, 0, 2), 5.5);

    REQUIRE_EQUALS(b(0, 1, 0), 7.5);
    REQUIRE_EQUALS(b(0, 1, 1), 8.5);
    REQUIRE_EQUALS(b(0, 1, 2), 9.5);

    REQUIRE_EQUALS(b(0, 2, 0), 11.5);
    REQUIRE_EQUALS(b(0, 2, 1), 12.5);
    REQUIRE_EQUALS(b(0, 2, 2), 13.5);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/6", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::apply(a, b, 1, 3, 3,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/7", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1, 1> b;

    Impl::apply(a, b, 1, 4, 4,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 8.5);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/8", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::apply(a, b, 1, 1, 1,  1, 2, 2,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/9", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2,  2, 2, 2,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.625);
    REQUIRE_EQUALS(b(1, 0, 1), 0.75);
    REQUIRE_EQUALS(b(1, 1, 0), 0.875);
    REQUIRE_EQUALS(b(1, 1, 1), 1.0);
}

DYN_AVGP3_TEST_CASE("dyn_pooling/avg3/10", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 3, 3, 3> b;

    Impl::apply(a, b, 2, 2, 2,  1, 1, 1,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0), 0.5 * 0.25);
    REQUIRE_EQUALS(b(0, 0, 1), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 0, 2), 0.5 * 0.5);
    REQUIRE_EQUALS(b(0, 1, 0), 0.5 * 1.0);
    REQUIRE_EQUALS(b(0, 1, 1), 0.5 * 2.5);
    REQUIRE_EQUALS(b(0, 1, 2), 0.5 * 1.5);
    REQUIRE_EQUALS(b(0, 2, 0), 0.5 * 0.75);
    REQUIRE_EQUALS(b(0, 2, 1), 0.5 * 1.75);
    REQUIRE_EQUALS(b(0, 2, 2), 0.5 * 1.0);

    REQUIRE_EQUALS(b(1, 0, 0), 0.75);
    REQUIRE_EQUALS(b(1, 0, 1), 1.75);
    REQUIRE_EQUALS(b(1, 0, 2), 1.0);
    REQUIRE_EQUALS(b(1, 1, 0), 2.0);
    REQUIRE_EQUALS(b(1, 1, 1), 4.5);
    REQUIRE_EQUALS(b(1, 1, 2), 2.5);
    REQUIRE_EQUALS(b(1, 2, 0), 1.25);
    REQUIRE_EQUALS(b(1, 2, 1), 2.75);
    REQUIRE_EQUALS(b(1, 2, 2), 1.5);
}
