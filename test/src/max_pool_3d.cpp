//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "pool_3d_test.hpp"

#include <vector>

MP3_TEST_CASE("pooling/max3/1", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<2, 2, 2, 2, 2, 2, 0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 22.0);
    REQUIRE_EQUALS(b(0, 0, 1), 24.0);
    REQUIRE_EQUALS(b(0, 1, 0), 30.0);
    REQUIRE_EQUALS(b(0, 1, 1), 32.0);
}

MP3_TEST_CASE("pooling/max3/2", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 1, 1, 2> b;

    Impl::template apply<2, 4, 2, 2, 4, 2, 0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 30.0);
    REQUIRE_EQUALS(b(0, 0, 1), 32.0);
}

MP3_TEST_CASE("pooling/max3/3", "[pooling]") {
    etl::fast_matrix<T, 2, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0});
    etl::fast_matrix<T, 2, 1, 1> b;

    Impl::template apply<1, 4, 4, 1, 4, 4, 0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
    REQUIRE_EQUALS(b(1, 0, 0), 32.0);
}

MP3_TEST_CASE("pooling/max3/4", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::template apply<2, 2, 2,  2, 2, 2,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 8.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 8.0);
}

MP3_TEST_CASE("pooling/max3/5", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 1, 1> b;

    Impl::template apply<1, 2, 2,  1, 2, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 8.0);
}

MP3_TEST_CASE("pooling/max3/6", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 1, 1> b;

    Impl::template apply<1, 2, 2,  1, 2, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 8.0);
}

MP3_TEST_CASE("pooling/max3/7", "[pooling]") {
    etl::fast_matrix<T, 25, 25, 8, 8> a;
    etl::fast_matrix<T, 25, 25, 4, 4> b;

    Impl::template apply<1, 2, 2,  1, 2, 2,  0, 0, 0>(a, b);
}

MP3_TEST_CASE("pooling/max3/8", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 3, 3> b;

    Impl::template apply<1, 2, 2,  1, 1, 1,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 0, 2), 8.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
    REQUIRE_EQUALS(b(0, 1, 2), 12.0);

    REQUIRE_EQUALS(b(0, 2, 0), 14.0);
    REQUIRE_EQUALS(b(0, 2, 1), 15.0);
    REQUIRE_EQUALS(b(0, 2, 2), 16.0);
}

MP3_TEST_CASE("pooling/max3/9", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<1, 3, 3,  1, 1, 1,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 0, 1), 12.0);

    REQUIRE_EQUALS(b(0, 1, 0), 15.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);
}

MP3_TEST_CASE("pooling/max3/10", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1, 1> b;

    Impl::template apply<1, 4, 4,  1, 1, 4,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
}

MP3_TEST_CASE("pooling/max3/11", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::template apply<1, 1, 1,  1, 2, 2,  0, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

MP3_TEST_CASE("pooling/max3/12", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::template apply<2, 2, 2,  2, 2, 2,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
}

MP3_TEST_CASE("pooling/max3/13", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 3, 3, 3> b;

    Impl::template apply<2, 2, 2,  1, 1, 1,  1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 0, 2), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);
    REQUIRE_EQUALS(b(0, 1, 2), 4.0);
    REQUIRE_EQUALS(b(0, 2, 0), 3.0);
    REQUIRE_EQUALS(b(0, 2, 1), 4.0);
    REQUIRE_EQUALS(b(0, 2, 2), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 0, 2), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 2), 8.0);
    REQUIRE_EQUALS(b(1, 2, 0), 7.0);
    REQUIRE_EQUALS(b(1, 2, 1), 8.0);
    REQUIRE_EQUALS(b(1, 2, 2), 8.0);

    REQUIRE_EQUALS(b(2, 0, 0), 5.0);
    REQUIRE_EQUALS(b(2, 0, 1), 6.0);
    REQUIRE_EQUALS(b(2, 0, 2), 6.0);
    REQUIRE_EQUALS(b(2, 1, 0), 7.0);
    REQUIRE_EQUALS(b(2, 1, 1), 8.0);
    REQUIRE_EQUALS(b(2, 1, 2), 8.0);
    REQUIRE_EQUALS(b(2, 2, 0), 7.0);
    REQUIRE_EQUALS(b(2, 2, 1), 8.0);
    REQUIRE_EQUALS(b(2, 2, 2), 8.0);
}

// Dynamic versions

DYN_MP3_TEST_CASE("dyn_pooling/max3/1", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(1, 2, 2);

    Impl::apply(a, b, 2, 2, 2, 2, 2, 2, 0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 22.0);
    REQUIRE_EQUALS(b(0, 0, 1), 24.0);
    REQUIRE_EQUALS(b(0, 1, 0), 30.0);
    REQUIRE_EQUALS(b(0, 1, 1), 32.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/2", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(1, 1, 2);

    Impl::apply(a, b, 2, 4, 2, 2, 4, 2, 0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 30.0);
    REQUIRE_EQUALS(b(0, 0, 1), 32.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/3", "[pooling]") {
    etl::dyn_matrix<T, 3> a(2, 4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0));
    etl::dyn_matrix<T, 3> b(2, 1, 1);

    Impl::apply(a, b, 1, 4, 4, 1, 4, 4, 0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
    REQUIRE_EQUALS(b(1, 0, 0), 32.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/4", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2,  2, 2, 2,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 8.0);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(1, 1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 8.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/5", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 1, 1> b;

    Impl::apply(a, b, 1, 2, 2,  1, 2, 2,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 8.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/6", "[pooling]") {
    etl::fast_matrix<T, 25, 25, 8, 8> a;
    etl::fast_matrix<T, 25, 25, 4, 4> b;

    Impl::apply(a, b, 1, 2, 2,  1, 2, 2,  0, 0, 0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/7", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 3, 3> b;

    Impl::apply(a, b, 1, 2, 2,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 0, 2), 8.0);

    REQUIRE_EQUALS(b(0, 1, 0), 10.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
    REQUIRE_EQUALS(b(0, 1, 2), 12.0);

    REQUIRE_EQUALS(b(0, 2, 0), 14.0);
    REQUIRE_EQUALS(b(0, 2, 1), 15.0);
    REQUIRE_EQUALS(b(0, 2, 2), 16.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/8", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::apply(a, b, 1, 3, 3,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 0, 1), 12.0);

    REQUIRE_EQUALS(b(0, 1, 0), 15.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/9", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1, 1> b;

    Impl::apply(a, b, 1, 4, 4,  1, 1, 1,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 16.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/10", "[pooling]") {
    etl::fast_matrix<T, 1, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 2, 2> b;

    Impl::apply(a, b, 1, 1, 1,  1, 2, 2,  0, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 3.0);

    REQUIRE_EQUALS(b(0, 1, 0), 9.0);
    REQUIRE_EQUALS(b(0, 1, 1), 11.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/11", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2,  2, 2, 2,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
}

DYN_MP3_TEST_CASE("dyn_pooling/max3/12", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 3, 3, 3> b;

    Impl::apply(a, b, 2, 2, 2,  1, 1, 1,  1, 1, 1);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 0, 2), 2.0);
    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);
    REQUIRE_EQUALS(b(0, 1, 2), 4.0);
    REQUIRE_EQUALS(b(0, 2, 0), 3.0);
    REQUIRE_EQUALS(b(0, 2, 1), 4.0);
    REQUIRE_EQUALS(b(0, 2, 2), 4.0);

    REQUIRE_EQUALS(b(1, 0, 0), 5.0);
    REQUIRE_EQUALS(b(1, 0, 1), 6.0);
    REQUIRE_EQUALS(b(1, 0, 2), 6.0);
    REQUIRE_EQUALS(b(1, 1, 0), 7.0);
    REQUIRE_EQUALS(b(1, 1, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 2), 8.0);
    REQUIRE_EQUALS(b(1, 2, 0), 7.0);
    REQUIRE_EQUALS(b(1, 2, 1), 8.0);
    REQUIRE_EQUALS(b(1, 2, 2), 8.0);

    REQUIRE_EQUALS(b(2, 0, 0), 5.0);
    REQUIRE_EQUALS(b(2, 0, 1), 6.0);
    REQUIRE_EQUALS(b(2, 0, 2), 6.0);
    REQUIRE_EQUALS(b(2, 1, 0), 7.0);
    REQUIRE_EQUALS(b(2, 1, 1), 8.0);
    REQUIRE_EQUALS(b(2, 1, 2), 8.0);
    REQUIRE_EQUALS(b(2, 2, 0), 7.0);
    REQUIRE_EQUALS(b(2, 2, 1), 8.0);
    REQUIRE_EQUALS(b(2, 2, 2), 8.0);
}
