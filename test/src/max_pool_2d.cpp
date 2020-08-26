//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

MP2_TEST_CASE("pooling/max2/8", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 3, 3> b;

    Impl::template apply<2, 2, 1, 1, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 2), 8.0);

    REQUIRE_EQUALS(b(1, 0), 10.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
    REQUIRE_EQUALS(b(1, 2), 12.0);

    REQUIRE_EQUALS(b(2, 0), 14.0);
    REQUIRE_EQUALS(b(2, 1), 15.0);
    REQUIRE_EQUALS(b(2, 2), 16.0);
}

MP2_TEST_CASE("pooling/max2/9", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 2> b;

    Impl::template apply<3, 3, 1, 1, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 1), 12.0);

    REQUIRE_EQUALS(b(1, 0), 15.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

MP2_TEST_CASE("pooling/max2/10", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 1, 1> b;

    Impl::template apply<4, 4, 1, 1, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

MP2_TEST_CASE("pooling/max2/11", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 2> b;

    Impl::template apply<1, 1, 2, 2, 0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

MP2_TEST_CASE("pooling/max2/12", "[pooling]") {
    etl::fast_matrix<T, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<T, 2, 2> b;

    Impl::template apply<2, 2, 2, 2, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
}

MP2_TEST_CASE("pooling/max2/13", "[pooling]") {
    etl::fast_matrix<T, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<T, 3, 3> b;

    Impl::template apply<2, 2, 1, 1, 1, 1>(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 2), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
    REQUIRE_EQUALS(b(1, 2), 4.0);

    REQUIRE_EQUALS(b(2, 0), 3.0);
    REQUIRE_EQUALS(b(2, 1), 4.0);
    REQUIRE_EQUALS(b(2, 2), 4.0);
}

MP2_TEST_CASE("pooling/max3/14", "[pooling]") {
    etl::fast_matrix<T, 2, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<T, 2, 2, 1, 1> b;

    Impl::template apply<2, 2,  2, 2,  0, 0>(a, b);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0, 0, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 8.0);
}

// Dynamic version

DYN_MP2_TEST_CASE("dyn_pooling/max2/1", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 2);

    Impl::apply(a, b, 2, 2, 2, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/2", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(1, 1);

    Impl::apply(a, b, 4, 4, 4, 4, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/3", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(1, 2);

    Impl::apply(a, b, 4, 2, 4, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/4", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 1);

    Impl::apply(a, b, 2, 4, 2, 4, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 8.0);
    REQUIRE_EQUALS(b(1, 0), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/5", "[pooling]") {
    etl::fast_matrix<T, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 4, 4> a;
    a(0) = aa;
    a(1) = aa;

    etl::fast_matrix<T, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 0, 1), 8.0);
    REQUIRE_EQUALS(b(0, 1, 0), 14.0);
    REQUIRE_EQUALS(b(0, 1, 1), 16.0);

    REQUIRE_EQUALS(b(1, 0, 0), 6.0);
    REQUIRE_EQUALS(b(1, 0, 1), 8.0);
    REQUIRE_EQUALS(b(1, 1, 0), 14.0);
    REQUIRE_EQUALS(b(1, 1, 1), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/6", "[pooling]") {
    etl::fast_matrix<T, 4, 4> aa({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 2, 2, 4, 4> a;
    a(0)(0) = aa;
    a(0)(1) = aa;
    a(1)(0) = aa;
    a(1)(1) = aa;

    etl::fast_matrix<T, 2, 2, 2, 2> b;

    Impl::apply(a, b, 2, 2, 2, 2, 0, 0);

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

DYN_MP2_TEST_CASE("dyn_pooling/max2/7", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(3, 3);

    Impl::apply(a, b, 2, 2, 1, 1, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 6.0);
    REQUIRE_EQUALS(b(0, 1), 7.0);
    REQUIRE_EQUALS(b(0, 2), 8.0);

    REQUIRE_EQUALS(b(1, 0), 10.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
    REQUIRE_EQUALS(b(1, 2), 12.0);

    REQUIRE_EQUALS(b(2, 0), 14.0);
    REQUIRE_EQUALS(b(2, 1), 15.0);
    REQUIRE_EQUALS(b(2, 2), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/8", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 2);

    Impl::apply(a, b, 3, 3, 1, 1, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 11.0);
    REQUIRE_EQUALS(b(0, 1), 12.0);

    REQUIRE_EQUALS(b(1, 0), 15.0);
    REQUIRE_EQUALS(b(1, 1), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/9", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(1, 1);

    Impl::apply(a, b, 4, 4, 1, 1, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 16.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/10", "[pooling]") {
    etl::dyn_matrix<T, 2> a(4, 4, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0));
    etl::dyn_matrix<T, 2> b(2, 2);

    Impl::apply(a, b, 1, 1, 2, 2, 0, 0);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);

    REQUIRE_EQUALS(b(1, 0), 9.0);
    REQUIRE_EQUALS(b(1, 1), 11.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/11", "[pooling]") {
    etl::dyn_matrix<T, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<T, 2> b(2, 2);

    Impl::apply(a, b, 2, 2, 2, 2, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/12", "[pooling]") {
    etl::dyn_matrix<T, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<T, 2> b(3, 3);

    Impl::apply(a, b, 2, 2, 1, 1, 1, 1);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 2), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
    REQUIRE_EQUALS(b(1, 2), 4.0);

    REQUIRE_EQUALS(b(2, 0), 3.0);
    REQUIRE_EQUALS(b(2, 1), 4.0);
    REQUIRE_EQUALS(b(2, 2), 4.0);
}

DYN_MP2_TEST_CASE("dyn_pooling/max2/13", "[pooling]") {
    etl::fast_matrix<T, 4, 4> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0});
    etl::fast_matrix<T, 4, 4> b;

    Impl::apply(a, b, 2, 2, 2, 2, 2, 2);

    // CUDNN does not return 0 for values pooled for a fully-padded block
    // Instead, it uses -FLOAT_MAX as the default
    T pad_value = T(0);
    if (b(0, 0) == -std::numeric_limits<T>::max()) {
        pad_value = -std::numeric_limits<T>::max();
    }

    REQUIRE_EQUALS(b(0, 0), pad_value);
    REQUIRE_EQUALS(b(0, 1), pad_value);
    REQUIRE_EQUALS(b(0, 2), pad_value);
    REQUIRE_EQUALS(b(0, 3), pad_value);

    REQUIRE_EQUALS(b(1, 0), pad_value);
    REQUIRE_EQUALS(b(1, 1), 6.0);
    REQUIRE_EQUALS(b(1, 2), 8.0);
    REQUIRE_EQUALS(b(1, 3), pad_value);

    REQUIRE_EQUALS(b(2, 0), pad_value);
    REQUIRE_EQUALS(b(2, 1), 14.0);
    REQUIRE_EQUALS(b(2, 2), 16.0);
    REQUIRE_EQUALS(b(2, 3), pad_value);

    REQUIRE_EQUALS(b(3, 0), pad_value);
    REQUIRE_EQUALS(b(3, 1), pad_value);
    REQUIRE_EQUALS(b(3, 2), pad_value);
    REQUIRE_EQUALS(b(3, 3), pad_value);
}
