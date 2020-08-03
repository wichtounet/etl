//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("upsample/2d/0", "[upsample]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 1, 2, 2> b({1.0, 2.0, 3.0, 4.0});

    auto expr2 = etl::upsample_2d<2, 2>(a);
    using expr2_type = decltype(expr2);

    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::template dim<0>(), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::template dim<1>(), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::size(), 16UL);

    auto expr3 = etl::upsample_2d<2, 2>(b);
    using expr3_type = decltype(expr3);

    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::dimensions(), 3UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::template dim<0>(), 1UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::template dim<1>(), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::template dim<2>(), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::size(), 16UL);
}

TEMPLATE_TEST_CASE_2("upsample/2d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 4, 4> c;

    c = etl::upsample_2d<2, 2>(a);

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 1.0);
    REQUIRE_EQUALS(c(1, 0), 1.0);
    REQUIRE_EQUALS(c(1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 3), 2.0);
    REQUIRE_EQUALS(c(1, 2), 2.0);
    REQUIRE_EQUALS(c(1, 3), 2.0);

    REQUIRE_EQUALS(c(2, 0), 3.0);
    REQUIRE_EQUALS(c(2, 1), 3.0);
    REQUIRE_EQUALS(c(3, 0), 3.0);
    REQUIRE_EQUALS(c(3, 1), 3.0);

    REQUIRE_EQUALS(c(2, 2), 4.0);
    REQUIRE_EQUALS(c(2, 3), 4.0);
    REQUIRE_EQUALS(c(3, 2), 4.0);
    REQUIRE_EQUALS(c(3, 3), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/2d/2", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 3, 3> c;

    c = etl::upsample_2d<2, 2, 1, 1>(a);

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 3.0);
    REQUIRE_EQUALS(c(0, 2), 2.0);

    REQUIRE_EQUALS(c(1, 0), 4.0);
    REQUIRE_EQUALS(c(1, 1), 10.0);
    REQUIRE_EQUALS(c(1, 2), 6.0);

    REQUIRE_EQUALS(c(2, 0), 3.0);
    REQUIRE_EQUALS(c(2, 1), 7.0);
    REQUIRE_EQUALS(c(2, 2), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/2d/3", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 2, 2> c;

    c = etl::upsample_2d<2, 2, 2, 2, 1, 1>(a);

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 2.0);

    REQUIRE_EQUALS(c(1, 0), 3.0);
    REQUIRE_EQUALS(c(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/2d/4", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 5.0, 7.0, 8.0});
    etl::fast_matrix<Z, 4, 4> c;

    c = etl::upsample_2d<2, 2, 2, 2, 1, 1>(a);

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 3.0);
    REQUIRE_EQUALS(c(0, 2), 3.0);
    REQUIRE_EQUALS(c(0, 3), 4.0);

    REQUIRE_EQUALS(c(1, 0), 5.0);
    REQUIRE_EQUALS(c(1, 1), 7.0);
    REQUIRE_EQUALS(c(1, 2), 7.0);
    REQUIRE_EQUALS(c(1, 3), 8.0);

    REQUIRE_EQUALS(c(2, 0), 5.0);
    REQUIRE_EQUALS(c(2, 1), 7.0);
    REQUIRE_EQUALS(c(2, 2), 7.0);
    REQUIRE_EQUALS(c(2, 3), 8.0);

    REQUIRE_EQUALS(c(3, 0), 5.0);
    REQUIRE_EQUALS(c(3, 1), 7.0);
    REQUIRE_EQUALS(c(3, 2), 7.0);
    REQUIRE_EQUALS(c(3, 3), 8.0);
}

TEMPLATE_TEST_CASE_2("upsample/2d/5", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a({0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<Z, 2, 2> c;

    c = etl::upsample_2d<2, 2, 2, 2, 2, 2>(a);

    REQUIRE_EQUALS(c(0, 0), 4.0);
    REQUIRE_EQUALS(c(0, 1), 4.0);

    REQUIRE_EQUALS(c(1, 0), 4.0);
    REQUIRE_EQUALS(c(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/3d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 1, 2, 2> a({1.0, 2.0, 3.0, 4.0});
    etl::fast_matrix<Z, 1, 4, 4> c;

    c = etl::upsample_3d<1, 2, 2>(a);

    REQUIRE_EQUALS(c(0, 0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 0, 1), 1.0);
    REQUIRE_EQUALS(c(0, 1, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 0, 3), 2.0);
    REQUIRE_EQUALS(c(0, 1, 2), 2.0);
    REQUIRE_EQUALS(c(0, 1, 3), 2.0);

    REQUIRE_EQUALS(c(0, 2, 0), 3.0);
    REQUIRE_EQUALS(c(0, 2, 1), 3.0);
    REQUIRE_EQUALS(c(0, 3, 0), 3.0);
    REQUIRE_EQUALS(c(0, 3, 1), 3.0);

    REQUIRE_EQUALS(c(0, 2, 2), 4.0);
    REQUIRE_EQUALS(c(0, 2, 3), 4.0);
    REQUIRE_EQUALS(c(0, 3, 2), 4.0);
    REQUIRE_EQUALS(c(0, 3, 3), 4.0);
}

TEMPLATE_TEST_CASE_2("upsample/deep/2d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 2, 4, 4> c;

    c = etl::upsample_2d<2, 2>(a);

    REQUIRE_EQUALS(c(0, 0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 0, 1), 1.0);
    REQUIRE_EQUALS(c(0, 1, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 0, 3), 2.0);
    REQUIRE_EQUALS(c(0, 1, 2), 2.0);
    REQUIRE_EQUALS(c(0, 1, 3), 2.0);

    REQUIRE_EQUALS(c(0, 2, 0), 3.0);
    REQUIRE_EQUALS(c(0, 2, 1), 3.0);
    REQUIRE_EQUALS(c(0, 3, 0), 3.0);
    REQUIRE_EQUALS(c(0, 3, 1), 3.0);

    REQUIRE_EQUALS(c(0, 2, 2), 4.0);
    REQUIRE_EQUALS(c(0, 2, 3), 4.0);
    REQUIRE_EQUALS(c(0, 3, 2), 4.0);
    REQUIRE_EQUALS(c(0, 3, 3), 4.0);

    REQUIRE_EQUALS(c(1, 0, 0), 5.0);
    REQUIRE_EQUALS(c(1, 0, 1), 5.0);
    REQUIRE_EQUALS(c(1, 1, 0), 5.0);
    REQUIRE_EQUALS(c(1, 1, 1), 5.0);

    REQUIRE_EQUALS(c(1, 0, 2), 6.0);
    REQUIRE_EQUALS(c(1, 0, 3), 6.0);
    REQUIRE_EQUALS(c(1, 1, 2), 6.0);
    REQUIRE_EQUALS(c(1, 1, 3), 6.0);

    REQUIRE_EQUALS(c(1, 2, 0), 7.0);
    REQUIRE_EQUALS(c(1, 2, 1), 7.0);
    REQUIRE_EQUALS(c(1, 3, 0), 7.0);
    REQUIRE_EQUALS(c(1, 3, 1), 7.0);

    REQUIRE_EQUALS(c(1, 2, 2), 8.0);
    REQUIRE_EQUALS(c(1, 2, 3), 8.0);
    REQUIRE_EQUALS(c(1, 3, 2), 8.0);
    REQUIRE_EQUALS(c(1, 3, 3), 8.0);
}

TEMPLATE_TEST_CASE_2("upsample/deep/3d/1", "[pooling]", Z, float, double) {
    etl::fast_matrix<Z, 2, 1, 2, 2> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    etl::fast_matrix<Z, 2, 1, 4, 4> c;

    c = etl::upsample_3d<1, 2, 2>(a);

    REQUIRE_EQUALS(c(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 0, 0, 1), 1.0);
    REQUIRE_EQUALS(c(0, 0, 1, 0), 1.0);
    REQUIRE_EQUALS(c(0, 0, 1, 1), 1.0);

    REQUIRE_EQUALS(c(0, 0, 0, 2), 2.0);
    REQUIRE_EQUALS(c(0, 0, 0, 3), 2.0);
    REQUIRE_EQUALS(c(0, 0, 1, 2), 2.0);
    REQUIRE_EQUALS(c(0, 0, 1, 3), 2.0);

    REQUIRE_EQUALS(c(0, 0, 2, 0), 3.0);
    REQUIRE_EQUALS(c(0, 0, 2, 1), 3.0);
    REQUIRE_EQUALS(c(0, 0, 3, 0), 3.0);
    REQUIRE_EQUALS(c(0, 0, 3, 1), 3.0);

    REQUIRE_EQUALS(c(0, 0, 2, 2), 4.0);
    REQUIRE_EQUALS(c(0, 0, 2, 3), 4.0);
    REQUIRE_EQUALS(c(0, 0, 3, 2), 4.0);
    REQUIRE_EQUALS(c(0, 0, 3, 3), 4.0);

    REQUIRE_EQUALS(c(1, 0, 0, 0), 5.0);
    REQUIRE_EQUALS(c(1, 0, 0, 1), 5.0);
    REQUIRE_EQUALS(c(1, 0, 1, 0), 5.0);
    REQUIRE_EQUALS(c(1, 0, 1, 1), 5.0);

    REQUIRE_EQUALS(c(1, 0, 0, 2), 6.0);
    REQUIRE_EQUALS(c(1, 0, 0, 3), 6.0);
    REQUIRE_EQUALS(c(1, 0, 1, 2), 6.0);
    REQUIRE_EQUALS(c(1, 0, 1, 3), 6.0);

    REQUIRE_EQUALS(c(1, 0, 2, 0), 7.0);
    REQUIRE_EQUALS(c(1, 0, 2, 1), 7.0);
    REQUIRE_EQUALS(c(1, 0, 3, 0), 7.0);
    REQUIRE_EQUALS(c(1, 0, 3, 1), 7.0);

    REQUIRE_EQUALS(c(1, 0, 2, 2), 8.0);
    REQUIRE_EQUALS(c(1, 0, 2, 3), 8.0);
    REQUIRE_EQUALS(c(1, 0, 3, 2), 8.0);
    REQUIRE_EQUALS(c(1, 0, 3, 3), 8.0);
}
