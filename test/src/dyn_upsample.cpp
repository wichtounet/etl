//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <vector>

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/0", "[upsample]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 3> b(1, 2, 2, etl::values(1.0, 2.0, 3.0, 4.0));

    auto expr2 = etl::upsample_2d(a, 2, 2);
    using expr2_type = decltype(expr2);

    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::dim(expr2, 0), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::dim(expr2, 1), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr2_type>::size(expr2), 16UL);

    auto expr3 = etl::upsample_2d(b, 2, 2);
    using expr3_type = decltype(expr3);

    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::dimensions(), 3UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::dim(expr3, 0), 1UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::dim(expr3, 1), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::dim(expr3, 2), 4UL);
    REQUIRE_EQUALS(etl::decay_traits<expr3_type>::size(expr3), 16UL);
}

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> c(4, 4);

    c = etl::upsample_2d(a, 2, 2);

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

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/2", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> c(3, 3);

    c = etl::upsample_2d(a, 2, 2, 1, 1);

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

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/4", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(1.0, 3.0, 4.0, 5.0, 7.0, 8.0, 5.0, 7.0, 8.0));
    etl::dyn_matrix<Z> c(4, 4);

    c = etl::upsample_2d(a, 2, 2, 2, 2, 1, 1);

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

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/5", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z> a(3, 3, etl::values(0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0));
    etl::dyn_matrix<Z> c(2, 2);

    c = etl::upsample_2d(a, 2, 2, 2, 2, 2, 2);

    REQUIRE_EQUALS(c(0, 0), 4.0);
    REQUIRE_EQUALS(c(0, 1), 4.0);

    REQUIRE_EQUALS(c(1, 0), 4.0);
    REQUIRE_EQUALS(c(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("dyn_upsample/2d/3", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 2> c(2, 2);

    c = etl::upsample_2d(a, 2, 2, 2, 2, 1, 1);

    REQUIRE_EQUALS(c(0, 0), 1.0);
    REQUIRE_EQUALS(c(0, 1), 2.0);

    REQUIRE_EQUALS(c(1, 0), 3.0);
    REQUIRE_EQUALS(c(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("dyn_upsample/3d/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(1, 2, 2, etl::values(1.0, 2.0, 3.0, 4.0));
    etl::dyn_matrix<Z, 3> c(1, 4, 4);

    c = etl::upsample_3d(a, 1, 2, 2);

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

TEMPLATE_TEST_CASE_2("dyn_upsample/deep/2d/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
    etl::dyn_matrix<Z, 3> c(2, 4, 4);

    c = etl::upsample_2d(a, 2, 2);

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

TEMPLATE_TEST_CASE_2("dyn_upsample/deep/3d/1", "[pooling]", Z, float, double) {
    etl::dyn_matrix<Z, 4> a(2, 1, 2, 2, etl::values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
    etl::dyn_matrix<Z, 4> c(2, 1, 4, 4);

    c = etl::upsample_3d(a, 1, 2, 2);

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
