//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("uni_upper/1", "[uni_upper]", Z, float, double) {
    etl::uni_upper_matrix<etl::fast_matrix<Z, 2,2>> a;
    etl::uni_upper_matrix<etl::fast_matrix<Z, 1,1>> b;
    etl::uni_upper_matrix<etl::fast_dyn_matrix<Z, 1,1>> c;

    using a_t = decltype(a);
    using b_t = decltype(b);
    using c_t = decltype(c);

    REQUIRE_EQUALS(a.dimensions(), 2UL);
    REQUIRE_EQUALS(b.dimensions(), 2UL);
    REQUIRE_EQUALS(c.dimensions(), 2UL);

    REQUIRE_DIRECT(etl::etl_traits<a_t>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<b_t>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<c_t>::is_fast);

    REQUIRE_EQUALS(etl::etl_traits<a_t>::size(a), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<b_t>::size(b), 1UL);
    REQUIRE_EQUALS(etl::etl_traits<c_t>::size(c), 1UL);

    REQUIRE_DIRECT(etl::is_uni_upper_triangular(a));
    REQUIRE_DIRECT(etl::is_uni_upper_triangular(b));
    REQUIRE_DIRECT(etl::is_uni_upper_triangular(c));
}

TEMPLATE_TEST_CASE_2("uni_upper/2", "[uni_upper]", Z, float, double) {
    etl::uni_upper_matrix<etl::fast_matrix<Z, 3, 3>> a;

    a(0, 1) = 1.2;
    a(0, 2) = 1.3;
    a(1, 2) = 1.5;

    REQUIRE_EQUALS(a(0, 0), Z(1.0));
    REQUIRE_EQUALS(a(0, 1), Z(1.2));
    REQUIRE_EQUALS(a(0, 2), Z(1.3));

    REQUIRE_EQUALS(a(1, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 1), Z(1.0));
    REQUIRE_EQUALS(a(1, 2), Z(1.5));

    REQUIRE_EQUALS(a(2, 0), Z(0.0));
    REQUIRE_EQUALS(a(2, 1), Z(0.0));
    REQUIRE_EQUALS(a(2, 2), Z(1.0));
}

TEMPLATE_TEST_CASE_2("uni_upper/3", "[uni_upper]", Z, float, double) {
    etl::uni_upper_matrix<etl::dyn_matrix<Z>> a(3UL);

    REQUIRE_EQUALS(a(1, 1), Z(1.0));

    a(0, 2) += 5.5;

    REQUIRE_EQUALS(a(0, 2), Z(5.5));

    a(1, 2) = 1.5;

    REQUIRE_EQUALS(a(1, 2), Z(1.5));

    a(1, 2) *= 2.0;

    REQUIRE_EQUALS(a(1, 2), Z(3.0));
}

TEMPLATE_TEST_CASE_2("uni_upper/4", "[uni_upper]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 1.2, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0};
    etl::uni_upper_matrix<etl::fast_matrix<Z, 3, 3>> b;
    b = a;

    REQUIRE_EQUALS(a(0, 0), Z(1.0));
    REQUIRE_EQUALS(a(0, 1), Z(1.2));
    REQUIRE_EQUALS(a(0, 2), Z(1.3));

    REQUIRE_EQUALS(a(1, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 1), Z(1.0));
    REQUIRE_EQUALS(a(1, 2), Z(1.5));

    REQUIRE_EQUALS(a(2, 0), Z(0.0));
    REQUIRE_EQUALS(a(2, 1), Z(0.0));
    REQUIRE_EQUALS(a(2, 2), Z(1.0));

    REQUIRE_DIRECT(b == a);
    REQUIRE_DIRECT(a == b);
}

TEMPLATE_TEST_CASE_2("uni_upper/5", "[uni_upper]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 1.2, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0};

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> b(3UL);
    b = a;

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    c = b >> c;

    REQUIRE_EQUALS(c(0, 0), Z(1.0));
    REQUIRE_EQUALS(c(0, 1), Z(1.2) * Z(1.2));
    REQUIRE_EQUALS(c(0, 2), Z(1.3) * Z(1.3));

    REQUIRE_EQUALS(c(1, 0), Z(0.0));
    REQUIRE_EQUALS(c(1, 1), Z(1.0));
    REQUIRE_EQUALS(c(1, 2), Z(1.5) * Z(1.5));

    REQUIRE_EQUALS(c(2, 0), Z(0.0));
    REQUIRE_EQUALS(c(2, 1), Z(0.0));
    REQUIRE_EQUALS(c(2, 2), Z(1.0));
}

TEMPLATE_TEST_CASE_2("uni_upper/6", "[uni_upper][fast]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 1.2, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0};

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> b(3UL);
    b = a;

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    c >>= b;

    REQUIRE_EQUALS(c(0, 0), Z(1.0));
    REQUIRE_EQUALS(c(0, 1), Z(1.2) * Z(1.2));
    REQUIRE_EQUALS(c(0, 2), Z(1.3) * Z(1.3));

    REQUIRE_EQUALS(c(1, 0), Z(0.0));
    REQUIRE_EQUALS(c(1, 1), Z(1.0));
    REQUIRE_EQUALS(c(1, 2), Z(1.5) * Z(1.5));

    REQUIRE_EQUALS(c(2, 0), Z(0.0));
    REQUIRE_EQUALS(c(2, 1), Z(0.0));
    REQUIRE_EQUALS(c(2, 2), Z(1.0));
}

TEMPLATE_TEST_CASE_2("uni_upper/7", "[uni_upper][fast]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {0.05, 1.2, 1.3, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0};
    etl::uni_upper_matrix<etl::fast_matrix<Z, 3, 3>> b;

    REQUIRE_THROWS(b = a);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 0.0);
    REQUIRE_EQUALS(b(0, 2), 0.0);

    REQUIRE_EQUALS(b(1, 0), 0.0);
    REQUIRE_EQUALS(b(1, 1), 1.0);
    REQUIRE_EQUALS(b(1, 2), 0.0);

    REQUIRE_EQUALS(b(2, 0), 0.0);
    REQUIRE_EQUALS(b(2, 1), 0.0);
    REQUIRE_EQUALS(b(2, 2), 1.0);
}

TEMPLATE_TEST_CASE_2("uni_upper/8", "[uni_upper][fast]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 1.2, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0};

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    a(2,1) = 2.0;

    REQUIRE_THROWS(c += a);

    REQUIRE_EQUALS(c(0, 0), Z(1.0));
    REQUIRE_EQUALS(c(0, 1), Z(1.2));
    REQUIRE_EQUALS(c(0, 2), Z(1.3));

    REQUIRE_EQUALS(c(1, 0), Z(0.0));
    REQUIRE_EQUALS(c(1, 1), Z(1.0));
    REQUIRE_EQUALS(c(1, 2), Z(1.5));

    REQUIRE_EQUALS(c(2, 0), Z(0.0));
    REQUIRE_EQUALS(c(2, 1), Z(0.0));
    REQUIRE_EQUALS(c(2, 2), Z(1.0));
}

TEMPLATE_TEST_CASE_2("uni_upper/9", "[uni_upper][fast]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 1.2, 1.3, 0.0, 1.0, 1.5, 0.0, 0.0, 1.0};

    etl::uni_upper_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    REQUIRE_THROWS(c(1,1) = 1.1);
    REQUIRE_THROWS(c(2,1) = 1.1);
}
