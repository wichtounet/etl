//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <vector>

/// Dim

TEMPLATE_TEST_CASE_2("dim/fast_matrix_1", "dim<1>", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<Z, 3> b;
    etl::fast_vector<Z, 3> c;

    b = etl::dim<1>(a, 0);
    c = etl::dim<1>(a, 1);

    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::dim<1>(a, 0))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::dim<1>(a, 1))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 4.0);

    REQUIRE_EQUALS(c[0], 3.0);
    REQUIRE_EQUALS(c[1], 0.5);
    REQUIRE_EQUALS(c[2], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/fast_matrix_2", "dim<2>", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<Z, 2> b;
    etl::fast_vector<Z, 2> c;
    etl::fast_vector<Z, 2> d;

    b = etl::dim<2>(a, 0);
    c = etl::dim<2>(a, 1);
    d = etl::dim<2>(a, 2);

    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::dim<2>(a, 0))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::dim<2>(a, 1))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::dim<2>(a, 2))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 3.0);

    REQUIRE_EQUALS(c[0], -2.0);
    REQUIRE_EQUALS(c[1], 0.5);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/fast_matrix_3", "row", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<Z, 3> b;
    etl::fast_vector<Z, 3> c;

    b = etl::row(a, 0);
    c = etl::row(a, 1);

    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::row(a, 0))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::row(a, 1))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 4.0);

    REQUIRE_EQUALS(c[0], 3.0);
    REQUIRE_EQUALS(c[1], 0.5);
    REQUIRE_EQUALS(c[2], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/fast_matrix_4", "col", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<Z, 2> b;
    etl::fast_vector<Z, 2> c;
    etl::fast_vector<Z, 2> d;

    b = etl::col(a, 0);
    c = etl::col(a, 1);
    d = etl::col(a, 2);

    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::col(a, 0))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::col(a, 1))>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<decltype(etl::col(a, 2))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 3.0);

    REQUIRE_EQUALS(c[0], -2.0);
    REQUIRE_EQUALS(c[1], 0.5);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/dyn_matrix_1", "dim<1>", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, std::initializer_list<Z>({1.0, -2.0, 4.0, 3.0, 0.5, -0.1}));
    etl::dyn_vector<Z> b;
    etl::dyn_vector<Z> c;

    b = etl::dim<1>(a, 0);
    c = etl::dim<1>(a, 1);

    REQUIRE_DIRECT(!etl::etl_traits<decltype(etl::dim<1>(a, 0))>::is_fast);
    REQUIRE_DIRECT(!etl::etl_traits<decltype(etl::dim<1>(a, 1))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 4.0);

    REQUIRE_EQUALS(c[0], 3.0);
    REQUIRE_EQUALS(c[1], 0.5);
    REQUIRE_EQUALS(c[2], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/dyn_matrix_2", "dim<2>", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, std::initializer_list<Z>({1.0, -2.0, 4.0, 3.0, 0.5, -0.1}));
    etl::dyn_vector<Z> b;
    etl::dyn_vector<Z> c;
    etl::dyn_vector<Z> d;

    b = etl::dim<2>(a, 0);
    c = etl::dim<2>(a, 1);
    d = etl::dim<2>(a, 2);

    REQUIRE_DIRECT(!etl::etl_traits<decltype(etl::dim<2>(a, 0))>::is_fast);
    REQUIRE_DIRECT(!etl::etl_traits<decltype(etl::dim<2>(a, 1))>::is_fast);
    REQUIRE_DIRECT(!etl::etl_traits<decltype(etl::dim<2>(a, 2))>::is_fast);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 3.0);

    REQUIRE_EQUALS(c[0], -2.0);
    REQUIRE_EQUALS(c[1], 0.5);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("dim/mix", "dim", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<Z, 3> b({0.1, 0.2, 0.3});
    etl::fast_vector<Z, 3> c;

    c = b >> row(a, 1);

    REQUIRE_EQUALS_APPROX(c[0], 0.3);
    REQUIRE_EQUALS_APPROX(c[1], 0.1);
    REQUIRE_EQUALS_APPROX(c[2], -0.03);
}

// reshape

TEMPLATE_TEST_CASE_2("reshape/fast_vector_1", "reshape<2,2>", Z, float, double) {
    etl::fast_vector<Z, 4> a({1, 2, 3, 4});
    etl::fast_matrix<Z, 2, 2> b;
    b = etl::reshape<2, 2>(a);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("reshape/fast_vector_2", "reshape<2,3>", Z, float, double) {
    etl::fast_vector<Z, 6> a({1, 2, 3, 4, 5, 6});
    etl::fast_matrix<Z, 2, 3> b;
    b = etl::reshape<2, 3>(a);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 2), 3.0);

    REQUIRE_EQUALS(b(1, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1), 5.0);
    REQUIRE_EQUALS(b(1, 2), 6.0);
}

TEMPLATE_TEST_CASE_2("reshape/fast_vector_3", "reshape<2,2>", Z, float, double) {
    etl::fast_vector<Z, 4> a({1, 2, 3, 4});
    etl::fast_matrix<Z, 1, 2, 2> b;
    b = etl::reshape<1, 2, 2>(a);

    REQUIRE_EQUALS(b(0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1), 2.0);

    REQUIRE_EQUALS(b(0, 1, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("reshape/fast_vector_4", "reshape<2,2>", Z, float, double) {
    etl::fast_vector<Z, 4> a({1, 2, 3, 4});
    etl::fast_matrix<Z, 1, 2, 2, 1> b;
    b = etl::reshape<1, 2, 2, 1>(a);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 2.0);

    REQUIRE_EQUALS(b(0, 1, 0, 0), 3.0);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 4.0);

    const auto& c = etl::reshape<1, 2, 2, 1>(a);

    REQUIRE_EQUALS(c[0], a[0]);
}

TEMPLATE_TEST_CASE_2("reshape/traits", "traits<reshape<2,3>>", Z, float, double) {
    etl::fast_vector<Z, 6> a({1, 2, 3, 4, 5, 6});

    using expr_type = decltype(etl::reshape<2, 3>(a));
    expr_type expr((etl::fast_matrix_view<etl::fast_vector<Z, 6>&, true, 2, 3>(a)));

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 2UL);
    REQUIRE_EQUALS(etl::columns(expr), 3UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();

    REQUIRE_EQUALS(size_1, 6UL);

    constexpr const auto size_2    = etl::size(expr);
    constexpr const auto rows_2    = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 2UL);
    REQUIRE_EQUALS(columns_2, 3UL);
}

TEMPLATE_TEST_CASE_2("reshape/dyn_vector_1", "reshape(2,2)", Z, float, double) {
    etl::dyn_vector<Z> a({1, 2, 3, 4});
    etl::dyn_matrix<Z> b;
    b = etl::reshape(a, 2, 2);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);

    REQUIRE_EQUALS(b(1, 0), 3.0);
    REQUIRE_EQUALS(b(1, 1), 4.0);
}

TEMPLATE_TEST_CASE_2("reshape/dyn_vector_2", "reshape(2,3)", Z, float, double) {
    etl::dyn_vector<Z> a({1, 2, 3, 4, 5, 6});
    etl::dyn_matrix<Z> b;
    b = etl::reshape(a, 2, 3);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 2.0);
    REQUIRE_EQUALS(b(0, 2), 3.0);

    REQUIRE_EQUALS(b(1, 0), 4.0);
    REQUIRE_EQUALS(b(1, 1), 5.0);
    REQUIRE_EQUALS(b(1, 2), 6.0);

    const auto& c = etl::reshape(a, 2, 3);

    REQUIRE_EQUALS(c[0], a[0]);
}

TEMPLATE_TEST_CASE_2("reshape/dyn_vector_3", "reshape(6)", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values<Z>(1, 2, 3, 4, 5, 6));
    etl::dyn_vector<Z> b;
    b = etl::reshape(a, 6);

    REQUIRE_EQUALS(b(0), 1.0);
    REQUIRE_EQUALS(b(1), 2.0);
    REQUIRE_EQUALS(b(2), 3.0);
    REQUIRE_EQUALS(b(3), 4.0);
    REQUIRE_EQUALS(b(4), 5.0);
    REQUIRE_EQUALS(b(5), 6.0);

    const auto& c = etl::reshape(a, 6);

    REQUIRE_EQUALS(c[0], a[0]);
}

TEMPLATE_TEST_CASE_2("reshape/dyn/1", "", Z, float, double) {
    etl::dyn_matrix<Z> a(4, 4, etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<Z, 4> b;
    b = etl::reshape(a, 2, 2, 2, 2);

    REQUIRE_EQUALS(b(0, 0, 0, 0), 1);
    REQUIRE_EQUALS(b(0, 0, 0, 1), 2);
    REQUIRE_EQUALS(b(0, 0, 1, 0), 3);
    REQUIRE_EQUALS(b(0, 0, 1, 1), 4);
    REQUIRE_EQUALS(b(0, 1, 0, 0), 5);
    REQUIRE_EQUALS(b(0, 1, 0, 1), 6);
    REQUIRE_EQUALS(b(0, 1, 1, 0), 7);
    REQUIRE_EQUALS(b(0, 1, 1, 1), 8);

    REQUIRE_EQUALS(b(1, 0, 0, 0), 9);
    REQUIRE_EQUALS(b(1, 0, 0, 1), 10);
    REQUIRE_EQUALS(b(1, 0, 1, 0), 11);
    REQUIRE_EQUALS(b(1, 0, 1, 1), 12);
    REQUIRE_EQUALS(b(1, 1, 0, 0), 13);
    REQUIRE_EQUALS(b(1, 1, 0, 1), 14);
    REQUIRE_EQUALS(b(1, 1, 1, 0), 15);
    REQUIRE_EQUALS(b(1, 1, 1, 1), 16);

    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 0, 0, 0), 1);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 0, 0, 1), 2);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 0, 1, 0), 3);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 0, 1, 1), 4);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 1, 0, 0), 5);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 1, 0, 1), 6);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 1, 1, 0), 7);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(0, 1, 1, 1), 8);

    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 0, 0, 0), 9);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 0, 0, 1), 10);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 0, 1, 0), 11);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 0, 1, 1), 12);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 1, 0, 0), 13);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 1, 0, 1), 14);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 1, 1, 0), 15);
    REQUIRE_EQUALS(etl::reshape(a, 2, 2, 2, 2)(1, 1, 1, 1), 16);
}

TEMPLATE_TEST_CASE_2("reshape/dyn_traits", "traits<reshape<2,3>>", Z, float, double) {
    etl::dyn_vector<Z> a({1, 2, 3, 4, 5, 6});

    using expr_type = decltype(etl::reshape(a, 2, 3));
    expr_type expr((etl::dyn_matrix_view<etl::dyn_vector<Z>&, 2>(a, 2UL, 3UL)));

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 2UL);
    REQUIRE_EQUALS(etl::columns(expr), 3UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_fast);
}

TEMPLATE_TEST_CASE_2("reshape/expr", "reshape(a+b)", Z, float, double) {
    etl::dyn_vector<Z> a({1, 2, 3, 4, 5, 6});
    etl::dyn_vector<Z> b({1, 2, 3, 4, 5, 6});

    etl::dyn_matrix<Z> c;
    c = etl::reshape(a + b, 2, 3);

    REQUIRE_EQUALS(c(0, 0), 2.0);
    REQUIRE_EQUALS(c(0, 1), 4.0);
    REQUIRE_EQUALS(c(0, 2), 6.0);

    REQUIRE_EQUALS(c(1, 0), 8.0);
    REQUIRE_EQUALS(c(1, 1), 10.0);
    REQUIRE_EQUALS(c(1, 2), 12.0);
}

// sub

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_1_1", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(a(0)(0, 0), Z(1.1));
    REQUIRE_EQUALS(a(0)(0, 1), 2.0);
    REQUIRE_EQUALS(a(1)(0, 0), Z(1.1));
    REQUIRE_EQUALS(a(1)(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_1_2", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(a(0)(0)(0), Z(1.1));
    REQUIRE_EQUALS(a(0)(0)(1), 2.0);
    REQUIRE_EQUALS(a(0)(1)(0), 5.0);
    REQUIRE_EQUALS(a(0)(1)(1), 1.0);
    REQUIRE_EQUALS(a(1)(0)(0), Z(1.1));
    REQUIRE_EQUALS(a(1)(0)(1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_1_3", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> b;
    b = a(1);

    REQUIRE_EQUALS(b(0, 0), Z(1.1));
    REQUIRE_EQUALS(b(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_1_1", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    REQUIRE_EQUALS(a(0, 0, 0), Z(1.1));
    REQUIRE_EQUALS(a(0)(0, 0), Z(1.1));
    REQUIRE_EQUALS(a(0)(0, 1), 2.0);
    REQUIRE_EQUALS(a(1)(0, 0), Z(1.1));
    REQUIRE_EQUALS(a(1)(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_1_2", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    REQUIRE_EQUALS(a(0)(0)(0), Z(1.1));
    REQUIRE_EQUALS(a(0)(0)(1), 2.0);
    REQUIRE_EQUALS(a(0)(1)(0), 5.0);
    REQUIRE_EQUALS(a(0)(1)(1), 1.0);
    REQUIRE_EQUALS(a(1)(0)(0), Z(1.1));
    REQUIRE_EQUALS(a(1)(0)(1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_1_3", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    etl::dyn_matrix<Z> b;
    b = a(1);

    REQUIRE_EQUALS(b(0, 0), Z(1.1));
    REQUIRE_EQUALS(b(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_1", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(etl::sub(a, 0)(0, 0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(a, 0)(0, 1), 2.0);
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_2", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 0)(0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 0)(1), 2.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 1)(0), 5.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 1)(1), 1.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 1), 0)(0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 1), 0)(1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_3", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> b;
    b = etl::sub(a, 1);

    REQUIRE_EQUALS(b(0, 0), Z(1.1));
    REQUIRE_EQUALS(b(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_4", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_vector<Z, 2> b;
    b = etl::sub(etl::sub(a, 1), 0);

    REQUIRE_EQUALS(b(0), Z(1.1));
    REQUIRE_EQUALS(b(1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_5", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_vector<Z, 2> b;
    b = 2.0 * etl::sub(2.0 * etl::sub(a, 1), 0);

    REQUIRE_EQUALS(b(0), Z(4.4));
    REQUIRE_EQUALS(b(1), 8.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_6", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 0), 1.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 1), -2.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 0), 3.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 1), 0.5);
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 0), 0.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 1), -1);

    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 0), 1.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 1), -2.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 0), 3.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 1), 0.5);
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 0), 0.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 1), -1);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_view_7", "[sub]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    test_matrix(0)(0, 0) = 2.0;
    test_matrix(1)(0, 0) = 2.0;

    auto s = test_matrix(1);
    s(1,1) = Z(0.7);

    auto&& ss = s;
    ss(2, 0) = Z(7.4);

    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 0), Z(2.0));
    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 1), Z(-2.0));
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 0), Z(3.0));
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 1), Z(0.5));
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 0), Z(0.0));
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 1), Z(-1));

    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 0), Z(2.0));
    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 1), Z(-2.0));
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 0), Z(3.0));
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 1), Z(0.7));
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 0), Z(7.4));
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 1), Z(-1));
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_1", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    REQUIRE_EQUALS(a(0, 0, 0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(a, 0)(0, 0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(a, 0)(0, 1), 2.0);
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_2", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 0)(0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 0)(1), 2.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 1)(0), 5.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 0), 1)(1), 1.0);
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 1), 0)(0), Z(1.1));
    REQUIRE_EQUALS(etl::sub(etl::sub(a, 1), 0)(1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_3", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    etl::dyn_matrix<Z> b;
    b = etl::sub(a, 1);

    REQUIRE_EQUALS(b(0, 0), Z(1.1));
    REQUIRE_EQUALS(b(0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_4", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    etl::dyn_vector<Z> b;
    b = etl::sub(etl::sub(a, 1), 0);

    REQUIRE_EQUALS(b(0), Z(1.1));
    REQUIRE_EQUALS(b(1), 2.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_5", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    etl::dyn_vector<Z> b;
    b = 2.0 * etl::sub(2.0 * etl::sub(a, 1), 0);

    REQUIRE_EQUALS(b(0), Z(4.4));
    REQUIRE_EQUALS(b(1), 8.0);
}

TEMPLATE_TEST_CASE_2("dyn_matrix/sub_view_6", "dyn_matrix::sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> test_matrix(2, 3, 2, etl::values(1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1));

    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 0), 1.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(0, 1), -2.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 0), 3.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(1, 1), 0.5);
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 0), 0.0);
    REQUIRE_EQUALS(sub(test_matrix, 0)(2, 1), -1);

    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 0), 1.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(0, 1), -2.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 0), 3.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(1, 1), 0.5);
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 0), 0.0);
    REQUIRE_EQUALS(sub(test_matrix, 1)(2, 1), -1);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_compound_1", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    a(0) += 5.0;

    REQUIRE_EQUALS(a(0, 0, 0), Z(6.1));
    REQUIRE_EQUALS(a(0, 0, 1), 7.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_compound_2", "fast_matrix::sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    a(0) += a(1) * 2.0;

    REQUIRE_EQUALS_APPROX(a(0, 0, 0), 3.3);
    REQUIRE_EQUALS_APPROX(a(0, 0, 1), 6.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
    REQUIRE_EQUALS(a(1, 0, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_min_1", "fast_matrix::sub::min", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(etl::max(etl::sub(a, 0)), 5.0);
    REQUIRE_EQUALS(etl::min(etl::sub(a, 0)), 1.0);

    auto m1 = etl::max(etl::sub(a, 0));
    auto m2 = etl::min(etl::sub(a, 0));

    REQUIRE_EQUALS(m1, 5.0);
    REQUIRE_EQUALS(m2, 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_min_2", "fast_matrix::sub::min", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> b = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    auto const& a = b;

    REQUIRE_EQUALS(etl::max(etl::sub(a, 0)), 5.0);
    REQUIRE_EQUALS(etl::min(etl::sub(a, 0)), 1.0);

    auto m1 = etl::max(etl::sub(a, 0));
    auto m2 = etl::min(etl::sub(a, 0));

    REQUIRE_EQUALS(m1, 5.0);
    REQUIRE_EQUALS(m2, 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_min_3", "fast_matrix::sub::min", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE_EQUALS(etl::max(a), 5.0);
    REQUIRE_EQUALS(etl::min(a), 1.0);

    auto& m1 = etl::max(a);
    auto& m2 = etl::min(a);

    REQUIRE_EQUALS(m1, 5.0);
    REQUIRE_EQUALS(m2, 1.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/sub_min_4", "fast_matrix::sub::min", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> b = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    auto const& a = b;

    REQUIRE_EQUALS(etl::max(a), 5.0);
    REQUIRE_EQUALS(etl::min(a), 1.0);

    auto& m1 = etl::max(a);
    auto& m2 = etl::min(a);

    REQUIRE_EQUALS(m1, 5.0);
    REQUIRE_EQUALS(m2, 1.0);
}

/// lvalue access

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_1", "lvalue dim<1>", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    auto b = etl::dim<1>(a, 0);
    auto c = etl::dim<1>(a, 1);

    b[0] = 3.0;
    b[1] = -5.0;
    b[2] = 10.0;

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], -5.0);
    REQUIRE_EQUALS(b[2], 10.0);

    c[0] = -3.0;
    c[1] = 5.0;
    c[2] = -10.0;

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], -10.0);
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_2", "lvalue col", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    auto b = etl::col(a, 0);
    auto c = etl::col(a, 1);
    auto d = etl::col(a, 2);

    b[0] = 3.0;
    b[1] = 2.5;

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], 2.5);

    c[0] = -3.0;
    c[1] = -2.5;

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], -2.5);

    d[0] = -5.0;
    d[1] = -7.5;

    REQUIRE_EQUALS(d[0], -5.0);
    REQUIRE_EQUALS(d[1], -7.5);
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_3", "lvalue dim<1>", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    auto b = etl::dim<1>(a, 0);
    auto c = etl::dim<1>(a, 1);

    b(0) = 3.0;
    b(1) = -5.0;
    b(2) = 10.0;

    REQUIRE_EQUALS(b(0), 3.0);
    REQUIRE_EQUALS(b(1), -5.0);
    REQUIRE_EQUALS(b(2), 10.0);

    c(0) = -3.0;
    c(1) = 5.0;
    c(2) = -10.0;

    REQUIRE_EQUALS(c(0), -3.0);
    REQUIRE_EQUALS(c(1), 5.0);
    REQUIRE_EQUALS(c(2), -10.0);
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_4", "lvalue col", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    auto b = etl::col(a, 0);
    auto c = etl::col(a, 1);
    auto d = etl::col(a, 2);

    b(0) = 3.0;
    b(1) = 2.5;

    REQUIRE_EQUALS(b(0), 3.0);
    REQUIRE_EQUALS(b(1), 2.5);

    c(0) = -3.0;
    c(1) = -2.5;

    REQUIRE_EQUALS(c(0), -3.0);
    REQUIRE_EQUALS(c(1), -2.5);

    d(0) = -5.0;
    d(1) = -7.5;

    REQUIRE_EQUALS(d(0), -5.0);
    REQUIRE_EQUALS(d(1), -7.5);
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_5", "lvalue const col", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});

    const auto& ref = a;

    auto b = etl::col(ref, 0);
    auto c = etl::col(ref, 1);
    auto d = etl::col(ref, 2);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 3.0);

    REQUIRE_EQUALS(c[0], -2.0);
    REQUIRE_EQUALS(c[1], 0.5);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_6", "lvalue const dim<1>", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});

    const auto& ref = a;

    auto b = etl::dim<1>(ref, 0);
    auto c = etl::dim<1>(ref, 1);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 4.0);

    REQUIRE_EQUALS(c[0], 3.0);
    REQUIRE_EQUALS(c[1], 0.5);
    REQUIRE_EQUALS(c[2], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_7", "lvalue reshape", Z, float, double) {
    etl::fast_vector<Z, 4> a({1, 2, 3, 4});
    auto b = etl::reshape<2, 2>(a);

    b(0, 0) = 32.0;
    b(0, 1) = 23.0;
    b(1, 0) = 11.0;
    b(1, 1) = 12.0;

    REQUIRE_EQUALS(b(0, 0), 32.0);
    REQUIRE_EQUALS(b(0, 1), 23.0);

    REQUIRE_EQUALS(b(1, 0), 11.0);
    REQUIRE_EQUALS(b(1, 1), 12.0);
}

TEMPLATE_TEST_CASE_2("lvalue/fast_matrix_8", "lvalue sub", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::sub(a, 0)(0, 0) = 2.2;
    etl::sub(a, 0)(0, 1) = 3.2;
    etl::sub(a, 1)(0, 0) = 4.2;
    etl::sub(a, 1)(0, 1) = 5.2;

    REQUIRE_EQUALS(etl::sub(a, 0)(0, 0), Z(2.2));
    REQUIRE_EQUALS(etl::sub(a, 0)(0, 1), Z(3.2));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 0), Z(4.2));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 1), Z(5.2));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_1", "lvalue dim<1>", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));
    auto b = etl::dim<1>(a, 0);
    auto c = etl::dim<1>(a, 1);

    b[0] = 3.0;
    b[1] = -5.0;
    b[2] = 10.0;

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], -5.0);
    REQUIRE_EQUALS(b[2], 10.0);

    c[0] = -3.0;
    c[1] = 5.0;
    c[2] = -10.0;

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], 5.0);
    REQUIRE_EQUALS(c[2], -10.0);
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_2", "lvalue col", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));
    auto b = etl::col(a, 0);
    auto c = etl::col(a, 1);
    auto d = etl::col(a, 2);

    b[0] = 3.0;
    b[1] = 2.5;

    REQUIRE_EQUALS(b[0], 3.0);
    REQUIRE_EQUALS(b[1], 2.5);

    c[0] = -3.0;
    c[1] = -2.5;

    REQUIRE_EQUALS(c[0], -3.0);
    REQUIRE_EQUALS(c[1], -2.5);

    d[0] = -5.0;
    d[1] = -7.5;

    REQUIRE_EQUALS(d[0], -5.0);
    REQUIRE_EQUALS(d[1], -7.5);
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_3", "lvalue dim<1>", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));
    auto b = etl::dim<1>(a, 0);
    auto c = etl::dim<1>(a, 1);

    b(0) = 3.0;
    b(1) = -5.0;
    b(2) = 10.0;

    REQUIRE_EQUALS(b(0), 3.0);
    REQUIRE_EQUALS(b(1), -5.0);
    REQUIRE_EQUALS(b(2), 10.0);

    c(0) = -3.0;
    c(1) = 5.0;
    c(2) = -10.0;

    REQUIRE_EQUALS(c(0), -3.0);
    REQUIRE_EQUALS(c(1), 5.0);
    REQUIRE_EQUALS(c(2), -10.0);
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_4", "lvalue col", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));
    auto b = etl::col(a, 0);
    auto c = etl::col(a, 1);
    auto d = etl::col(a, 2);

    b(0) = 3.0;
    b(1) = 2.5;

    REQUIRE_EQUALS(b(0), 3.0);
    REQUIRE_EQUALS(b(1), 2.5);

    c(0) = -3.0;
    c(1) = -2.5;

    REQUIRE_EQUALS(c(0), -3.0);
    REQUIRE_EQUALS(c(1), -2.5);

    d(0) = -5.0;
    d(1) = -7.5;

    REQUIRE_EQUALS(d(0), -5.0);
    REQUIRE_EQUALS(d(1), -7.5);
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_5", "lvalue const col", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));

    const auto& ref = a;

    auto b = etl::col(ref, 0);
    auto c = etl::col(ref, 1);
    auto d = etl::col(ref, 2);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], 3.0);

    REQUIRE_EQUALS(c[0], -2.0);
    REQUIRE_EQUALS(c[1], 0.5);

    REQUIRE_EQUALS(d[0], 4.0);
    REQUIRE_EQUALS(d[1], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_6", "lvalue const dim<1>", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 3, etl::values(1.0, -2.0, 4.0, 3.0, 0.5, -0.1));

    const auto& ref = a;

    auto b = etl::dim<1>(ref, 0);
    auto c = etl::dim<1>(ref, 1);

    REQUIRE_EQUALS(b[0], 1.0);
    REQUIRE_EQUALS(b[1], -2.0);
    REQUIRE_EQUALS(b[2], 4.0);

    REQUIRE_EQUALS(c[0], 3.0);
    REQUIRE_EQUALS(c[1], 0.5);
    REQUIRE_EQUALS(c[2], Z(-0.1));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_7", "lvalue reshape", Z, float, double) {
    etl::dyn_vector<Z> a(4, etl::values(1, 2, 3, 4));
    auto b = etl::reshape(a, 2, 2);

    b(0, 0) = 32.0;
    b(0, 1) = 23.0;
    b(1, 0) = 11.0;
    b(1, 1) = 12.0;

    REQUIRE_EQUALS(b(0, 0), 32.0);
    REQUIRE_EQUALS(b(0, 1), 23.0);

    REQUIRE_EQUALS(b(1, 0), 11.0);
    REQUIRE_EQUALS(b(1, 1), 12.0);
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_8", "lvalue sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    etl::sub(a, 0)(0, 0) = 2.2;
    etl::sub(a, 0)(0, 1) = 3.2;
    etl::sub(a, 1)(0, 0) = 4.2;
    etl::sub(a, 1)(0, 1) = 5.2;

    REQUIRE_EQUALS(etl::sub(a, 0)(0, 0), Z(2.2));
    REQUIRE_EQUALS(etl::sub(a, 0)(0, 1), Z(3.2));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 0), Z(4.2));
    REQUIRE_EQUALS(etl::sub(a, 1)(0, 1), Z(5.2));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_9", "lvalue sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0));

    a(0) = 0.0;

    REQUIRE_EQUALS(a(0, 0, 0), 0.0);
    REQUIRE_EQUALS(a(0, 0, 1), 0.0);
    REQUIRE_EQUALS(a(0, 1, 0), 0.0);
    REQUIRE_EQUALS(a(0, 1, 1), 0.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_10", "lvalue sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(0.1, 3.0, 15.0, 2.0, 1.1, 2.0, 5.0, 1.0));

    a(0) = a(1);

    REQUIRE_EQUALS(a(0, 0, 0), Z(1.1));
    REQUIRE_EQUALS(a(0, 0, 1), 2.0);
    REQUIRE_EQUALS(a(0, 1, 0), 5.0);
    REQUIRE_EQUALS(a(0, 1, 1), 1.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_11", "lvalue sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(0.1, 3.0, 15.0, 2.0, 1.1, 2.0, 5.0, 1.0));

    a(0) = 2.0 * a(0);

    REQUIRE_EQUALS(a(0, 0, 0), Z(0.2));
    REQUIRE_EQUALS(a(0, 0, 1), 6.0);
    REQUIRE_EQUALS(a(0, 1, 0), 30.0);
    REQUIRE_EQUALS(a(0, 1, 1), 4.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
}

TEMPLATE_TEST_CASE_2("lvalue/dyn_matrix_12", "lvalue sub", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 2, 2, etl::values(0.1, 3.0, 15.0, 2.0, 1.1, 2.0, 5.0, 1.0));

    std::vector<Z> test({1.0, 2.0, 3.0, 4.0});

    a(0) = test;

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 2.0);
    REQUIRE_EQUALS(a(0, 1, 0), 3.0);
    REQUIRE_EQUALS(a(0, 1, 1), 4.0);
    REQUIRE_EQUALS(a(1, 0, 0), Z(1.1));
}
