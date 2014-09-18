//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/dyn_vector.hpp"

///{{{ Dim

TEST_CASE( "dim/fast_matrix_1", "dim<1>" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b(etl::dim<1>(a, 0));
    etl::fast_vector<double, 3> c(etl::dim<1>(a, 1));

    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/fast_matrix_2", "dim<2>" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 2> b(etl::dim<2>(a, 0));
    etl::fast_vector<double, 2> c(etl::dim<2>(a, 1));
    etl::fast_vector<double, 2> d(etl::dim<2>(a, 2));

    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/fast_matrix_3", "row" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b(etl::row(a, 0));
    etl::fast_vector<double, 3> c(etl::row(a, 1));

    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/fast_matrix_4", "col" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 2> b(etl::col(a, 0));
    etl::fast_vector<double, 2> c(etl::col(a, 1));
    etl::fast_vector<double, 2> d(etl::col(a, 2));

    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(etl::etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/dyn_matrix_1", "dim<1>" ) {
    etl::dyn_matrix<double> a(2, 3, {1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::dyn_vector<double> b(etl::dim<1>(a, 0));
    etl::dyn_vector<double> c(etl::dim<1>(a, 1));

    REQUIRE(!etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(!etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/dyn_matrix_2", "dim<2>" ) {
    etl::dyn_matrix<double> a(2, 3, {1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::dyn_vector<double> b(etl::dim<2>(a, 0));
    etl::dyn_vector<double> c(etl::dim<2>(a, 1));
    etl::dyn_vector<double> d(etl::dim<2>(a, 2));

    REQUIRE(!etl::etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(!etl::etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(!etl::etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/mix", "dim" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b({0.1, 0.2, 0.3});
    etl::fast_vector<double, 3> c(b * row(a,1));

    REQUIRE(c[0] == Approx(0.3));
    REQUIRE(c[1] == Approx(0.1));
    REQUIRE(c[2] == Approx(-0.03));
}

///}}}

//{{{ reshape

TEST_CASE( "reshape/fast_vector_1", "reshape<2,2>" ) {
    etl::fast_vector<double, 4> a({1,2,3,4});
    etl::fast_matrix<double, 2, 2> b(etl::reshape<2,2>(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 2.0);

    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 4.0);
}

TEST_CASE( "reshape/fast_vector_2", "reshape<2,3>" ) {
    etl::fast_vector<double, 6> a({1,2,3,4,5,6});
    etl::fast_matrix<double, 2, 3> b(etl::reshape<2,3>(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 2.0);
    REQUIRE(b(0,2) == 3.0);

    REQUIRE(b(1,0) == 4.0);
    REQUIRE(b(1,1) == 5.0);
    REQUIRE(b(1,2) == 6.0);
}

TEST_CASE( "reshape/traits", "traits<reshape<2,3>>" ) {
    etl::fast_vector<double, 6> a({1,2,3,4,5,6});

    using expr_type = decltype(etl::reshape<2,3>(a));
    expr_type expr((etl::fast_matrix_view<etl::fast_vector<double, 6>, 2, 3>(a)));

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::etl_traits<expr_type>::rows(expr) == 2);
    REQUIRE(etl::rows(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::columns(expr) == 3);
    REQUIRE(etl::columns(expr) == 3);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto rows_1 = etl::etl_traits<expr_type>::rows();
    constexpr const auto columns_1 = etl::etl_traits<expr_type>::columns();

    REQUIRE(size_1 == 6);
    REQUIRE(rows_1 == 2);
    REQUIRE(columns_1 == 3);

    constexpr const auto size_2 = etl::size(expr);
    constexpr const auto rows_2 = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 2);
    REQUIRE(columns_2 == 3);
}

TEST_CASE( "reshape/dyn_vector_1", "reshape(2,2)" ) {
    etl::dyn_vector<double> a({1,2,3,4});
    etl::dyn_matrix<double> b(etl::reshape(a,2,2));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 2.0);

    REQUIRE(b(1,0) == 3.0);
    REQUIRE(b(1,1) == 4.0);
}

TEST_CASE( "reshape/dyn_vector_2", "reshape(2,3)" ) {
    etl::dyn_vector<double> a({1,2,3,4,5,6});
    etl::dyn_matrix<double> b(etl::reshape(a,2,3));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 2.0);
    REQUIRE(b(0,2) == 3.0);

    REQUIRE(b(1,0) == 4.0);
    REQUIRE(b(1,1) == 5.0);
    REQUIRE(b(1,2) == 6.0);
}

TEST_CASE( "reshape/dyn_traits", "traits<reshape<2,3>>" ) {
    etl::dyn_vector<double> a({1,2,3,4,5,6});

    using expr_type = decltype(etl::reshape(a,2,3));
    expr_type expr((etl::dyn_matrix_view<etl::dyn_vector<double>>(a,2,3)));

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::etl_traits<expr_type>::rows(expr) == 2);
    REQUIRE(etl::rows(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::columns(expr) == 3);
    REQUIRE(etl::columns(expr) == 3);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);
}

//}}}

//{{{ sub

TEST_CASE( "fast_matrix/sub_view_1", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE(etl::sub(a, 0)(0, 0) == 1.1);
    REQUIRE(etl::sub(a, 0)(0, 1) == 2.0);
    REQUIRE(etl::sub(a, 1)(0, 0) == 1.1);
    REQUIRE(etl::sub(a, 1)(0, 1) == 2.0);
}

TEST_CASE( "fast_matrix/sub_view_2", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    REQUIRE(etl::sub(etl::sub(a, 0), 0)(0) == 1.1);
    REQUIRE(etl::sub(etl::sub(a, 0), 0)(1) == 2.0);
    REQUIRE(etl::sub(etl::sub(a, 0), 1)(0) == 5.0);
    REQUIRE(etl::sub(etl::sub(a, 0), 1)(1) == 1.0);
    REQUIRE(etl::sub(etl::sub(a, 1), 0)(0) == 1.1);
    REQUIRE(etl::sub(etl::sub(a, 1), 0)(1) == 2.0);
}

TEST_CASE( "fast_matrix/sub_view_3", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_matrix<double, 2, 2> b(etl::sub(a, 1));

    REQUIRE(b(0, 0) == 1.1);
    REQUIRE(b(0, 1) == 2.0);
}

TEST_CASE( "fast_matrix/sub_view_4", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_vector<double, 2> b(etl::sub(etl::sub(a, 1), 0));

    REQUIRE(b(0) == 1.1);
    REQUIRE(b(1) == 2.0);
}

TEST_CASE( "fast_matrix/sub_view_5", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 2, 2> a = {1.1, 2.0, 5.0, 1.0, 1.1, 2.0, 5.0, 1.0};

    etl::fast_vector<double, 2> b(2.0 * etl::sub(2.0 * etl::sub(a, 1), 0));

    REQUIRE(b(0) == 4.4);
    REQUIRE(b(1) == 8.0);
}

TEST_CASE( "fast_matrix/sub_view_6", "fast_matrix::sub" ) {
    etl::fast_matrix<double, 2, 3, 2> test_matrix({1.0, -2.0, 3.0, 0.5, 0.0, -1, 1.0, -2.0, 3.0, 0.5, 0.0, -1});

    REQUIRE(sub(test_matrix, 0)(0, 0) == 1.0);
    REQUIRE(sub(test_matrix, 0)(0, 1) == -2.0);
    REQUIRE(sub(test_matrix, 0)(1, 0) == 3.0);
    REQUIRE(sub(test_matrix, 0)(1, 1) == 0.5);
    REQUIRE(sub(test_matrix, 0)(2, 0) == 0.0);
    REQUIRE(sub(test_matrix, 0)(2, 1) == -1);

    REQUIRE(sub(test_matrix, 1)(0, 0) == 1.0);
    REQUIRE(sub(test_matrix, 1)(0, 1) == -2.0);
    REQUIRE(sub(test_matrix, 1)(1, 0) == 3.0);
    REQUIRE(sub(test_matrix, 1)(1, 1) == 0.5);
    REQUIRE(sub(test_matrix, 1)(2, 0) == 0.0);
    REQUIRE(sub(test_matrix, 1)(2, 1) == -1);
}

//}}}