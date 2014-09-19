//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/traits.hpp"

TEST_CASE( "etl_traits/fast_vector_1", "etl_traits<fast_vector>" ) {
    using type = etl::fast_vector<double, 4>;
    type test_vector(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::dimensions() == 1);
    REQUIRE(etl::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::is_vector);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(!etl::etl_traits<type>::is_matrix);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    REQUIRE(size_1 == 4);

    constexpr const auto size_2 = etl::size(test_vector);
    REQUIRE(size_2 == 4);

    constexpr const auto size_3 = etl::etl_traits<type>::dim<0>();
    REQUIRE(size_3 == 4);

    constexpr const auto size_4 = etl::etl_traits<type>::dimensions();
    REQUIRE(size_4 == 1);

    constexpr const auto size_5 = etl::dimensions(test_vector);
    REQUIRE(size_5 == 1);
}

TEST_CASE( "etl_traits/fast_matrix_1", "etl_traits<fast_matrix>" ) {
    using type = etl::fast_matrix<double, 3, 2>;
    type test_matrix(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(etl::size(test_matrix) == 6);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 2);
    REQUIRE(etl::dimensions(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::is_matrix);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(!etl::etl_traits<type>::is_vector);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1 = etl::etl_traits<type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2 = etl::size(test_matrix);
    constexpr const auto rows_2 = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2 = etl::dimensions(test_matrix);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3 = etl::etl_traits<type>::dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}

TEST_CASE( "etl_traits/fast_matrix_2", "etl_traits<fast_matrix>" ) {
    using type = etl::fast_matrix<double, 3, 2, 4, 1>;
    type test_matrix(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 24);
    REQUIRE(etl::size(test_matrix) == 24);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 4);
    REQUIRE(etl::dimensions(test_matrix) == 4);
    REQUIRE(etl::etl_traits<type>::is_matrix);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(!etl::etl_traits<type>::is_vector);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1 = etl::etl_traits<type>::dimensions();

    REQUIRE(size_1 == 24);
    REQUIRE(dim_1 == 4);

    constexpr const auto size_2 = etl::size(test_matrix);
    constexpr const auto rows_2 = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2 = etl::dimensions(test_matrix);

    REQUIRE(size_2 == 24);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 4);

    constexpr const auto rows_3 = etl::etl_traits<type>::dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::dim<1>();
    constexpr const auto dim_2_3 = etl::etl_traits<type>::dim<2>();
    constexpr const auto dim_3_3 = etl::etl_traits<type>::dim<3>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
    REQUIRE(dim_2_3 == 4);
    REQUIRE(dim_3_3 == 1);
}

TEST_CASE( "etl_traits/dyn_vector_1", "etl_traits<dyn_vector>" ) {
    using type = etl::dyn_vector<double>;
    type test_vector(4, 3.3);

    REQUIRE(etl::etl_traits<type>::size(test_vector) == 4);
    REQUIRE(etl::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::dim(test_vector, 0) == 4);
    REQUIRE(etl::etl_traits<type>::dimensions() == 1);
    REQUIRE(etl::dimensions(test_vector) == 1);
    REQUIRE(etl::etl_traits<type>::is_vector);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);
    REQUIRE(!etl::etl_traits<type>::is_matrix);
}

TEST_CASE( "etl_traits/dyn_matrix_1", "etl_traits<dyn_matrix>" ) {
    using type = etl::dyn_matrix<double>;
    type test_matrix(3, 2, 3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(etl::size(test_matrix) == 6);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 2);
    REQUIRE(etl::dimensions(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dim(test_matrix, 0) == 3);
    REQUIRE(etl::etl_traits<type>::dim(test_matrix, 1) == 2);
    REQUIRE(etl::etl_traits<type>::is_matrix);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);
    REQUIRE(!etl::etl_traits<type>::is_vector);
}

TEST_CASE( "etl_traits/unary_dyn_mat", "etl_traits<unary<dyn_mat>>" ) {
    using mat_type = etl::dyn_matrix<double>;
    mat_type test_matrix(3, 2, 3.3);

    using expr_type = decltype(log(test_matrix));
    expr_type expr(test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dim(expr, 0) == 3);
    REQUIRE(etl::etl_traits<expr_type>::dim(expr, 1) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);
}

TEST_CASE( "etl_traits/binary_dyn_mat", "etl_traits<binary<dyn_mat, dyn_mat>>" ) {
    using mat_type = etl::dyn_matrix<double>;
    mat_type test_matrix(3, 2, 3.3);

    using expr_type = decltype(test_matrix + test_matrix);
    expr_type expr(test_matrix, test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dim(expr, 0) == 3);
    REQUIRE(etl::etl_traits<expr_type>::dim(expr, 1) == 2);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);
}

TEST_CASE( "etl_traits/unary_fast_mat", "etl_traits<unary<fast_mat>>" ) {
    using mat_type = etl::fast_matrix<double, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(log(test_matrix));
    expr_type expr(test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1 = etl::etl_traits<expr_type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2 = etl::size(expr);
    constexpr const auto rows_2 = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2 = etl::dimensions(expr);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3 = etl::etl_traits<expr_type>::dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}

TEST_CASE( "etl_traits/binary_fast_mat", "etl_traits<binary<fast_mat, fast_mat>>" ) {
    using mat_type = etl::fast_matrix<double, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(test_matrix + test_matrix);
    expr_type expr(test_matrix, test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::is_matrix);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(etl::etl_traits<expr_type>::is_fast);
    REQUIRE(!etl::etl_traits<expr_type>::is_vector);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1 = etl::etl_traits<expr_type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2 = etl::size(expr);
    constexpr const auto rows_2 = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2 = etl::dimensions(expr);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3 = etl::etl_traits<expr_type>::dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}