//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("etl_traits/fast_vector_1", "etl_traits<fast_vector>", ZZZ, double, float) {
    using type = etl::fast_vector<ZZZ, 4>;
    type test_vector(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::dimensions() == 1);
    REQUIRE(etl::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(etl::etl_traits<type>::vectorizable);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    REQUIRE(size_1 == 4);

    constexpr const auto size_2 = etl::size(test_vector);
    REQUIRE(size_2 == 4);

    constexpr const auto size_3 = etl::etl_traits<type>::template dim<0>();
    REQUIRE(size_3 == 4);

    constexpr const auto size_4 = etl::etl_traits<type>::dimensions();
    REQUIRE(size_4 == 1);

    constexpr const auto size_5 = etl::dimensions(test_vector);
    REQUIRE(size_5 == 1);
}

TEMPLATE_TEST_CASE_2("etl_traits/fast_matrix_1", "etl_traits<fast_matrix>", Z, float, double) {
    using type = etl::fast_matrix<Z, 3, 2>;
    type test_matrix(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(etl::size(test_matrix) == 6);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 2);
    REQUIRE(etl::dimensions(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(etl::etl_traits<type>::vectorizable);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1  = etl::etl_traits<type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2    = etl::size(test_matrix);
    constexpr const auto rows_2    = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2     = etl::dimensions(test_matrix);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3    = etl::etl_traits<type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::template dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}

TEMPLATE_TEST_CASE_2("etl_traits/fast_matrix_2", "etl_traits<fast_matrix>", Z, float, double) {
    using type = etl::fast_matrix<Z, 3, 2, 4, 1>;
    type test_matrix(3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 24);
    REQUIRE(etl::size(test_matrix) == 24);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 4);
    REQUIRE(etl::dimensions(test_matrix) == 4);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);
    REQUIRE(etl::etl_traits<type>::vectorizable);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1  = etl::etl_traits<type>::dimensions();

    REQUIRE(size_1 == 24);
    REQUIRE(dim_1 == 4);

    constexpr const auto size_2    = etl::size(test_matrix);
    constexpr const auto rows_2    = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2     = etl::dimensions(test_matrix);

    REQUIRE(size_2 == 24);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 4);

    constexpr const auto rows_3    = etl::etl_traits<type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::template dim<1>();
    constexpr const auto dim_2_3   = etl::etl_traits<type>::template dim<2>();
    constexpr const auto dim_3_3   = etl::etl_traits<type>::template dim<3>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
    REQUIRE(dim_2_3 == 4);
    REQUIRE(dim_3_3 == 1);
}

TEMPLATE_TEST_CASE_2("etl_traits/dyn_vector_1", "etl_traits<dyn_vector>", Z, float, double) {
    using type = etl::dyn_vector<Z>;
    type test_vector(4, 3.3);

    REQUIRE(etl::etl_traits<type>::size(test_vector) == 4);
    REQUIRE(etl::size(test_vector) == 4);
    REQUIRE(etl::etl_traits<type>::dim(test_vector, 0) == 4);
    REQUIRE(etl::etl_traits<type>::dimensions() == 1);
    REQUIRE(etl::dimensions(test_vector) == 1);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);
    REQUIRE(etl::etl_traits<type>::vectorizable);
}

TEMPLATE_TEST_CASE_2("etl_traits/dyn_matrix_1", "etl_traits<dyn_matrix>", Z, float, double) {
    using type = etl::dyn_matrix<Z>;
    type test_matrix(3, 2, 3.3);

    REQUIRE(etl::etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(etl::size(test_matrix) == 6);
    REQUIRE(etl::rows(test_matrix) == 3);
    REQUIRE(etl::columns(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dimensions() == 2);
    REQUIRE(etl::dimensions(test_matrix) == 2);
    REQUIRE(etl::etl_traits<type>::dim(test_matrix, 0) == 3);
    REQUIRE(etl::etl_traits<type>::dim(test_matrix, 1) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);
    REQUIRE(etl::etl_traits<type>::vectorizable);
}

TEMPLATE_TEST_CASE_2("etl_traits/unary_dyn_mat", "etl_traits<unary<dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
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
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(!etl::etl_traits<expr_type>::is_fast);
}

TEMPLATE_TEST_CASE_2("etl_traits/binary_dyn_mat", "etl_traits<binary<dyn_mat, dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
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
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE(etl::etl_traits<expr_type>::vectorizable);
}

TEMPLATE_TEST_CASE_2("etl_traits/unary_fast_mat", "etl_traits<unary<fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(log(test_matrix));
    expr_type expr(test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(etl::etl_traits<expr_type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1  = etl::etl_traits<expr_type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2    = etl::size(expr);
    constexpr const auto rows_2    = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2     = etl::dimensions(expr);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3    = etl::etl_traits<expr_type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::template dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}

TEMPLATE_TEST_CASE_2("etl_traits/binary_fast_mat", "etl_traits<binary<fast_mat, fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(test_matrix + test_matrix);
    expr_type expr(test_matrix, test_matrix);

    REQUIRE(etl::etl_traits<expr_type>::size(expr) == 6);
    REQUIRE(etl::size(expr) == 6);
    REQUIRE(etl::rows(expr) == 3);
    REQUIRE(etl::columns(expr) == 2);
    REQUIRE(etl::etl_traits<expr_type>::dimensions() == 2);
    REQUIRE(etl::dimensions(expr) == 2);
    REQUIRE(!etl::etl_traits<expr_type>::is_value);
    REQUIRE(etl::etl_traits<expr_type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1  = etl::etl_traits<expr_type>::dimensions();

    REQUIRE(size_1 == 6);
    REQUIRE(dim_1 == 2);

    constexpr const auto size_2    = etl::size(expr);
    constexpr const auto rows_2    = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2     = etl::dimensions(expr);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
    REQUIRE(dim_2 == 2);

    constexpr const auto rows_3    = etl::etl_traits<expr_type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::template dim<1>();

    REQUIRE(rows_3 == 3);
    REQUIRE(columns_3 == 2);
}

TEMPLATE_TEST_CASE_2("etl_traits/has_direct_access", "has_direct_access", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 5);

    //Values have direct access
    REQUIRE(etl::has_direct_access<mat_type_1>::value);
    REQUIRE(etl::has_direct_access<mat_type_2>::value);

    //The type should always be decayed
    REQUIRE(etl::has_direct_access<const mat_type_1&&>::value);
    REQUIRE(etl::has_direct_access<const mat_type_2&&>::value);

    //Values have direct access
    REQUIRE(etl::is_fast_matrix<decltype(a)>::value);
    REQUIRE(etl::has_direct_access<decltype(a)>::value);
    REQUIRE(etl::is_dyn_matrix<decltype(b)>::value);
    REQUIRE(etl::has_direct_access<decltype(b)>::value);

    //Sub have direct access
    REQUIRE(etl::is_unary_expr<decltype(a(0)(1)(3))>::value);
    REQUIRE(etl::has_direct_access<decltype(a(1))>::value);
    REQUIRE(etl::has_direct_access<decltype(b(2))>::value);

    //Sub have direct access
    REQUIRE(etl::is_unary_expr<decltype(a(0)(1)(3))>::value);
    REQUIRE(etl::has_direct_access<decltype(a(0)(1))>::value);
    REQUIRE(etl::has_direct_access<decltype(b(1)(2))>::value);

    //Sub have direct access
    REQUIRE(etl::is_unary_expr<decltype(a(0)(1)(3))>::value);
    REQUIRE(etl::has_direct_access<decltype(a(0)(1)(3))>::value);
    REQUIRE(etl::has_direct_access<decltype(b(1)(2)(0))>::value);

    //View have direct access
    REQUIRE(etl::is_unary_expr<decltype(etl::reshape<4, 30>(a))>::value);
    REQUIRE(etl::has_direct_access<decltype(etl::reshape<4, 30>(a))>::value);
    REQUIRE(etl::has_direct_access<decltype(etl::reshape(b, 3, 40))>::value);

    //Temporary unary expressions have direct access
    REQUIRE(etl::is_temporary_unary_expr<decltype(etl::fft_1d(a(1)(0)(0)))>::value);
    REQUIRE(etl::has_direct_access<decltype(etl::fft_1d(a(1)(0)(0)))>::value);
    REQUIRE(etl::has_direct_access<decltype(etl::fft_1d(b(1)(0)(0)))>::value);

    //Temporary binary expressions have direct access
    REQUIRE(etl::is_temporary_binary_expr<decltype(a(0)(0) * a(0)(0))>::value);
    REQUIRE(etl::has_direct_access<decltype(a(0)(0) * a(0)(0))>::value);
    REQUIRE(etl::has_direct_access<decltype(b(0)(0) * b(0)(0))>::value);

    //Mixes should have direct access even as deep as possible
    REQUIRE(etl::has_direct_access<decltype(etl::reshape<5, 2>(etl::reshape<2, 10>(a(0)(0) * a(0)(0))(1))(0))>::value);
    REQUIRE(etl::has_direct_access<decltype(etl::reshape<5, 2>(etl::reshape<2, 10>(b(0)(0) * b(0)(0))(1))(0))>::value);

    //Binary do not have direct access
    REQUIRE(!etl::has_direct_access<decltype(a + b)>::value);
    REQUIRE(!etl::has_direct_access<decltype(b + b)>::value);

    //Unary do not have direct access
    REQUIRE(!etl::has_direct_access<decltype(abs(a))>::value);
    REQUIRE(!etl::has_direct_access<decltype(abs(b))>::value);
}

TEMPLATE_TEST_CASE_2("etl_traits/vectorizable", "vectorizable", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 5);

    //Values have direct access
    REQUIRE(etl::etl_traits<mat_type_1>::vectorizable);
    REQUIRE(etl::etl_traits<mat_type_2>::vectorizable);

    //Values have direct access
    REQUIRE(etl::etl_traits<decltype(a)>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b)>::vectorizable);

    //Sub have direct access
    REQUIRE(etl::etl_traits<decltype(a(1))>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b(2))>::vectorizable);

    //Sub have direct access
    REQUIRE(etl::etl_traits<decltype(a(0)(1))>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b(1)(2))>::vectorizable);

    //Sub have direct access
    REQUIRE(etl::etl_traits<decltype(a(0)(1)(3))>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b(1)(2)(0))>::vectorizable);

    //Temporary binary expressions have direct access
    REQUIRE(etl::etl_traits<decltype(a(0)(0) * a(0)(0))>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b(0)(0) * b(0)(0))>::vectorizable);

    //Binary do not have direct access
    REQUIRE(etl::etl_traits<decltype(a + b)>::vectorizable);
    REQUIRE(etl::etl_traits<decltype(b + b)>::vectorizable);

    //abs do not have direct access
    REQUIRE(!etl::etl_traits<decltype(abs(a))>::vectorizable);
    REQUIRE(!etl::etl_traits<decltype(abs(b))>::vectorizable);
}

template <typename Z, typename E>
bool correct_type(E&& /*e*/) {
    if (std::is_same<Z, double>::value) {
        return etl::is_double_precision<E>::value;
    } else { //if(std::is_same<Z, float>::value){
        return etl::is_single_precision<E>::value;
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/precision", "is_X_precision", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 5);

    REQUIRE(correct_type<Z>(a));
    REQUIRE(correct_type<Z>(b));

    REQUIRE(correct_type<Z>(a(1)));
    REQUIRE(correct_type<Z>(b(2)));

    REQUIRE(correct_type<Z>(a(0)(1)));
    REQUIRE(correct_type<Z>(b(2)(0)));

    REQUIRE(correct_type<Z>(etl::reshape<4, 30>(a)));
    REQUIRE(correct_type<Z>(etl::reshape(b, 3, 40)));

    REQUIRE(correct_type<Z>(a(0)(0) * a(0)(0)));
    REQUIRE(correct_type<Z>(b(0)(0) * b(0)(0)));

    REQUIRE(correct_type<Z>(etl::reshape<5, 2>(etl::reshape<2, 10>(a(0)(0) * a(0)(0))(1))(0)));
    REQUIRE(correct_type<Z>(etl::reshape<5, 2>(etl::reshape<2, 10>(b(0)(0) * b(0)(0))(1))(0)));

    REQUIRE(correct_type<Z>(a + b));
    REQUIRE(correct_type<Z>(b + a));

    REQUIRE(correct_type<Z>(1.0 + a));
    REQUIRE(correct_type<Z>(b / 1.1));

    REQUIRE(correct_type<Z>(abs(a)));
    REQUIRE(correct_type<Z>(log(b)));
}

TEMPLATE_TEST_CASE_2("etl_traits/temporary", "make_temporary", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 5);

    REQUIRE((std::is_same<mat_type_1, std::decay_t<decltype(make_temporary(a))>>::value));
    REQUIRE((std::is_same<mat_type_2, std::decay_t<decltype(make_temporary(b))>>::value));

    REQUIRE((std::is_same<decltype(a(0)), std::decay_t<decltype(make_temporary(a(1)))>>::value));
    REQUIRE((std::is_same<decltype(b(1)), std::decay_t<decltype(make_temporary(b(0)))>>::value));

    REQUIRE((std::is_same<decltype(a(1)(0)), std::decay_t<decltype(make_temporary(a(1)(1)))>>::value));
    REQUIRE((std::is_same<decltype(b(1)(1)), std::decay_t<decltype(make_temporary(b(0)(1)))>>::value));

    REQUIRE((!std::is_same<decltype(a + a), std::decay_t<decltype(make_temporary(a + a))>>::value));
    REQUIRE((!std::is_same<decltype(b + b), std::decay_t<decltype(make_temporary(b + b))>>::value));
    REQUIRE((!std::is_same<decltype(a + b), std::decay_t<decltype(make_temporary(a + b))>>::value));

    //make_temporary should not affect an ETL value
    REQUIRE(a.memory_start() == make_temporary(a).memory_start());
    REQUIRE(b.memory_start() == make_temporary(b).memory_start());

    //make_temporary should not affect a sub view
    REQUIRE(a(0).memory_start() == make_temporary(a(0)).memory_start());
    REQUIRE(b(0).memory_start() == make_temporary(b(0)).memory_start());

    const auto& c = a(0);
    const auto& d = b(1);

    REQUIRE(c.memory_start() == make_temporary(c).memory_start());
    REQUIRE(d.memory_start() == make_temporary(d).memory_start());
}
