//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("etl_traits/fast_vector_1", "etl_traits<fast_vector>", ZZZ, double, float) {
    using type = etl::fast_vector<ZZZ, 4>;
    type test_vector(3.3);

    REQUIRE_EQUALS(etl::etl_traits<type>::size(test_vector), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dimensions(), 1UL);
    REQUIRE_EQUALS(etl::size(test_vector), 4UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    }

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    REQUIRE_EQUALS(size_1, 4UL);

    constexpr const auto size_2 = etl::size(test_vector);
    REQUIRE_EQUALS(size_2, 4UL);

    constexpr const auto size_3 = etl::etl_traits<type>::template dim<0>();
    REQUIRE_EQUALS(size_3, 4UL);

    constexpr const auto size_4 = etl::etl_traits<type>::dimensions();
    REQUIRE_EQUALS(size_4, 1UL);

    constexpr const auto size_5 = etl::dimensions(test_vector);
    REQUIRE_EQUALS(size_5, 1UL);
}

TEMPLATE_TEST_CASE_2("etl_traits/fast_matrix_1", "etl_traits<fast_matrix>", Z, float, double) {
    using type = etl::fast_matrix<Z, 3, 2>;
    type test_matrix(3.3);

    REQUIRE_EQUALS(etl::etl_traits<type>::size(test_matrix), 6UL);
    REQUIRE_EQUALS(etl::size(test_matrix), 6UL);
    REQUIRE_EQUALS(etl::rows(test_matrix), 3UL);
    REQUIRE_EQUALS(etl::columns(test_matrix), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(test_matrix), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    }

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1  = etl::etl_traits<type>::dimensions();

    REQUIRE_EQUALS(size_1, 6UL);
    REQUIRE_EQUALS(dim_1, 2UL);

    constexpr const auto size_2    = etl::size(test_matrix);
    constexpr const auto rows_2    = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2     = etl::dimensions(test_matrix);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);
    REQUIRE_EQUALS(dim_2, 2UL);

    constexpr const auto rows_3    = etl::etl_traits<type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::template dim<1>();

    REQUIRE_EQUALS(rows_3, 3UL);
    REQUIRE_EQUALS(columns_3, 2UL);
}

TEMPLATE_TEST_CASE_2("etl_traits/fast_matrix_2", "etl_traits<fast_matrix>", Z, float, double) {
    using type = etl::fast_matrix<Z, 3, 2, 4, 1>;
    type test_matrix(3.3);

    REQUIRE_EQUALS(etl::etl_traits<type>::size(test_matrix), 24UL);
    REQUIRE_EQUALS(etl::size(test_matrix), 24UL);
    REQUIRE_EQUALS(etl::rows(test_matrix), 3UL);
    REQUIRE_EQUALS(etl::columns(test_matrix), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dimensions(), 4UL);
    REQUIRE_EQUALS(etl::dimensions(test_matrix), 4UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    }

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    constexpr const auto dim_1  = etl::etl_traits<type>::dimensions();

    REQUIRE_EQUALS(size_1, 24UL);
    REQUIRE_EQUALS(dim_1, 4UL);

    constexpr const auto size_2    = etl::size(test_matrix);
    constexpr const auto rows_2    = etl::rows(test_matrix);
    constexpr const auto columns_2 = etl::columns(test_matrix);
    constexpr const auto dim_2     = etl::dimensions(test_matrix);

    REQUIRE_EQUALS(size_2, 24UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);
    REQUIRE_EQUALS(dim_2, 4UL);

    constexpr const auto rows_3    = etl::etl_traits<type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<type>::template dim<1>();
    constexpr const auto dim_2_3   = etl::etl_traits<type>::template dim<2>();
    constexpr const auto dim_3_3   = etl::etl_traits<type>::template dim<3>();

    REQUIRE_EQUALS(rows_3, 3UL);
    REQUIRE_EQUALS(columns_3, 2UL);
    REQUIRE_EQUALS(dim_2_3, 4UL);
    REQUIRE_EQUALS(dim_3_3, 1UL);
}

TEMPLATE_TEST_CASE_2("etl_traits/dyn_vector_1", "etl_traits<dyn_vector>", Z, float, double) {
    using type = etl::dyn_vector<Z>;
    type test_vector(4, 3.3);

    REQUIRE_EQUALS(etl::etl_traits<type>::size(test_vector), 4UL);
    REQUIRE_EQUALS(etl::size(test_vector), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dim(test_vector, 0), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dimensions(), 1UL);
    REQUIRE_EQUALS(etl::dimensions(test_vector), 1UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/dyn_matrix_1", "etl_traits<dyn_matrix>", Z, float, double) {
    using type = etl::dyn_matrix<Z>;
    type test_matrix(3, 2, 3.3);

    REQUIRE_EQUALS(etl::etl_traits<type>::size(test_matrix), 6UL);
    REQUIRE_EQUALS(etl::size(test_matrix), 6UL);
    REQUIRE_EQUALS(etl::rows(test_matrix), 3UL);
    REQUIRE_EQUALS(etl::columns(test_matrix), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(test_matrix), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dim(test_matrix, 0), 3UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::dim(test_matrix, 1), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<type>::template vectorizable<etl::vector_mode>);
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/unary_dyn_mat", "etl_traits<unary<dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
    mat_type test_matrix(3, 2, 3.3);

    using expr_type = decltype(log(test_matrix));
    expr_type expr(test_matrix);

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 3UL);
    REQUIRE_EQUALS(etl::columns(expr), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dim(expr, 0), 3UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dim(expr, 1), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(expr), 2UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_padded);
}

TEMPLATE_TEST_CASE_2("etl_traits/binary_dyn_mat", "etl_traits<binary<dyn_mat, dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
    mat_type test_matrix(3, 2, 3.3);

    using expr_type = decltype(test_matrix + test_matrix);
    expr_type expr(test_matrix, test_matrix);

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 3UL);
    REQUIRE_EQUALS(etl::columns(expr), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(expr), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dim(expr, 0), 3UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dim(expr, 1), 2UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_padded);

    if(etl::vec_enabled){
        REQUIRE_DIRECT(etl::etl_traits<expr_type>::template vectorizable<etl::vector_mode>);
    } else {
        REQUIRE_DIRECT(!etl::etl_traits<expr_type>::template vectorizable<etl::vector_mode>);
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/unary_fast_mat", "etl_traits<unary<fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(log(test_matrix));
    expr_type expr(test_matrix);

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 3UL);
    REQUIRE_EQUALS(etl::columns(expr), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(expr), 2UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_padded);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1  = etl::etl_traits<expr_type>::dimensions();

    REQUIRE_EQUALS(size_1, 6UL);
    REQUIRE_EQUALS(dim_1, 2UL);

    constexpr const auto size_2    = etl::size(expr);
    constexpr const auto rows_2    = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2     = etl::dimensions(expr);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);
    REQUIRE_EQUALS(dim_2, 2UL);

    constexpr const auto rows_3    = etl::etl_traits<expr_type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::template dim<1>();

    REQUIRE_EQUALS(rows_3, 3UL);
    REQUIRE_EQUALS(columns_3, 2UL);
}

TEMPLATE_TEST_CASE_2("etl_traits/binary_fast_mat", "etl_traits<binary<fast_mat, fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    using expr_type = decltype(test_matrix + test_matrix);
    expr_type expr(test_matrix, test_matrix);

    REQUIRE_EQUALS(etl::etl_traits<expr_type>::size(expr), 6UL);
    REQUIRE_EQUALS(etl::size(expr), 6UL);
    REQUIRE_EQUALS(etl::rows(expr), 3UL);
    REQUIRE_EQUALS(etl::columns(expr), 2UL);
    REQUIRE_EQUALS(etl::etl_traits<expr_type>::dimensions(), 2UL);
    REQUIRE_EQUALS(etl::dimensions(expr), 2UL);
    REQUIRE_DIRECT(!etl::etl_traits<expr_type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<expr_type>::is_padded);

    constexpr const auto size_1 = etl::etl_traits<expr_type>::size();
    constexpr const auto dim_1  = etl::etl_traits<expr_type>::dimensions();

    REQUIRE_EQUALS(size_1, 6UL);
    REQUIRE_EQUALS(dim_1, 2UL);

    constexpr const auto size_2    = etl::size(expr);
    constexpr const auto rows_2    = etl::rows(expr);
    constexpr const auto columns_2 = etl::columns(expr);
    constexpr const auto dim_2     = etl::dimensions(expr);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);
    REQUIRE_EQUALS(dim_2, 2UL);

    constexpr const auto rows_3    = etl::etl_traits<expr_type>::template dim<0>();
    constexpr const auto columns_3 = etl::etl_traits<expr_type>::template dim<1>();

    REQUIRE_EQUALS(rows_3, 3UL);
    REQUIRE_EQUALS(columns_3, 2UL);
}

TEMPLATE_TEST_CASE_2("etl_traits/is_dma", "is_dma", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 4>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 4);

    //Values have direct access
    REQUIRE_DIRECT(etl::is_dma<mat_type_1>);
    REQUIRE_DIRECT(etl::is_dma<mat_type_2>);

    //The type should always be decayed
    REQUIRE_DIRECT(etl::is_dma<const mat_type_1&&>);
    REQUIRE_DIRECT(etl::is_dma<const mat_type_2&&>);

    //Values have direct access
    REQUIRE_DIRECT(etl::is_fast_matrix<decltype(a)>);
    REQUIRE_DIRECT(etl::is_dma<decltype(a)>);
    REQUIRE_DIRECT(etl::is_dyn_matrix<decltype(b)>);
    REQUIRE_DIRECT(etl::is_dma<decltype(b)>);

    //Sub have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(a(1))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(b(2))>);

    //Sub have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(a(0)(1))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(b(1)(2))>);

    //Sub have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(a(0)(1)(3))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(b(1)(2)(0))>);

    //View have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::reshape<6, 16>(a))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::reshape(b, 6, 16))>);

    //Temporary unary expressions have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::fft_1d(a(1)(0)(0)))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::fft_1d(b(1)(0)(0)))>);

    //Temporary binary expressions have direct access
    REQUIRE_DIRECT(etl::is_dma<decltype(a(0)(0) * a(0)(0))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(b(0)(0) * b(0)(0))>);

    //Mixes should have direct access even as deep as possible
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::reshape<4, 2>(etl::reshape<2, 8>(a(0)(0) * a(0)(0))(1))(0))>);
    REQUIRE_DIRECT(etl::is_dma<decltype(etl::reshape<4, 2>(etl::reshape<2, 8>(b(0)(0) * b(0)(0))(1))(0))>);

    //Binary do not have direct access
    REQUIRE_DIRECT(!etl::is_dma<decltype(a + b)>);
    REQUIRE_DIRECT(!etl::is_dma<decltype(b + b)>);

    //Unary do not have direct access
    REQUIRE_DIRECT(!etl::is_dma<decltype(abs(a))>);
    REQUIRE_DIRECT(!etl::is_dma<decltype(abs(b))>);

    if (etl::vec_enabled) {
        //Some unary can be vectorizable
        REQUIRE_DIRECT((etl::all_vectorizable<etl::vector_mode, decltype(abs(a))>));
        REQUIRE_DIRECT((etl::all_vectorizable<etl::vector_mode, decltype(abs(b))>));
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/vectorizable", "vectorizable", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 4>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 4);

    if(etl::vec_enabled){
        //Values have direct access
        REQUIRE_DIRECT(etl::etl_traits<mat_type_1>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<mat_type_2>::template vectorizable<etl::vector_mode>);

        //Values have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a)>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b)>::template vectorizable<etl::vector_mode>);

        //Sub have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(1))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b(2))>::template vectorizable<etl::vector_mode>);

        //Sub have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(0)(1))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b(1)(2))>::template vectorizable<etl::vector_mode>);

        //Sub have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(0)(1)(3))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b(1)(2)(0))>::template vectorizable<etl::vector_mode>);

        //Sub have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(1) + a(0))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(0) + a(1))>::template vectorizable<etl::vector_mode>);

        //reshape have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::reshape<8, 5>(a(1) + a(0)))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::reshape<5>(a(1)(0) + a(0)(1)))>::template vectorizable<etl::vector_mode>);

        //reshape have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::reshape(a(1) + a(0), 5, 8))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::reshape(a(1)(0) + a(0)(1), 5))>::template vectorizable<etl::vector_mode>);

        //sub have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::sub(a(1) + a(0), 0))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(etl::sub(a(1)(0) + a(0)(1), 1))>::template vectorizable<etl::vector_mode>);

        //Temporary binary expressions have direct access
        REQUIRE_DIRECT(etl::etl_traits<decltype(a(0)(0) * a(0)(0))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b(0)(0) * b(0)(0))>::template vectorizable<etl::vector_mode>);

        //Binary is vectorizable
        REQUIRE_DIRECT(etl::etl_traits<decltype(a + b)>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(b + b)>::template vectorizable<etl::vector_mode>);

        //abs is vectorizable
        REQUIRE_DIRECT(etl::etl_traits<decltype(abs(a))>::template vectorizable<etl::vector_mode>);
        REQUIRE_DIRECT(etl::etl_traits<decltype(abs(b))>::template vectorizable<etl::vector_mode>);
    }
}

template <typename Z, typename E>
bool correct_type(E&& /*e*/) {
    if (std::is_same<Z, double>::value) {
        return etl::is_double_precision<E>;
    } else { //if(std::is_same<Z, float>::value){
        return etl::is_single_precision<E>;
    }
}

TEMPLATE_TEST_CASE_2("etl_traits/precision", "is_X_precision", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 4>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 4);

    REQUIRE_DIRECT(correct_type<Z>(a));
    REQUIRE_DIRECT(correct_type<Z>(b));

    REQUIRE_DIRECT(correct_type<Z>(a(1)));
    REQUIRE_DIRECT(correct_type<Z>(b(2)));

    REQUIRE_DIRECT(correct_type<Z>(a(0)(1)));
    REQUIRE_DIRECT(correct_type<Z>(b(2)(0)));

    REQUIRE_DIRECT(correct_type<Z>(etl::reshape<6, 16>(a)));
    REQUIRE_DIRECT(correct_type<Z>(etl::reshape(b, 6, 16)));

    REQUIRE_DIRECT(correct_type<Z>(a(0)(0) * a(0)(0)));
    REQUIRE_DIRECT(correct_type<Z>(b(0)(0) * b(0)(0)));

    REQUIRE_DIRECT(correct_type<Z>(etl::reshape(etl::reshape(a(0)(0) * a(0)(0), 2, 8)(1), 4, 2)(0)));
    REQUIRE_DIRECT(correct_type<Z>(etl::reshape(etl::reshape(b(0)(0) * b(0)(0), 2, 8)(1), 4, 2)(0)));

    REQUIRE_DIRECT(correct_type<Z>(etl::reshape<4, 2>(etl::reshape<2, 8>(a(0)(0) * a(0)(0))(1))(0)));
    REQUIRE_DIRECT(correct_type<Z>(etl::reshape<4, 2>(etl::reshape<2, 8>(b(0)(0) * b(0)(0))(1))(0)));

    REQUIRE_DIRECT(correct_type<Z>(a + b));
    REQUIRE_DIRECT(correct_type<Z>(b + a));

    REQUIRE_DIRECT(correct_type<Z>(1.0 + a));
    REQUIRE_DIRECT(correct_type<Z>(b / 1.1));

    REQUIRE_DIRECT(correct_type<Z>(abs(a)));
    REQUIRE_DIRECT(correct_type<Z>(log(b)));
}

TEMPLATE_TEST_CASE_2("etl_traits/temporary", "make_temporary", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    mat_type_1 a(3.3);

    using mat_type_2 = etl::dyn_matrix<Z, 4>;
    mat_type_2 b(3, 2, 4, 5);

    REQUIRE_DIRECT((std::is_same<mat_type_1, std::decay_t<decltype(make_temporary(a))>>::value));
    REQUIRE_DIRECT((std::is_same<mat_type_2, std::decay_t<decltype(make_temporary(b))>>::value));

    REQUIRE_DIRECT((std::is_same<decltype(a(0)), std::decay_t<decltype(make_temporary(a(1)))>>::value));
    REQUIRE_DIRECT((std::is_same<decltype(b(1)), std::decay_t<decltype(make_temporary(b(0)))>>::value));

    REQUIRE_DIRECT((std::is_same<decltype(a(1)(0)), std::decay_t<decltype(make_temporary(a(1)(1)))>>::value));
    REQUIRE_DIRECT((std::is_same<decltype(b(1)(1)), std::decay_t<decltype(make_temporary(b(0)(1)))>>::value));

    REQUIRE_DIRECT((!std::is_same<decltype(a + a), std::decay_t<decltype(make_temporary(a + a))>>::value));
    REQUIRE_DIRECT((!std::is_same<decltype(b + b), std::decay_t<decltype(make_temporary(b + b))>>::value));
    REQUIRE_DIRECT((!std::is_same<decltype(a + b), std::decay_t<decltype(make_temporary(a + b))>>::value));

    //make_temporary should not affect an ETL value
    REQUIRE_EQUALS(a.memory_start(), make_temporary(a).memory_start());
    REQUIRE_EQUALS(b.memory_start(), make_temporary(b).memory_start());

    //make_temporary should not affect a sub view
    REQUIRE_EQUALS(a(0).memory_start(), make_temporary(a(0)).memory_start());
    REQUIRE_EQUALS(b(0).memory_start(), make_temporary(b(0)).memory_start());

    const auto& c = a(0);
    const auto& d = b(1);

    REQUIRE_EQUALS(c.memory_start(), make_temporary(c).memory_start());
    REQUIRE_EQUALS(d.memory_start(), make_temporary(d).memory_start());
}

TEMPLATE_TEST_CASE_2("etl_traits/inplace_transpose_able", "inplace_transpose_able", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 2, 4, 5>;
    using mat_type_2 = etl::fast_matrix<Z, 3, 2>;
    using mat_type_3 = etl::fast_matrix<Z, 3, 3>;
    using mat_type_4 = etl::fast_matrix<Z, 2, 3>;
    using mat_type_5 = etl::dyn_matrix<Z, 4>;
    using mat_type_6 = etl::dyn_matrix<Z, 2>;

    REQUIRE_DIRECT(!etl::inplace_transpose_able<mat_type_1>);
    REQUIRE_DIRECT(!etl::inplace_transpose_able<mat_type_2>);
    REQUIRE_DIRECT(etl::inplace_transpose_able<mat_type_3>);
    REQUIRE_DIRECT(!etl::inplace_transpose_able<mat_type_4>);
    REQUIRE_DIRECT(!etl::inplace_transpose_able<mat_type_5>);
    REQUIRE_DIRECT(etl::inplace_transpose_able<mat_type_6>);
}

TEMPLATE_TEST_CASE_2("etl_traits/selected_expr", "[traits]", Z, float, double) {
    using mat_type_1 = etl::fast_matrix<Z, 3, 3>;
    mat_type_1 a;

    using mat_type_2 = etl::dyn_matrix<Z, 2>;
    mat_type_2 b(3, 3);

    using selected_type_1 = decltype(etl::selected<etl::gemm_impl, etl::gemm_impl::STD>(a * b));
    using selected_type_2 = decltype(selected_helper(etl::gemm_impl::STD, a * b));

    REQUIRE_DIRECT(etl::is_selected_expr<selected_type_1>);
    REQUIRE_DIRECT(etl::is_selected_expr<selected_type_1>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<selected_type_2>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<selected_type_2>);

    REQUIRE_DIRECT(etl::is_selected_expr<const selected_type_1>);
    REQUIRE_DIRECT(etl::is_selected_expr<const selected_type_1>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<const selected_type_2>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<const selected_type_2>);

    REQUIRE_DIRECT(etl::is_selected_expr<selected_type_1&>);
    REQUIRE_DIRECT(etl::is_selected_expr<selected_type_1&>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<selected_type_2&>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<selected_type_2&>);

    REQUIRE_DIRECT(etl::is_selected_expr<const selected_type_1&>);
    REQUIRE_DIRECT(etl::is_selected_expr<const selected_type_1&>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<const selected_type_2&>);
    REQUIRE_DIRECT(etl::is_wrapper_expr<const selected_type_2&>);
}

TEST_CASE("etl_traits/vectorizable_bool", "[traits]") {
    using mat_type_1 = etl::fast_matrix<bool, 3, 3>;
    mat_type_1 a;

    using mat_type_2 = etl::dyn_matrix<bool, 2>;
    mat_type_2 b(3, 3);

    using expr_type_1 = decltype(a + b);

    if(etl::sse3_enabled){
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT(!(etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if(etl::avx_enabled){
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/vectorizable_short", "[traits]") {
    using mat_type_1 = etl::fast_matrix<int16_t, 3, 3>;
    mat_type_1 a;

    using mat_type_2 = etl::dyn_matrix<int16_t, 2>;
    mat_type_2 b(3, 3);

    using expr_type_1 = decltype(a + b);

    if (etl::sse3_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if(etl::avx2_enabled){
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    } else {
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/vectorizable_integer", "[traits]") {
    using mat_type_1 = etl::fast_matrix<int, 3, 3>;
    mat_type_1 a;

    using mat_type_2 = etl::dyn_matrix<int, 2>;
    mat_type_2 b(3, 3);

    using expr_type_1 = decltype(a + b);

    if (etl::sse3_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if(etl::avx2_enabled){
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    } else {
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/vectorizable_long", "[traits]") {
    using mat_type_1 = etl::fast_matrix<long, 3, 3>;
    mat_type_1 a;

    using mat_type_2 = etl::dyn_matrix<long, 2>;
    mat_type_2 b(3, 3);

    using expr_type_1 = decltype(a + b);

    if (etl::sse3_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if (etl::avx2_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    } else {
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<mat_type_2>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT(!(etl::decay_traits<expr_type_1>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/vectorize/expr/1", "[traits]") {
    etl::fast_matrix<float, 3, 3> A;
    etl::fast_matrix<float, 3, 3> B;
    etl::fast_matrix<float, 3, 3> C;

    auto expr_0 = sigmoid(A * B);
    auto expr_1 = sigmoid(A * B + C);
    auto expr_2 = sigmoid(C + A * B);

    using expr_0_t = decltype(expr_0);
    using expr_1_t = decltype(expr_1);
    using expr_2_t = decltype(expr_2);

    if (etl::sse3_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<expr_0_t>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_1_t>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_2_t>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if (etl::avx_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<expr_0_t>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_1_t>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_2_t>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/vectorize/expr/2", "[traits]") {
    etl::dyn_matrix<float, 2> A(3, 3);
    etl::dyn_matrix<float, 2> B(3, 3);
    etl::dyn_matrix<float, 2> C(3, 3);

    auto expr_0 = sigmoid(A * B);
    auto expr_1 = sigmoid(A * B + C);
    auto expr_2 = sigmoid(C + A * B);

    using expr_0_t = decltype(expr_0);
    using expr_1_t = decltype(expr_1);
    using expr_2_t = decltype(expr_2);

    if (etl::sse3_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<expr_0_t>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_1_t>::template vectorizable<etl::vector_mode_t::SSE3>));
        REQUIRE_DIRECT((etl::decay_traits<expr_2_t>::template vectorizable<etl::vector_mode_t::SSE3>));
    }

    if (etl::avx_enabled) {
        REQUIRE_DIRECT((etl::decay_traits<expr_0_t>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_1_t>::template vectorizable<etl::vector_mode_t::AVX>));
        REQUIRE_DIRECT((etl::decay_traits<expr_2_t>::template vectorizable<etl::vector_mode_t::AVX>));
    }
}

TEST_CASE("etl_traits/is_temporary_expr", "[traits]") {
    etl::dyn_matrix<float, 2> A(3, 3);

    REQUIRE_DIRECT(!etl::is_temporary_expr<decltype(A)>);
    REQUIRE_DIRECT(!etl::is_temporary_expr<decltype(A + A)>);
    REQUIRE_DIRECT(etl::is_temporary_expr<decltype(A * A)>);
    REQUIRE_DIRECT(etl::is_temporary_expr<decltype(trans(A))>);
    REQUIRE_DIRECT(etl::is_temporary_expr<decltype(trans(A) * A)>);
}
