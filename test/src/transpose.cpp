//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "transpose_test.hpp"

TRANSPOSE_TEST_CASE("transpose/fast_matrix_1", "transpose") {
    etl::fast_matrix<T, 3, 2> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<T, 2, 3> b;

    Impl::apply(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);
    REQUIRE_EQUALS(b(0, 2), 0.0);
    REQUIRE_EQUALS(b(1, 0), -2.0);
    REQUIRE_EQUALS(b(1, 1), 0.5);
    REQUIRE_EQUALS(b(1, 2), -1);
}

TRANSPOSE_TEST_CASE("transpose/fast_matrix_2", "transpose") {
    etl::fast_matrix<T, 2, 3> a({1.0, -2.0, 3.0, 0.5, 0.0, -1});
    etl::fast_matrix<T, 3, 2> b;

    Impl::apply(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 0.5);
    REQUIRE_EQUALS(b(1, 0), -2.0);
    REQUIRE_EQUALS(b(1, 1), 0.0);
    REQUIRE_EQUALS(b(2, 0), 3.0);
    REQUIRE_EQUALS(b(2, 1), -1);
}

INPLACE_TRANSPOSE_TEST_CASE("transpose/inplace/1", "[transpose]") {
    etl::fast_matrix<T, 3, 3> a({1, 2, 3, 4, 5, 6, 7, 8, 9});

    Impl::apply(a);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 2), 8.0);
    REQUIRE_EQUALS(a(2, 0), 3.0);
    REQUIRE_EQUALS(a(2, 1), 6.0);
    REQUIRE_EQUALS(a(2, 2), 9.0);
}

TRANSPOSE_TEST_CASE("transpose/dyn_matrix_1", "transpose") {
    etl::dyn_matrix<T> a(3, 2, std::initializer_list<T>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<T> b;

    Impl::apply(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 3.0);
    REQUIRE_EQUALS(b(0, 2), 0.0);
    REQUIRE_EQUALS(b(1, 0), -2.0);
    REQUIRE_EQUALS(b(1, 1), 0.5);
    REQUIRE_EQUALS(b(1, 2), -1);
}

TRANSPOSE_TEST_CASE("transpose/dyn_matrix_2", "transpose") {
    etl::dyn_matrix<T> a(2, 3, std::initializer_list<T>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));
    etl::dyn_matrix<T> b;

    Impl::apply(a, b);

    REQUIRE_EQUALS(b(0, 0), 1.0);
    REQUIRE_EQUALS(b(0, 1), 0.5);
    REQUIRE_EQUALS(b(1, 0), -2.0);
    REQUIRE_EQUALS(b(1, 1), 0.0);
    REQUIRE_EQUALS(b(2, 0), 3.0);
    REQUIRE_EQUALS(b(2, 1), -1);
}

TRANSPOSE_TEST_CASE("transpose/dyn_matrix_3", "transpose") {
    etl::dyn_matrix<T> a(99, 33);
    etl::dyn_matrix<T> b;
    etl::dyn_matrix<T> c(33, 99);

    Impl::apply(a, b);

    for (size_t j = 0; j < 99; ++j) {
        for (size_t i = 0; i < 33; ++i) {
            c(i, j) = a(j, i);
        }
    }

    REQUIRE(b == c);
}

TRANSPOSE_TEST_CASE("transpose/dyn_matrix_4", "transpose") {
    const size_t N = 8 * 16 + 4 + 3;
    const size_t M = 9 * 16 + 4 + 2;

    etl::dyn_matrix<T> a(N, M);
    etl::dyn_matrix<T> b;
    etl::dyn_matrix<T> c(M, N);

    Impl::apply(a, b);

    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            c(i, j) = a(j, i);
        }
    }

    REQUIRE(b == c);
}

INPLACE_TRANSPOSE_TEST_CASE("transpose/inplace/2", "[transpose]") {
    etl::dyn_matrix<T> a(3, 3, std::initializer_list<T>({1, 2, 3, 4, 5, 6, 7, 8, 9}));

    Impl::apply(a);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 2), 8.0);
    REQUIRE_EQUALS(a(2, 0), 3.0);
    REQUIRE_EQUALS(a(2, 1), 6.0);
    REQUIRE_EQUALS(a(2, 2), 9.0);
}

INPLACE_TRANSPOSE_TEST_CASE("transpose/inplace/3", "[transpose]") {
    etl::dyn_matrix<T> a(3, 2, std::initializer_list<T>({1.0, -2.0, 3.0, 0.5, 0.0, -1}));

    Impl::apply(a);

    REQUIRE_EQUALS(etl::dim<0>(a), 2UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 3UL);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 3.0);
    REQUIRE_EQUALS(a(0, 2), 0.0);

    REQUIRE_EQUALS(a(1, 0), -2.0);
    REQUIRE_EQUALS(a(1, 1), 0.5);
    REQUIRE_EQUALS(a(1, 2), -1.0);

    Impl::apply(a);

    REQUIRE_EQUALS(etl::dim<0>(a), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 2UL);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), -2.0);

    REQUIRE_EQUALS(a(1, 0), 3.0);
    REQUIRE_EQUALS(a(1, 1), 0.5);

    REQUIRE_EQUALS(a(2, 0), 0.0);
    REQUIRE_EQUALS(a(2, 1), -1.0);
}

INPLACE_TRANSPOSE_TEST_CASE("transpose/inplace/4", "transpose") {
    etl::dyn_matrix<T> a(5, 3, std::initializer_list<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));

    Impl::apply(a);

    REQUIRE_EQUALS(etl::dim<0>(a), 3UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 5UL);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 2), 7.0);
    REQUIRE_EQUALS(a(0, 3), 10.0);
    REQUIRE_EQUALS(a(0, 4), 13.0);

    REQUIRE_EQUALS(a(1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 3), 11.0);
    REQUIRE_EQUALS(a(1, 4), 14.0);

    REQUIRE_EQUALS(a(2, 0), 3.0);
    REQUIRE_EQUALS(a(2, 1), 6.0);
    REQUIRE_EQUALS(a(2, 2), 9.0);
    REQUIRE_EQUALS(a(2, 3), 12.0);
    REQUIRE_EQUALS(a(2, 4), 15.0);

    Impl::apply(a);

    REQUIRE_EQUALS(etl::dim<0>(a), 5UL);
    REQUIRE_EQUALS(etl::dim<1>(a), 3UL);

    REQUIRE_EQUALS(a(0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 1), 2.0);
    REQUIRE_EQUALS(a(0, 2), 3.0);

    REQUIRE_EQUALS(a(1, 0), 4.0);
    REQUIRE_EQUALS(a(1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 2), 6.0);

    REQUIRE_EQUALS(a(2, 0), 7.0);
    REQUIRE_EQUALS(a(2, 1), 8.0);
    REQUIRE_EQUALS(a(2, 2), 9.0);

    REQUIRE_EQUALS(a(3, 0), 10.0);
    REQUIRE_EQUALS(a(3, 1), 11.0);
    REQUIRE_EQUALS(a(3, 2), 12.0);

    REQUIRE_EQUALS(a(4, 0), 13.0);
    REQUIRE_EQUALS(a(4, 1), 14.0);
    REQUIRE_EQUALS(a(4, 2), 15.0);
}

TEMPLATE_TEST_CASE_2("transpose/expr_1", "transpose", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(3, 3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

    a(1).transpose_inplace();

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 2.0);
    REQUIRE_EQUALS(a(0, 0, 2), 3.0);
    REQUIRE_EQUALS(a(0, 1, 0), 4.0);
    REQUIRE_EQUALS(a(0, 1, 1), 5.0);
    REQUIRE_EQUALS(a(0, 1, 2), 6.0);
    REQUIRE_EQUALS(a(0, 2, 0), 7.0);
    REQUIRE_EQUALS(a(0, 2, 1), 8.0);
    REQUIRE_EQUALS(a(0, 2, 2), 9.0);

    REQUIRE_EQUALS(a(1, 0, 0), 1.0);
    REQUIRE_EQUALS(a(1, 0, 1), 4.0);
    REQUIRE_EQUALS(a(1, 0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 2, 0), 3.0);
    REQUIRE_EQUALS(a(1, 2, 1), 6.0);
    REQUIRE_EQUALS(a(1, 2, 2), 9.0);

    REQUIRE_EQUALS(a(2, 0, 0), 1.0);
    REQUIRE_EQUALS(a(2, 0, 1), 2.0);
    REQUIRE_EQUALS(a(2, 0, 2), 3.0);
    REQUIRE_EQUALS(a(2, 1, 0), 4.0);
    REQUIRE_EQUALS(a(2, 1, 1), 5.0);
    REQUIRE_EQUALS(a(2, 1, 2), 6.0);
    REQUIRE_EQUALS(a(2, 2, 0), 7.0);
    REQUIRE_EQUALS(a(2, 2, 1), 8.0);
    REQUIRE_EQUALS(a(2, 2, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/1", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 3> a(2, 3, 3, etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9));

    a.deep_transpose_inplace();

    REQUIRE_EQUALS(a(0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 0, 2), 7.0);
    REQUIRE_EQUALS(a(0, 1, 0), 2.0);
    REQUIRE_EQUALS(a(0, 1, 1), 5.0);
    REQUIRE_EQUALS(a(0, 1, 2), 8.0);
    REQUIRE_EQUALS(a(0, 2, 0), 3.0);
    REQUIRE_EQUALS(a(0, 2, 1), 6.0);
    REQUIRE_EQUALS(a(0, 2, 2), 9.0);

    REQUIRE_EQUALS(a(1, 0, 0), 1.0);
    REQUIRE_EQUALS(a(1, 0, 1), 4.0);
    REQUIRE_EQUALS(a(1, 0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 2, 0), 3.0);
    REQUIRE_EQUALS(a(1, 2, 1), 6.0);
    REQUIRE_EQUALS(a(1, 2, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/2", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 4> a(2, 2, 3, 3, etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9));

    a.deep_transpose_inplace();

    REQUIRE_EQUALS(a(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 0, 0, 2), 7.0);
    REQUIRE_EQUALS(a(0, 0, 1, 0), 2.0);
    REQUIRE_EQUALS(a(0, 0, 1, 1), 5.0);
    REQUIRE_EQUALS(a(0, 0, 1, 2), 8.0);
    REQUIRE_EQUALS(a(0, 0, 2, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2, 1), 6.0);
    REQUIRE_EQUALS(a(0, 0, 2, 2), 9.0);

    REQUIRE_EQUALS(a(1, 1, 0, 0), 1.0);
    REQUIRE_EQUALS(a(1, 1, 0, 1), 4.0);
    REQUIRE_EQUALS(a(1, 1, 0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 1, 1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1, 1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 1, 1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 1, 2, 0), 3.0);
    REQUIRE_EQUALS(a(1, 1, 2, 1), 6.0);
    REQUIRE_EQUALS(a(1, 1, 2, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/3", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 4> a(2, 2, 3, 3,
        etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9,
                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                       1, 2, 3, 4, 5, 6, 7, 8, 9,
                       1, 2, 3, 4, 5, 6, 7, 8, 9));

    a(0).deep_transpose_inplace();
    a(1).deep_transpose_inplace();

    REQUIRE_EQUALS(a.dim(0), 2UL);
    REQUIRE_EQUALS(a.dim(1), 2UL);
    REQUIRE_EQUALS(a.dim(2), 3UL);
    REQUIRE_EQUALS(a.dim(3), 3UL);

    REQUIRE_EQUALS(a(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 0, 0, 2), 7.0);
    REQUIRE_EQUALS(a(0, 0, 1, 0), 2.0);
    REQUIRE_EQUALS(a(0, 0, 1, 1), 5.0);
    REQUIRE_EQUALS(a(0, 0, 1, 2), 8.0);
    REQUIRE_EQUALS(a(0, 0, 2, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2, 1), 6.0);
    REQUIRE_EQUALS(a(0, 0, 2, 2), 9.0);

    REQUIRE_EQUALS(a(1, 1, 0, 0), 1.0);
    REQUIRE_EQUALS(a(1, 1, 0, 1), 4.0);
    REQUIRE_EQUALS(a(1, 1, 0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 1, 1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1, 1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 1, 1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 1, 2, 0), 3.0);
    REQUIRE_EQUALS(a(1, 1, 2, 1), 6.0);
    REQUIRE_EQUALS(a(1, 1, 2, 2), 9.0);
}

TEMPLATE_TEST_CASE_2("deep_transpose/4", "[dyn][trans]", Z, float, double) {
    etl::dyn_matrix<Z, 4> a(2, 2, 4, 3,
        etl::values<Z>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));

    a.deep_transpose_inplace();

    REQUIRE_EQUALS(a.dim(0), 2UL);
    REQUIRE_EQUALS(a.dim(1), 2UL);
    REQUIRE_EQUALS(a.dim(2), 3UL);
    REQUIRE_EQUALS(a.dim(3), 4UL);

    REQUIRE_EQUALS(a(0, 0, 0, 0), 1.0);
    REQUIRE_EQUALS(a(0, 0, 0, 1), 4.0);
    REQUIRE_EQUALS(a(0, 0, 0, 2), 7.0);
    REQUIRE_EQUALS(a(0, 0, 0, 3), 10.0);
    REQUIRE_EQUALS(a(0, 0, 1, 0), 2.0);
    REQUIRE_EQUALS(a(0, 0, 1, 1), 5.0);
    REQUIRE_EQUALS(a(0, 0, 1, 2), 8.0);
    REQUIRE_EQUALS(a(0, 0, 1, 3), 11.0);
    REQUIRE_EQUALS(a(0, 0, 2, 0), 3.0);
    REQUIRE_EQUALS(a(0, 0, 2, 1), 6.0);
    REQUIRE_EQUALS(a(0, 0, 2, 2), 9.0);
    REQUIRE_EQUALS(a(0, 0, 2, 3), 12.0);

    REQUIRE_EQUALS(a(1, 1, 0, 0), 1.0);
    REQUIRE_EQUALS(a(1, 1, 0, 1), 4.0);
    REQUIRE_EQUALS(a(1, 1, 0, 2), 7.0);
    REQUIRE_EQUALS(a(1, 1, 0, 3), 10.0);
    REQUIRE_EQUALS(a(1, 1, 1, 0), 2.0);
    REQUIRE_EQUALS(a(1, 1, 1, 1), 5.0);
    REQUIRE_EQUALS(a(1, 1, 1, 2), 8.0);
    REQUIRE_EQUALS(a(1, 1, 1, 3), 11.0);
    REQUIRE_EQUALS(a(1, 1, 2, 0), 3.0);
    REQUIRE_EQUALS(a(1, 1, 2, 1), 6.0);
    REQUIRE_EQUALS(a(1, 1, 2, 2), 9.0);
    REQUIRE_EQUALS(a(1, 1, 2, 3), 12.0);
}
