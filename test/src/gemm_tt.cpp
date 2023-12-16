//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Matrix Matrix multiplication tests

GEMM_TT_TEST_CASE("gemm_tt/1", "[gemm]") {
    etl::fast_matrix<T, 2, 3> aa = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<T, 3, 2> a;
    etl::fast_matrix<T, 3, 2> bb = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 3> b;

    a = transpose(aa);
    b = transpose(bb);

    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
}

GEMM_TT_TEST_CASE("gemm_tt/2", "[gemm]") {
    etl::fast_matrix<T, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<T, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};
    etl::fast_matrix<T, 3, 3> c;

    a = transpose(a);
    b = transpose(b);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(0, 2), 70);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
    REQUIRE_EQUALS(c(1, 2), 169);
    REQUIRE_EQUALS(c(2, 0), 220);
    REQUIRE_EQUALS(c(2, 1), 244);
    REQUIRE_EQUALS(c(2, 2), 268);
}

GEMM_TT_TEST_CASE("gemm_tt/3", "[gemm]") {
    etl::dyn_matrix<T> a(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<T> b(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<T> c(4, 4);

    a = transpose(a);
    b = transpose(b);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 90);
    REQUIRE_EQUALS(c(0, 1), 100);
    REQUIRE_EQUALS(c(1, 0), 202);
    REQUIRE_EQUALS(c(1, 1), 228);
    REQUIRE_EQUALS(c(2, 0), 314);
    REQUIRE_EQUALS(c(2, 1), 356);
    REQUIRE_EQUALS(c(3, 0), 426);
    REQUIRE_EQUALS(c(3, 1), 484);
}

GEMM_TT_TEST_CASE("gemm_tt/4", "[gemm]") {
    etl::dyn_matrix<T> a(2, 2, etl::values(1, 2, 3, 4));
    etl::dyn_matrix<T> b(2, 2, etl::values(1, 2, 3, 4));
    etl::dyn_matrix<T> c(2, 2);

    a = transpose(a);
    b = transpose(b);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 7);
    REQUIRE_EQUALS(c(0, 1), 10);
    REQUIRE_EQUALS(c(1, 0), 15);
    REQUIRE_EQUALS(c(1, 1), 22);
}

GEMM_TT_TEST_CASE("gemm_tt/5", "[gemm]") {
    etl::dyn_matrix<T> a(3, 3, std::initializer_list<T>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    etl::dyn_matrix<T> b(3, 3, std::initializer_list<T>({7, 8, 9, 9, 10, 11, 11, 12, 13}));
    etl::dyn_matrix<T> c(3, 3);

    a = transpose(a);
    b = transpose(b);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(0, 2), 70);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
    REQUIRE_EQUALS(c(1, 2), 169);
    REQUIRE_EQUALS(c(2, 0), 220);
    REQUIRE_EQUALS(c(2, 1), 244);
    REQUIRE_EQUALS(c(2, 2), 268);
}

GEMM_TT_TEST_CASE("gemm_tt/6", "[gemm]") {
    etl::fast_matrix<T, 19, 19> a;
    etl::fast_matrix<T, 19, 19> b;
    etl::fast_matrix<T, 19, 19> c;

    a = etl::magic(19);
    b = etl::magic(19);

    a = transpose(a);
    b = transpose(b);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 828343);
    REQUIRE_EQUALS(c(1, 1), 825360);
    REQUIRE_EQUALS(c(2, 2), 826253);
    REQUIRE_EQUALS(c(3, 3), 824524);
    REQUIRE_EQUALS(c(18, 18), 828343);
}

GEMM_TT_TEST_CASE("gemm_tt/7", "[gemm]") {
    etl::dyn_matrix<T> a(84, 84);
    etl::dyn_matrix<T> b(84, 84);
    etl::dyn_matrix<T> c(84, 84);
    etl::dyn_matrix<T> r(84, 84);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < rows(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(j, k);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}

GEMM_TT_TEST_CASE("gemm_tt/8", "[gemm]") {
    etl::dyn_matrix<T> a(96, 84);
    etl::dyn_matrix<T> b(84, 96);
    etl::dyn_matrix<T> c(84, 84);
    etl::dyn_matrix<T> r(84, 84);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < rows(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(j, k);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}

GEMM_TT_TEST_CASE("gemm_tt/9", "[gemm]") {
    etl::dyn_matrix<T> a(84, 128);
    etl::dyn_matrix<T> b(96, 84);
    etl::dyn_matrix<T> c(128, 96);
    etl::dyn_matrix<T> r(128, 96);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < rows(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(j, k);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}
