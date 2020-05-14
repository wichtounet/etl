//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Matrix Matrix multiplication tests

GEMM_TN_TEST_CASE("gemm_tn/cm/1", "[gemm_tn]") {
    etl::fast_matrix_cm<T, 2, 3> aa = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix_cm<T, 3, 2> b = {7, 9, 11, 8, 10, 12};

    etl::fast_matrix_cm<T, 3, 2> a;
    etl::fast_matrix_cm<T, 2, 2> c;

    a = transpose(aa);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
}

GEMM_TN_TEST_CASE("gemm_tn/cm/2", "[gemm_tn]") {
    etl::fast_matrix_cm<T, 3, 3> a = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    etl::fast_matrix_cm<T, 3, 3> b = {7, 9, 11, 8, 10, 12, 9, 11, 13};
    etl::fast_matrix_cm<T, 3, 3> c;

    a = transpose(a);

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

GEMM_TN_TEST_CASE("gemm_tn/cm/3", "[gemm_tn]") {
    etl::dyn_matrix_cm<T> a(4, 4, etl::values(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16));
    etl::dyn_matrix_cm<T> b(4, 4, etl::values(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16));
    etl::dyn_matrix_cm<T> c(4, 4);

    a = transpose(a);

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

GEMM_TN_TEST_CASE("gemm_tn/cm/4", "[gemm_tn]") {
    etl::dyn_matrix_cm<T> a(2, 2, etl::values(1, 3, 2, 4));
    etl::dyn_matrix_cm<T> b(2, 2, etl::values(1, 3, 2, 4));
    etl::dyn_matrix_cm<T> c(2, 2);

    a = transpose(a);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 7);
    REQUIRE_EQUALS(c(0, 1), 10);
    REQUIRE_EQUALS(c(1, 0), 15);
    REQUIRE_EQUALS(c(1, 1), 22);
}

GEMM_TN_TEST_CASE("gemm_tn/cm/5", "[gemm_tn]") {
    etl::dyn_matrix_cm<T> a(3, 3, std::initializer_list<T>({1, 4, 7, 2, 5, 8, 3, 6, 9}));
    etl::dyn_matrix_cm<T> b(3, 3, std::initializer_list<T>({7, 9, 11, 8, 10, 12, 9, 11, 13}));
    etl::dyn_matrix_cm<T> c(3, 3);

    a = transpose(a);

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

GEMM_TN_TEST_CASE("gemm_tn/cm/6", "[gemm_tn]") {
    etl::fast_matrix_cm<T, 19, 19> a;
    etl::fast_matrix_cm<T, 19, 19> b;
    etl::fast_matrix_cm<T, 19, 19> c;

    a = etl::magic(19);
    b = etl::magic(19);

    a = transpose(a);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 828343);
    REQUIRE_EQUALS(c(1, 1), 825360);
    REQUIRE_EQUALS(c(2, 2), 826253);
    REQUIRE_EQUALS(c(3, 3), 824524);
    REQUIRE_EQUALS(c(18, 18), 828343);
}

GEMM_TN_TEST_CASE("gemm_tn/cm/7", "[gemm]") {
    etl::dyn_matrix_cm<T> a(84, 84);
    etl::dyn_matrix_cm<T> b(84, 84);
    etl::dyn_matrix_cm<T> c(84, 84);
    etl::dyn_matrix_cm<T> r(84, 84);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(k, j);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}

GEMM_TN_TEST_CASE("gemm_tn/cm/8", "[gemm]") {
    etl::dyn_matrix_cm<T> a(96, 84);
    etl::dyn_matrix_cm<T> b(96, 84);
    etl::dyn_matrix_cm<T> c(84, 84);
    etl::dyn_matrix_cm<T> r(84, 84);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(k, j);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}

GEMM_TN_TEST_CASE("gemm_tn/cm/9", "[gemm]") {
    etl::dyn_matrix_cm<T> a(84, 128);
    etl::dyn_matrix_cm<T> b(84, 96);
    etl::dyn_matrix_cm<T> c(128, 96);
    etl::dyn_matrix_cm<T> r(128, 96);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < columns(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < rows(a); k++) {
                t += a(k, i) * b(k, j);
            }
            r(i,j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, base_eps_etl_large));
}
