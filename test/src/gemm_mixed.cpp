//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Matrix Matrix multiplication tests with various storage order

// RM = CM * RM

GEMM_TEST_CASE("gemm/mixed/1", "[gemm]") {
    etl::dyn_matrix_cm<T> a(128, 128);
    etl::dyn_matrix<T> b(128, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/2", "[gemm]") {
    etl::dyn_matrix_cm<T> a(128, 256);
    etl::dyn_matrix<T> b(256, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/3", "[gemm]") {
    etl::dyn_matrix_cm<T> a(194, 128);
    etl::dyn_matrix<T> b(128, 156);
    etl::dyn_matrix<T> c(194, 156);
    etl::dyn_matrix<T> r(194, 156);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// RM = CM * RM

GEMM_TEST_CASE("gemm/mixed/4", "[gemm]") {
    etl::dyn_matrix<T> a(128, 128);
    etl::dyn_matrix_cm<T> b(128, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/5", "[gemm]") {
    etl::dyn_matrix<T> a(128, 256);
    etl::dyn_matrix_cm<T> b(256, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/6", "[gemm]") {
    etl::dyn_matrix<T> a(194, 128);
    etl::dyn_matrix_cm<T> b(128, 156);
    etl::dyn_matrix<T> c(194, 156);
    etl::dyn_matrix<T> r(194, 156);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// RM = CM * CM

GEMM_TEST_CASE("gemm/mixed/7", "[gemm]") {
    etl::dyn_matrix_cm<T> a(128, 128);
    etl::dyn_matrix_cm<T> b(128, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/8", "[gemm]") {
    etl::dyn_matrix_cm<T> a(128, 256);
    etl::dyn_matrix_cm<T> b(256, 128);
    etl::dyn_matrix<T> c(128, 128);
    etl::dyn_matrix<T> r(128, 128);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/9", "[gemm]") {
    etl::dyn_matrix_cm<T> a(194, 128);
    etl::dyn_matrix_cm<T> b(128, 156);
    etl::dyn_matrix<T> c(194, 156);
    etl::dyn_matrix<T> r(194, 156);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    r = 0;

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                r(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same<Impl, strassen_gemm>::value ? 10 * base_eps_etl_large : base_eps_etl_large));
}
