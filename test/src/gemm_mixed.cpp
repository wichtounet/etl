//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_matrix<T> b(64, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/2", "[gemm]") {
    etl::dyn_matrix_cm<T> a(64, 128);
    etl::dyn_matrix<T> b(128, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/3", "[gemm]") {
    etl::dyn_matrix_cm<T> a(102, 64);
    etl::dyn_matrix<T> b(64, 92);
    etl::dyn_matrix<T> c(102, 92);
    etl::dyn_matrix<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// RM = CM * RM

GEMM_TEST_CASE("gemm/mixed/4", "[gemm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_matrix_cm<T> b(64, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/5", "[gemm]") {
    etl::dyn_matrix<T> a(64, 128);
    etl::dyn_matrix_cm<T> b(128, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/6", "[gemm]") {
    etl::dyn_matrix<T> a(102, 64);
    etl::dyn_matrix_cm<T> b(64, 92);
    etl::dyn_matrix<T> c(102, 92);
    etl::dyn_matrix<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t j = 0; j < columns(b); j++) {
        for (size_t i = 0; i < rows(a); i++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// RM = CM * CM

GEMM_TEST_CASE("gemm/mixed/7", "[gemm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_matrix_cm<T> b(64, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t j = 0; j < columns(b); j++) {
        for (size_t i = 0; i < rows(a); i++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/8", "[gemm]") {
    etl::dyn_matrix_cm<T> a(64, 96);
    etl::dyn_matrix_cm<T> b(96, 64);
    etl::dyn_matrix<T> c(64, 64);
    etl::dyn_matrix<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t j = 0; j < columns(b); j++) {
        for (size_t i = 0; i < rows(a); i++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/9", "[gemm]") {
    etl::dyn_matrix_cm<T> a(102, 64);
    etl::dyn_matrix_cm<T> b(64, 92);
    etl::dyn_matrix<T> c(102, 92);
    etl::dyn_matrix<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// CM = RM * CM

GEMM_TEST_CASE("gemm/mixed/10", "[gemm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_matrix_cm<T> b(64, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/11", "[gemm]") {
    etl::dyn_matrix<T> a(64, 128);
    etl::dyn_matrix_cm<T> b(128, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/12", "[gemm]") {
    etl::dyn_matrix<T> a(102, 64);
    etl::dyn_matrix_cm<T> b(64, 92);
    etl::dyn_matrix_cm<T> c(102, 92);
    etl::dyn_matrix_cm<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// CM = CM * RM

GEMM_TEST_CASE("gemm/mixed/13", "[gemm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_matrix<T> b(64, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/14", "[gemm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_matrix<T> b(64, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/15", "[gemm]") {
    etl::dyn_matrix_cm<T> a(102, 64);
    etl::dyn_matrix<T> b(64, 92);
    etl::dyn_matrix_cm<T> c(102, 92);
    etl::dyn_matrix_cm<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

// CM = RM * RM

GEMM_TEST_CASE("gemm/mixed/16", "[gemm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_matrix<T> b(64, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE("gemm/mixed/17", "[gemm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_matrix<T> b(64, 64);
    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> r(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}

GEMM_TEST_CASE_FAST("gemm/mixed/18", "[gemm]") {
    etl::dyn_matrix<T> a(102, 64);
    etl::dyn_matrix<T> b(64, 92);
    etl::dyn_matrix_cm<T> c(102, 92);
    etl::dyn_matrix_cm<T> r(102, 92);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    for (size_t i = 0; i < rows(a); i++) {
        for (size_t j = 0; j < columns(b); j++) {
            T t(0);
            for (size_t k = 0; k < columns(a); k++) {
                t += a(i, k) * b(k, j);
            }
            r(i, j) = t;
        }
    }

    REQUIRE_DIRECT(etl::approx_equals(c, r, std::is_same_v<Impl, strassen_gemm> ? 10 * base_eps_etl_large : base_eps_etl_large));
}
