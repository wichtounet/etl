//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

GEMV_TEST_CASE("gemv/mixed/0", "[gemv]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_vector<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_TEST_CASE("gemv/mixed/1", "[gemv]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector_cm<T> c(64);
    etl::dyn_vector_cm<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_TEST_CASE("gemv/mixed/2", "[gemv]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_T_TEST_CASE("gemv_t/mixed/0", "[gemv]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_vector<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(k, i) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_T_TEST_CASE("gemv_t/mixed/1", "[gemv]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector_cm<T> c(64);
    etl::dyn_vector_cm<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(k, i) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_T_TEST_CASE("gemv_t/mixed/2", "[gemv]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 64; i++) {
        for (size_t k = 0; k < 64; k++) {
            c_ref(i) += a(k, i) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}
