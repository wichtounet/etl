//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

GEVM_TEST_CASE("gevm/mixed/0", "[gevm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_vector<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_TEST_CASE("gevm/mixed/1", "[gevm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector_cm<T> c(64);
    etl::dyn_vector_cm<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_TEST_CASE("gevm/mixed/2", "[gevm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_T_TEST_CASE("gevm_t/mixed/0", "[gevm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_vector<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(j, k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_T_TEST_CASE("gevm_t/mixed/1", "[gevm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector_cm<T> c(64);
    etl::dyn_vector_cm<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(j, k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_T_TEST_CASE("gevm_t/mixed/2", "[gevm]") {
    etl::dyn_matrix<T> a(64, 64);
    etl::dyn_vector_cm<T> b(64);

    etl::dyn_vector<T> c(64);
    etl::dyn_vector<T> c_ref(64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 64; k++) {
        for (size_t j = 0; j < 64; j++) {
            c_ref(j) += b(k) * a(j, k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}
