//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

GEVM_TEST_CASE("gevm/cm/0", "[gevm]") {
    etl::fast_matrix_cm<T, 3, 2> a = {1, 3, 5, 2, 4, 6};
    etl::fast_vector<T, 3> b    = {7, 8, 9};
    etl::fast_matrix_cm<T, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS(c(0), 76);
    REQUIRE_EQUALS(c(1), 100);
}

GEVM_TEST_CASE("gevm/cm/1", "[gevm]") {
    etl::dyn_matrix_cm<T> a(3, 2, etl::values(1, 3, 5, 2, 4, 6));
    etl::dyn_vector<T> b(3, etl::values(7, 8, 9));
    etl::dyn_vector<T> c(2);

    Impl::apply(b, a, c);

    REQUIRE_EQUALS(c(0), 76);
    REQUIRE_EQUALS(c(1), 100);
}

GEVM_TEST_CASE("gevm/cm/2", "[gevm]") {
    etl::dyn_matrix_cm<T> a(512, 512);
    etl::dyn_vector<T> b(512);

    etl::dyn_vector<T> c(512);
    etl::dyn_vector<T> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 512; k++) {
        for (size_t j = 0; j < 512; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_TEST_CASE("gevm/cm/3", "[gevm]") {
    etl::dyn_matrix_cm<T> a(512, 368);
    etl::dyn_vector<T> b(512);

    etl::dyn_vector<T> c(368);
    etl::dyn_vector<T> c_ref(368);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 512; k++) {
        for (size_t j = 0; j < 368; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_TEST_CASE("gevm/cm/4", "[gevm]") {
    etl::dyn_matrix_cm<T> a(368, 512);
    etl::dyn_vector<T> b(368);

    etl::dyn_vector<T> c(512);
    etl::dyn_vector<T> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t k = 0; k < 368; k++) {
        for (size_t j = 0; j < 512; j++) {
            c_ref(j) += b(k) * a(k, j);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_T_TEST_CASE("gevm_t/cm//1", "[gevm][gevm_t]") {
    etl::dyn_matrix_cm<T> a(512, 368);
    etl::dyn_vector<T> b(368);

    etl::dyn_vector<T> c(512);
    etl::dyn_vector<T> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t j = 0; j < 512; j++) {
        for (size_t k = 0; k < 368; k++) {
            c_ref(j) += b(k) * a(j, k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_T_TEST_CASE("gevm_t/cm/2", "[gevm][gevm_t]") {
    etl::dyn_matrix_cm<T> a(368, 512);
    etl::dyn_vector<T> b(512);

    etl::dyn_vector<T> c(368);
    etl::dyn_vector<T> c_ref(368);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(b, a, c);

    c_ref = 0;

    for (size_t j = 0; j < 368; j++) {
        for (size_t k = 0; k < 512; k++) {
            c_ref(j) += b(k) * a(j, k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}
