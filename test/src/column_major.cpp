//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

#include "mmul_test.hpp"
#include "conv_test.hpp"

#define CZ(a, b) std::complex<Z>(a, b)
#define ECZ(a, b) etl::complex<Z>(a, b)

TEMPLATE_TEST_CASE_2("column_major/1", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 2, 3> test_matrix(0);

    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    REQUIRE_EQUALS(test_matrix.template dim<0>(), 2UL);
    REQUIRE_EQUALS(test_matrix.template dim<1>(), 3UL);
    REQUIRE_EQUALS(test_matrix.dim(0), 2UL);
    REQUIRE_EQUALS(test_matrix.dim(1), 3UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        test_matrix[i] = i + 1;
    }

    REQUIRE_EQUALS(test_matrix(0, 0), 1);
    REQUIRE_EQUALS(test_matrix(0, 1), 3);
    REQUIRE_EQUALS(test_matrix(0, 2), 5);
    REQUIRE_EQUALS(test_matrix(1, 0), 2);
    REQUIRE_EQUALS(test_matrix(1, 1), 4);
    REQUIRE_EQUALS(test_matrix(1, 2), 6);
}

TEMPLATE_TEST_CASE_2("column_major/2", "[dyn][cm]", Z, int, long) {
    etl::dyn_matrix_cm<Z> test_matrix(2, 3);

    test_matrix = 0;

    REQUIRE_EQUALS(test_matrix.size(), 6UL);

    REQUIRE_EQUALS(test_matrix.template dim<0>(), 2UL);
    REQUIRE_EQUALS(test_matrix.template dim<1>(), 3UL);
    REQUIRE_EQUALS(test_matrix.dim(0), 2UL);
    REQUIRE_EQUALS(test_matrix.dim(1), 3UL);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        test_matrix[i] = i + 1;
    }

    REQUIRE_EQUALS(test_matrix(0, 0), 1);
    REQUIRE_EQUALS(test_matrix(0, 1), 3);
    REQUIRE_EQUALS(test_matrix(0, 2), 5);
    REQUIRE_EQUALS(test_matrix(1, 0), 2);
    REQUIRE_EQUALS(test_matrix(1, 1), 4);
    REQUIRE_EQUALS(test_matrix(1, 2), 6);
}

TEMPLATE_TEST_CASE_2("column_major/3", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 2, 3, 4> test_matrix;

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        test_matrix[i] = i + 1;
    }

    REQUIRE_EQUALS(test_matrix(0, 0, 0), 1);
    REQUIRE_EQUALS(test_matrix(0, 0, 1), 7);
    REQUIRE_EQUALS(test_matrix(0, 0, 2), 13);
    REQUIRE_EQUALS(test_matrix(0, 0, 3), 19);
    REQUIRE_EQUALS(test_matrix(0, 1, 0), 3);
    REQUIRE_EQUALS(test_matrix(0, 1, 1), 9);
    REQUIRE_EQUALS(test_matrix(0, 1, 2), 15);
    REQUIRE_EQUALS(test_matrix(0, 1, 3), 21);
    REQUIRE_EQUALS(test_matrix(0, 2, 0), 5);
    REQUIRE_EQUALS(test_matrix(0, 2, 1), 11);
    REQUIRE_EQUALS(test_matrix(0, 2, 2), 17);
    REQUIRE_EQUALS(test_matrix(0, 2, 3), 23);
    REQUIRE_EQUALS(test_matrix(1, 0, 0), 2);
    REQUIRE_EQUALS(test_matrix(1, 0, 1), 8);
    REQUIRE_EQUALS(test_matrix(1, 0, 2), 14);
    REQUIRE_EQUALS(test_matrix(1, 0, 3), 20);
    REQUIRE_EQUALS(test_matrix(1, 1, 0), 4);
    REQUIRE_EQUALS(test_matrix(1, 1, 1), 10);
    REQUIRE_EQUALS(test_matrix(1, 1, 2), 16);
    REQUIRE_EQUALS(test_matrix(1, 1, 3), 22);
    REQUIRE_EQUALS(test_matrix(1, 2, 0), 6);
    REQUIRE_EQUALS(test_matrix(1, 2, 1), 12);
    REQUIRE_EQUALS(test_matrix(1, 2, 2), 18);
    REQUIRE_EQUALS(test_matrix(1, 2, 3), 24);
}

TEMPLATE_TEST_CASE_2("column_major/4", "[dyn][cm]", Z, int, long) {
    etl::dyn_matrix_cm<Z, 3> test_matrix(2, 3, 4);

    for (size_t i = 0; i < test_matrix.size(); ++i) {
        test_matrix[i] = i + 1;
    }

    REQUIRE_EQUALS(test_matrix(0, 0, 0), 1);
    REQUIRE_EQUALS(test_matrix(0, 0, 1), 7);
    REQUIRE_EQUALS(test_matrix(0, 0, 2), 13);
    REQUIRE_EQUALS(test_matrix(0, 0, 3), 19);
    REQUIRE_EQUALS(test_matrix(0, 1, 0), 3);
    REQUIRE_EQUALS(test_matrix(0, 1, 1), 9);
    REQUIRE_EQUALS(test_matrix(0, 1, 2), 15);
    REQUIRE_EQUALS(test_matrix(0, 1, 3), 21);
    REQUIRE_EQUALS(test_matrix(0, 2, 0), 5);
    REQUIRE_EQUALS(test_matrix(0, 2, 1), 11);
    REQUIRE_EQUALS(test_matrix(0, 2, 2), 17);
    REQUIRE_EQUALS(test_matrix(0, 2, 3), 23);
    REQUIRE_EQUALS(test_matrix(1, 0, 0), 2);
    REQUIRE_EQUALS(test_matrix(1, 0, 1), 8);
    REQUIRE_EQUALS(test_matrix(1, 0, 2), 14);
    REQUIRE_EQUALS(test_matrix(1, 0, 3), 20);
    REQUIRE_EQUALS(test_matrix(1, 1, 0), 4);
    REQUIRE_EQUALS(test_matrix(1, 1, 1), 10);
    REQUIRE_EQUALS(test_matrix(1, 1, 2), 16);
    REQUIRE_EQUALS(test_matrix(1, 1, 3), 22);
    REQUIRE_EQUALS(test_matrix(1, 2, 0), 6);
    REQUIRE_EQUALS(test_matrix(1, 2, 1), 12);
    REQUIRE_EQUALS(test_matrix(1, 2, 2), 18);
    REQUIRE_EQUALS(test_matrix(1, 2, 3), 24);
}

TEMPLATE_TEST_CASE_2("column_major/transpose/1", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 2, 3> a;
    a = etl::sequence_generator(1);

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 3);
    REQUIRE_EQUALS(a(0, 2), 5);
    REQUIRE_EQUALS(a(1, 0), 2);
    REQUIRE_EQUALS(a(1, 1), 4);
    REQUIRE_EQUALS(a(1, 2), 6);

    etl::fast_matrix_cm<Z, 3, 2> b;
    b = etl::transpose(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(1, 0), 3);
    REQUIRE_EQUALS(b(2, 0), 5);
    REQUIRE_EQUALS(b(0, 1), 2);
    REQUIRE_EQUALS(b(1, 1), 4);
    REQUIRE_EQUALS(b(2, 1), 6);
}

TEMPLATE_TEST_CASE_2("column_major/tranpose/2", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 3, 2> a;
    a = etl::sequence_generator(1);

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 4);
    REQUIRE_EQUALS(a(1, 0), 2);
    REQUIRE_EQUALS(a(1, 1), 5);
    REQUIRE_EQUALS(a(2, 0), 3);
    REQUIRE_EQUALS(a(2, 1), 6);

    etl::fast_matrix_cm<Z, 2, 3> b;
    b = etl::transpose(a);

    REQUIRE_EQUALS(b(0, 0), 1);
    REQUIRE_EQUALS(b(0, 1), 2);
    REQUIRE_EQUALS(b(0, 2), 3);
    REQUIRE_EQUALS(b(1, 0), 4);
    REQUIRE_EQUALS(b(1, 1), 5);
    REQUIRE_EQUALS(b(1, 2), 6);
}

TEMPLATE_TEST_CASE_2("column_major/binary/1", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 3, 2> a;
    etl::fast_matrix_cm<Z, 3, 2> b;

    a = etl::sequence_generator(1);
    b = a + a - a + a;

    REQUIRE_EQUALS(a(0, 0), 1);
    REQUIRE_EQUALS(a(0, 1), 4);
    REQUIRE_EQUALS(a(1, 0), 2);
    REQUIRE_EQUALS(a(1, 1), 5);
    REQUIRE_EQUALS(a(2, 0), 3);
    REQUIRE_EQUALS(a(2, 1), 6);
}

TEMPLATE_TEST_CASE_2("column_major/sub/1", "[fast][cm]", Z, int, long) {
    etl::fast_matrix_cm<Z, 3, 3, 2> a;

    a(0) = etl::sequence_generator(1);
    a(1) = etl::sequence_generator(7);
    a(2) = etl::sequence_generator(13);

    etl::fast_matrix_cm<Z, 3, 2> b;
    b = a(1);

    REQUIRE_EQUALS(a(0)(0, 0), 1);
    REQUIRE_EQUALS(a(0)(0, 1), 4);
    REQUIRE_EQUALS(a(0)(1, 0), 2);
    REQUIRE_EQUALS(a(0)(1, 1), 5);
    REQUIRE_EQUALS(a(0)(2, 0), 3);
    REQUIRE_EQUALS(a(0)(2, 1), 6);

    REQUIRE_EQUALS(b(0, 0), 7);
    REQUIRE_EQUALS(b(0, 1), 10);
    REQUIRE_EQUALS(b(1, 0), 8);
    REQUIRE_EQUALS(b(1, 1), 11);
    REQUIRE_EQUALS(b(2, 0), 9);
    REQUIRE_EQUALS(b(2, 1), 12);
}

GEMM_TEST_CASE("column_major/mul/1", "mmul") {
    etl::fast_matrix_cm<T, 2, 3> a = {1, 4, 2, 5, 3, 6};
    etl::fast_matrix_cm<T, 3, 2> b = {7, 9, 11, 8, 10, 12};
    etl::fast_matrix_cm<T, 2, 2> c;

    REQUIRE_EQUALS(etl::rows(a), 2UL);
    REQUIRE_EQUALS(etl::columns(a), 3UL);
    REQUIRE_EQUALS(etl::rows(b), 3UL);
    REQUIRE_EQUALS(etl::columns(b), 2UL);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
}

GEMM_TEST_CASE("column_major/gemm/1", "[cm][gevm]") {
    etl::dyn_matrix_cm<T> a(64, 64);
    etl::dyn_matrix_cm<T> b(64, 64);

    etl::dyn_matrix_cm<T> c(64, 64);
    etl::dyn_matrix_cm<T> c_ref(64, 64);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t j = 0; j < 64; j++) {
        for (size_t k = 0; k < 64; k++) {
            for (size_t i = 0; i < 64; i++) {
                c_ref(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX_E(c[i], c_ref[i], base_eps * 30);
    }
}

GEMV_TEST_CASE("column_major/gemv/0", "[mul]") {
    etl::fast_matrix_cm<T, 2, 3> a = {1, 4, 2, 5, 3, 6};
    etl::fast_vector_cm<T, 3> b    = {7, 8, 9};
    etl::fast_matrix_cm<T, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0), 50);
    REQUIRE_EQUALS(c(1), 122);
}

GEMV_TEST_CASE("column_major/gemv/1", "[gemv]") {
    etl::dyn_matrix_cm<T> a(512, 512);
    etl::dyn_matrix_cm<T,1> b(512);

    etl::dyn_matrix_cm<T,1> c(512);
    etl::dyn_matrix_cm<T,1> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t k = 0; k < 512; k++) {
        for (size_t i = 0; i < 512; i++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEVM_TEST_CASE("column_major/gevm/0", "[cm][gevm]") {
    etl::fast_matrix_cm<T, 3, 2> a = {1, 3, 5, 2, 4, 6};
    etl::fast_vector_cm<T, 3> b    = {7, 8, 9};
    etl::fast_matrix_cm<T, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS(c(0), 76);
    REQUIRE_EQUALS(c(1), 100);
}

GEVM_TEST_CASE("column_major/gevm/1", "[cm][gevm]") {
    etl::dyn_matrix_cm<T> a(512, 512);
    etl::dyn_matrix_cm<T, 1> b(512);

    etl::dyn_matrix_cm<T, 1> c(512);
    etl::dyn_matrix_cm<T, 1> c_ref(512);

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

CONV1_FULL_TEST_CASE("column_major/conv/full_1", "[cm][conv]") {
    etl::fast_vector_cm<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector_cm<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector_cm<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c[0], 0.0);
    REQUIRE_EQUALS_APPROX(c[1], 1.0);
    REQUIRE_EQUALS_APPROX(c[2], 2.5);
    REQUIRE_EQUALS_APPROX(c[3], 4.0);
    REQUIRE_EQUALS_APPROX(c[4], 1.5);
}

CONV2_FULL_TEST_CASE_CM("column_major/conv2/full_1", "[cm][conv2]") {
    etl::fast_matrix_cm<T, 3, 3> a = {1.0, 0.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0};
    etl::fast_matrix_cm<T, 2, 2> b = {2.0, 0.5, 0.0, 0.5};
    etl::fast_matrix_cm<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(0.5));
}

CONV2_FULL_TEST_CASE_CM("column_major/conv2/full_2", "[cm][conv2]") {
    etl::fast_matrix_cm<T, 3, 3> a;
    etl::fast_matrix_cm<T, 2, 2> b;
    etl::fast_matrix_cm<T, 4, 4> c;

    a = etl::magic<T>(3);
    b = etl::magic<T>(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(8));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(25));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(9));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(18));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(35));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(34));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(48));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(33));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(16));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(47));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(67));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(20));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(16));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(44));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(26));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(4));
}

CONV2_FULL_TEST_CASE_CM("column_major/conv2/full_3", "[cm][conv2]") {
    etl::fast_matrix_cm<T, 2, 6> a = {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
    etl::fast_matrix_cm<T, 2, 2> b = {1, 3, 2, 4};
    etl::fast_matrix_cm<T, 3, 7> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(7));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(10));
    REQUIRE_EQUALS_APPROX(c(0, 4), T(13));
    REQUIRE_EQUALS_APPROX(c(0, 5), T(16));
    REQUIRE_EQUALS_APPROX(c(0, 6), T(12));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(10));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(32));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(42));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(52));
    REQUIRE_EQUALS_APPROX(c(1, 4), T(62));
    REQUIRE_EQUALS_APPROX(c(1, 5), T(72));
    REQUIRE_EQUALS_APPROX(c(1, 6), T(48));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(21));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(52));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(59));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(66));
    REQUIRE_EQUALS_APPROX(c(2, 4), T(73));
    REQUIRE_EQUALS_APPROX(c(2, 5), T(80));
    REQUIRE_EQUALS_APPROX(c(2, 6), T(48));
}

TEMPLATE_TEST_CASE_2("column_major/compound/add_1", "[cm]", Z, float, double) {
    etl::fast_matrix_cm<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix_cm<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a(0, 0), 1.5);
    REQUIRE_EQUALS(a(0, 1), 9.0);
    REQUIRE_EQUALS(a(1, 0), 5.0);
    REQUIRE_EQUALS(a(1, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("column_major/compound/add_2", "[cm]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a    = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix_cm<Z, 2, 2> b = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a(0, 0), 1.5);
    REQUIRE_EQUALS(a(0, 1), 6.0);
    REQUIRE_EQUALS(a(1, 0), 8.0);
    REQUIRE_EQUALS(a(1, 1), 2.0);
}

TEMPLATE_TEST_CASE_2("column_major/compound/add_3", "[cm]", Z, float, double) {
    etl::fast_matrix_cm<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};
    etl::fast_matrix<Z, 2, 2> b    = {2.5, 3.0, 4.0, 1.0};

    a += b;

    REQUIRE_EQUALS(a(0, 0), 1.5);
    REQUIRE_EQUALS(a(0, 1), 8.0);
    REQUIRE_EQUALS(a(1, 0), 6.0);
    REQUIRE_EQUALS(a(1, 1), 2.0);
}

// Complex multiplication tests

GEMM_TEST_CASE("column_major/complex/mul/0", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix_cm<std::complex<Z>, 2, 3> a = {CZ(1, 1), CZ(0, 0), CZ(-2, -2),CZ(1, 1), CZ(2, 3), CZ(2, 2)};
    etl::fast_matrix_cm<std::complex<Z>, 3, 2> b = {CZ(1, 1), CZ(3, 2), CZ(1, -1), CZ(2, 2),CZ(1, 0), CZ(2, 2)};
    etl::fast_matrix_cm<std::complex<Z>, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0).real(), 3.0);
    REQUIRE_EQUALS(c(0, 0).imag(), -7.0);
    REQUIRE_EQUALS(c(0, 1).real(), -4.0);
    REQUIRE_EQUALS(c(0, 1).imag(), 12.0);
    REQUIRE_EQUALS(c(1, 0).real(), 5.0);
    REQUIRE_EQUALS(c(1, 0).imag(), 5.0);
    REQUIRE_EQUALS(c(1, 1).real(), 1.0);
    REQUIRE_EQUALS(c(1, 1).imag(), 9.0);
}

GEMV_TEST_CASE("column_major/complex/mul/1", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix_cm<etl::complex<Z>, 2, 3> a = {ECZ(1, 1), ECZ(0, 0), ECZ(-2, -2),ECZ(1, 1), ECZ(2, 3), ECZ(2, 2)};
    etl::fast_vector_cm<etl::complex<Z>, 3> b    = {ECZ(1, 1), ECZ(-3, -3), ECZ(5, 0.1)};
    etl::fast_matrix_cm<etl::complex<Z>, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS_APPROX(c(0).real, Z(9.7));
    REQUIRE_EQUALS_APPROX(c(0).imag, Z(29.2));
    REQUIRE_EQUALS_APPROX(c(1).real, Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag, Z(4.2));
}

GEVM_TEST_CASE("column_major/complex/mul/2", "[mul][complex]") {
    using Z = T;

    etl::fast_matrix_cm<std::complex<Z>, 3, 2> a = {CZ(1, 1), CZ(2, 3), CZ(1, 1), CZ(-2, -2), CZ(0, 0), CZ(2, 2)};
    etl::fast_vector_cm<std::complex<Z>, 3> b    = {CZ(1, 1), CZ(-3, -3), CZ(5, 0.1)};
    etl::fast_matrix_cm<std::complex<Z>, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), Z(7.9));
    REQUIRE_EQUALS_APPROX(c(0).imag(), Z(-7.9));
    REQUIRE_EQUALS_APPROX(c(1).real(), Z(9.8));
    REQUIRE_EQUALS_APPROX(c(1).imag(), Z(6.2));
}
