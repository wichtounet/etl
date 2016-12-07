//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

// Matrix Matrix multiplication tests

GEMM_TEST_CASE("multiplication/mm_mul_1", "[gemm]") {
    etl::fast_matrix<T, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<T, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 58);
    REQUIRE_EQUALS(c(0, 1), 64);
    REQUIRE_EQUALS(c(1, 0), 139);
    REQUIRE_EQUALS(c(1, 1), 154);
}

GEMM_TEST_CASE("multiplication/mm_mul_2", "[gemm]") {
    etl::fast_matrix<T, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<T, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};
    etl::fast_matrix<T, 3, 3> c;

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

GEMM_TEST_CASE("multiplication/mm_mul_3", "[gemm]") {
    etl::dyn_matrix<T> a(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<T> b(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<T> c(4, 4);

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

GEMM_TEST_CASE("multiplication/mm_mul_4", "[gemm]") {
    etl::dyn_matrix<T> a(2, 2, etl::values(1, 2, 3, 4));
    etl::dyn_matrix<T> b(2, 2, etl::values(1, 2, 3, 4));
    etl::dyn_matrix<T> c(2, 2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 7);
    REQUIRE_EQUALS(c(0, 1), 10);
    REQUIRE_EQUALS(c(1, 0), 15);
    REQUIRE_EQUALS(c(1, 1), 22);
}

GEMM_TEST_CASE("multiplication/mm_mul_5", "[gemm]") {
    etl::dyn_matrix<T> a(3, 3, std::initializer_list<T>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    etl::dyn_matrix<T> b(3, 3, std::initializer_list<T>({7, 8, 9, 9, 10, 11, 11, 12, 13}));
    etl::dyn_matrix<T> c(3, 3);

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

GEMM_TEST_CASE("multiplication/mm_mul_6", "[gemm]") {
    etl::fast_matrix<T, 19, 19> a(etl::magic(19));
    etl::fast_matrix<T, 19, 19> b(etl::magic(19));
    etl::fast_matrix<T, 19, 19> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0, 0), 828343);
    REQUIRE_EQUALS(c(1, 1), 825360);
    REQUIRE_EQUALS(c(2, 2), 826253);
    REQUIRE_EQUALS(c(3, 3), 824524);
    REQUIRE_EQUALS(c(18, 18), 828343);
}

TEMPLATE_TEST_CASE_2("multiplication/mm_mul_7", "[gemm]", Z, double, float) {
    etl::fast_matrix<Z, 1, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<Z, 1, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<Z, 1, 2, 2> c;

    c(0) = a(0) * b(0);

    REQUIRE_EQUALS(c(0, 0, 0), 58);
    REQUIRE_EQUALS(c(0, 0, 1), 64);
    REQUIRE_EQUALS(c(0, 1, 0), 139);
    REQUIRE_EQUALS(c(0, 1, 1), 154);
}

GEMM_TEST_CASE_PRE("multiplication/mm_mul_8", "[gemm]") {
    etl::fast_matrix<T, 128, 128> a;
    etl::fast_matrix<T, 128, 128> b;

    etl::fast_matrix<T, 128, 128> c;
    etl::fast_matrix<T, 128, 128> c_ref;

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (std::size_t i = 0; i < 128; i++) {
        for (std::size_t k = 0; k < 128; k++) {
            for (std::size_t j = 0; j < 128; j++) {
                c_ref(i, j) += a(i, k) * b(k, j);
            }
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX_E(c[i], c_ref[i], base_eps);
    }
}

// Matrix-Vector Multiplication

GEMV_TEST_CASE("multiplication/gemv/0", "[gemv]") {
    etl::fast_matrix<T, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<T, 3> b    = {7, 8, 9};
    etl::fast_matrix<T, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0), 50);
    REQUIRE_EQUALS(c(1), 122);
}

GEMV_TEST_CASE("multiplication/gemv/1", "[gemv]") {
    etl::fast_matrix<T, 2, 5> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    etl::fast_vector<T, 5> b    = {7, 8, 9, 10, 11};
    etl::fast_matrix<T, 2> c;

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0), 145);
    REQUIRE_EQUALS(c(1), 370);
}

GEMV_TEST_CASE("multiplication/gemv/2", "[gemv]") {
    etl::dyn_matrix<T> a(2, 3, etl::values(1, 2, 3, 4, 5, 6));
    etl::dyn_vector<T> b(3, etl::values(7, 8, 9));
    etl::dyn_vector<T> c(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0), 50);
    REQUIRE_EQUALS(c(1), 122);
}

GEMV_TEST_CASE("multiplication/gemv/3", "[gemv]") {
    etl::dyn_matrix<T> a(2, 5, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
    etl::dyn_vector<T> b(5, etl::values(7, 8, 9, 10, 11));
    etl::dyn_vector<T> c(2);

    Impl::apply(a, b, c);

    REQUIRE_EQUALS(c(0), 145);
    REQUIRE_EQUALS(c(1), 370);
}

GEMV_TEST_CASE("multiplication/gemv/4", "[gemv]") {
    etl::dyn_matrix<T> a(512, 512);
    etl::dyn_vector<T> b(512);

    etl::dyn_vector<T> c(512);
    etl::dyn_vector<T> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 512; i++) {
        for (size_t k = 0; k < 512; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_TEST_CASE("multiplication/gemv/5", "[gemv]") {
    etl::dyn_matrix<T> a(512, 368);
    etl::dyn_vector<T> b(368);

    etl::dyn_vector<T> c(512);
    etl::dyn_vector<T> c_ref(512);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 512; i++) {
        for (size_t k = 0; k < 368; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

GEMV_TEST_CASE("multiplication/gemv/6", "[gemv]") {
    etl::dyn_matrix<T> a(368, 512);
    etl::dyn_vector<T> b(512);

    etl::dyn_vector<T> c(368);
    etl::dyn_vector<T> c_ref(368);

    a = 0.01 * etl::sequence_generator(1.0);
    b = -0.032 * etl::sequence_generator(1.0);

    Impl::apply(a, b, c);

    c_ref = 0;

    for (size_t i = 0; i < 368; i++) {
        for (size_t k = 0; k < 512; k++) {
            c_ref(i) += a(i, k) * b(k);
        }
    }

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

// Vector-Matrix Multiplication

GEVM_TEST_CASE("multiplication/gevm/0", "[gevm]") {
    etl::fast_matrix<T, 3, 2> a = {1, 2, 3, 4, 5, 6};
    etl::fast_vector<T, 3> b    = {7, 8, 9};
    etl::fast_matrix<T, 2> c;

    Impl::apply(b, a, c);

    REQUIRE_EQUALS(c(0), 76);
    REQUIRE_EQUALS(c(1), 100);
}

GEVM_TEST_CASE("multiplication/gevm/1", "[gevm]") {
    etl::dyn_matrix<T> a(3, 2, etl::values(1, 2, 3, 4, 5, 6));
    etl::dyn_vector<T> b(3, etl::values(7, 8, 9));
    etl::dyn_vector<T> c(2);

    Impl::apply(b, a, c);

    REQUIRE_EQUALS(c(0), 76);
    REQUIRE_EQUALS(c(1), 100);
}

GEVM_TEST_CASE("multiplication/gevm/2", "[gevm]") {
    etl::dyn_matrix<T> a(512, 512);
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

GEVM_TEST_CASE("multiplication/gevm/3", "[gevm]") {
    etl::dyn_matrix<T> a(512, 368);
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

GEVM_TEST_CASE("multiplication/gevm/4", "[gevm]") {
    etl::dyn_matrix<T> a(368, 512);
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

//Test using expressions directly
TEMPLATE_TEST_CASE_2("multiplication/expression", "[gemm]", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<Z, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};

    auto c = *etl::mul(a, b);

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

TEMPLATE_TEST_CASE_2("multiplication/expr_mmul_1", "[gemm]", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    etl::dyn_matrix<Z> b(3, 3, std::initializer_list<Z>({7, 8, 9, 9, 10, 11, 11, 12, 13}));
    etl::dyn_matrix<Z> c(3, 3);

    etl::force(etl::mul(a + b - b, a + b - a, c));

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

TEMPLATE_TEST_CASE_2("multiplication/expr_mmul_2", "[gemm]", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    etl::dyn_matrix<Z> b(3, 3, std::initializer_list<Z>({7, 8, 9, 9, 10, 11, 11, 12, 13}));
    etl::dyn_matrix<Z> c(3, 3);

    etl::force(etl::mul(abs(a), abs(b), c));

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

TEMPLATE_TEST_CASE_2("multiplication/stop_mmul_1", "[gemm]", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 3, std::initializer_list<Z>({1, 2, 3, 4, 5, 6, 7, 8, 9}));
    etl::dyn_matrix<Z> b(3, 3, std::initializer_list<Z>({7, 8, 9, 9, 10, 11, 11, 12, 13}));
    etl::dyn_matrix<Z> c(3, 3);

    c = etl::mul(s(abs(a)), s(abs(b)));

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

// Expressions

TEMPLATE_TEST_CASE_2("multiplication/expression_1", "expression", Z, double, float) {
    etl::dyn_matrix<Z> a(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<Z> b(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<Z> c(4, 4);

    c = 2.0 * mul(a, b) + mul(a, b) / 1.1;

    REQUIRE_EQUALS_APPROX(c(0, 0), (2.0 + 1.0 / 1.1) * 90);
    REQUIRE_EQUALS_APPROX(c(0, 1), (2.0 + 1.0 / 1.1) * 100);
    REQUIRE_EQUALS_APPROX(c(1, 0), (2.0 + 1.0 / 1.1) * 202);
    REQUIRE_EQUALS_APPROX(c(1, 1), (2.0 + 1.0 / 1.1) * 228);
    REQUIRE_EQUALS_APPROX(c(2, 0), (2.0 + 1.0 / 1.1) * 314);
    REQUIRE_EQUALS_APPROX(c(2, 1), (2.0 + 1.0 / 1.1) * 356);
    REQUIRE_EQUALS_APPROX(c(3, 0), (2.0 + 1.0 / 1.1) * 426);
    REQUIRE_EQUALS_APPROX(c(3, 1), (2.0 + 1.0 / 1.1) * 484);
}

TEMPLATE_TEST_CASE_2("multiplication/expression_2", "expression", Z, double, float) {
    etl::dyn_matrix<Z> a(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<Z> b(4, 4, etl::values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    etl::dyn_matrix<Z> c(4, 4);

    c = 2.0 * etl::lazy_mul(a, b) + etl::lazy_mul(a, b) / 1.1;

    REQUIRE_EQUALS_APPROX(c(0, 0), (2.0 + 1.0 / 1.1) * 90);
    REQUIRE_EQUALS_APPROX(c(0, 1), (2.0 + 1.0 / 1.1) * 100);
    REQUIRE_EQUALS_APPROX(c(1, 0), (2.0 + 1.0 / 1.1) * 202);
    REQUIRE_EQUALS_APPROX(c(1, 1), (2.0 + 1.0 / 1.1) * 228);
    REQUIRE_EQUALS_APPROX(c(2, 0), (2.0 + 1.0 / 1.1) * 314);
    REQUIRE_EQUALS_APPROX(c(2, 1), (2.0 + 1.0 / 1.1) * 356);
    REQUIRE_EQUALS_APPROX(c(3, 0), (2.0 + 1.0 / 1.1) * 426);
    REQUIRE_EQUALS_APPROX(c(3, 1), (2.0 + 1.0 / 1.1) * 484);
}

TEMPLATE_TEST_CASE_2("multiplication/expression_3", "expression", Z, double, float) {
    etl::fast_matrix<Z, 2, 3> a = {1, 2, 3, 4, 5, 6};
    etl::fast_matrix<Z, 3, 2> b = {7, 8, 9, 10, 11, 12};
    etl::fast_matrix<Z, 2, 2> c;

    c = a * b;
    c += a * b;

    REQUIRE_EQUALS(c(0, 0), 2 * 58);
    REQUIRE_EQUALS(c(0, 1), 2 * 64);
    REQUIRE_EQUALS(c(1, 0), 2 * 139);
    REQUIRE_EQUALS(c(1, 1), 2 * 154);
}

// outer product

TEMPLATE_TEST_CASE_2("fast_vector/outer_1", "sum", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector<Z, 3> b = {4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 3, 3> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 4.0);
    REQUIRE_EQUALS(c(0, 1), 5.0);
    REQUIRE_EQUALS(c(0, 2), 6.0);

    REQUIRE_EQUALS(c(1, 0), 8.0);
    REQUIRE_EQUALS(c(1, 1), 10.0);
    REQUIRE_EQUALS(c(1, 2), 12.0);

    REQUIRE_EQUALS(c(2, 0), 12.0);
    REQUIRE_EQUALS(c(2, 1), 15.0);
    REQUIRE_EQUALS(c(2, 2), 18.0);
}

TEMPLATE_TEST_CASE_2("fast_vector/outer_2", "sum", Z, float, double) {
    etl::fast_vector<Z, 2> a = {1.0, 2.0};
    etl::fast_vector<Z, 4> b = {2.0, 3.0, 4.0, 5.0};
    etl::fast_matrix<Z, 2, 4> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 2);
    REQUIRE_EQUALS(c(0, 1), 3);
    REQUIRE_EQUALS(c(0, 2), 4);
    REQUIRE_EQUALS(c(0, 3), 5);

    REQUIRE_EQUALS(c(1, 0), 4);
    REQUIRE_EQUALS(c(1, 1), 6);
    REQUIRE_EQUALS(c(1, 2), 8);
    REQUIRE_EQUALS(c(1, 3), 10);
}

TEMPLATE_TEST_CASE_2("fast_vector/outer_3", "sum", Z, float, double) {
    etl::fast_vector<Z, 2> a = {1.0, 2.0};
    etl::dyn_vector<Z> b(4, etl::values(2.0, 3.0, 4.0, 5.0));
    etl::fast_matrix<Z, 2, 4> c;

    c = outer(a, b);

    REQUIRE_EQUALS(c(0, 0), 2);
    REQUIRE_EQUALS(c(0, 1), 3);
    REQUIRE_EQUALS(c(0, 2), 4);
    REQUIRE_EQUALS(c(0, 3), 5);

    REQUIRE_EQUALS(c(1, 0), 4);
    REQUIRE_EQUALS(c(1, 1), 6);
    REQUIRE_EQUALS(c(1, 2), 8);
    REQUIRE_EQUALS(c(1, 3), 10);
}

TEMPLATE_TEST_CASE_2("lvalue/mmul1", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 3> a = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    etl::fast_matrix<Z, 2, 3, 2> b = {7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<Z, 2, 2, 2> c;

    auto s = etl::sub(c, 0);

    static_assert(etl::is_etl_expr<decltype(s)>::value, "");

    etl::force(etl::mul(etl::sub(a, 0), etl::sub(b, 0), s));

    REQUIRE_EQUALS(c(0, 0, 0), 58);
    REQUIRE_EQUALS(c(0, 0, 1), 64);
    REQUIRE_EQUALS(c(0, 1, 0), 139);
    REQUIRE_EQUALS(c(0, 1, 1), 154);

    etl::sub(c, 1) = etl::sub(c, 0);

    REQUIRE_EQUALS(c(1, 0, 0), 58);
    REQUIRE_EQUALS(c(1, 0, 1), 64);
    REQUIRE_EQUALS(c(1, 1, 0), 139);
    REQUIRE_EQUALS(c(1, 1, 1), 154);
}

TEMPLATE_TEST_CASE_2("lvalue/mmul2", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 3> a = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    etl::fast_matrix<Z, 2, 3, 2> b = {7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<Z, 2, 2, 2> c;

    etl::force(etl::mul(etl::sub(a, 0), etl::sub(b, 0), etl::sub(c, 0)));

    REQUIRE_EQUALS(c(0, 0, 0), 58);
    REQUIRE_EQUALS(c(0, 0, 1), 64);
    REQUIRE_EQUALS(c(0, 1, 0), 139);
    REQUIRE_EQUALS(c(0, 1, 1), 154);

    etl::sub(c, 1) = etl::sub(c, 0);

    REQUIRE_EQUALS(c(1, 0, 0), 58);
    REQUIRE_EQUALS(c(1, 0, 1), 64);
    REQUIRE_EQUALS(c(1, 1, 0), 139);
    REQUIRE_EQUALS(c(1, 1, 1), 154);
}

// batch outer product

TEMPLATE_TEST_CASE_2("batch_outer/1", "[outer]", Z, float, double) {
    etl::fast_matrix<Z, 2, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    etl::fast_matrix<Z, 2, 3> b = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    etl::fast_matrix<Z, 3, 3> c;
    etl::fast_matrix<Z, 3, 3> c_ref;

    c = batch_outer(a, b);

    c_ref = 0;

    for (std::size_t bb = 0; bb < 2; ++bb) {
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_outer/2", "[outer]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(32, 31);
    etl::dyn_matrix<Z, 2> b(32, 23);

    a = Z(0.01) * etl::sequence_generator<Z>(1.0);
    b = Z(-0.032) * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 2> c(31, 23);
    etl::dyn_matrix<Z, 2> c_ref(31, 23);

    c = batch_outer(a, b);

    c_ref = 0;

    for (std::size_t bb = 0; bb < 32; ++bb) {
        for (std::size_t i = 0; i < 31; ++i) {
            for (std::size_t j = 0; j < 23; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("batch_outer/3", "[outer]", Z, float, double) {
    etl::dyn_matrix<Z, 2> a(32, 24);
    etl::dyn_matrix<Z, 2> b(32, 33);

    a = Z(0.01) * etl::sequence_generator<Z>(1.0);
    b = Z(-0.032) * etl::sequence_generator<Z>(1.0);

    etl::dyn_matrix<Z, 2> c(24, 33);
    etl::dyn_matrix<Z, 2> c_ref(24, 33);

    c = batch_outer(a, b);

    c_ref = 0;

    for (std::size_t bb = 0; bb < 32; ++bb) {
        for (std::size_t i = 0; i < 24; ++i) {
            for (std::size_t j = 0; j < 33; ++j) {

                c_ref(i, j) += a(bb, i) * b(bb, j);
            }
        }
    }

    for(size_t i = 0; i < c_ref.size(); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

#ifdef ETL_CUDA

TEMPLATE_TEST_CASE_2("gpu/mmul_1", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<Z, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};
    etl::fast_matrix<Z, 3, 3> c;

    c = a * b;

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

TEMPLATE_TEST_CASE_2("gpu/mmul_2", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<Z, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};
    etl::fast_matrix<Z, 3, 3> c = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    etl::fast_matrix<Z, 3, 3> d;

    d = a * b * c;

    REQUIRE_EQUALS(d(0, 0), 58);
    REQUIRE_EQUALS(d(0, 1), 64);
    REQUIRE_EQUALS(d(0, 2), 70);
    REQUIRE_EQUALS(d(1, 0), 139);
    REQUIRE_EQUALS(d(1, 1), 154);
    REQUIRE_EQUALS(d(1, 2), 169);
    REQUIRE_EQUALS(d(2, 0), 220);
    REQUIRE_EQUALS(d(2, 1), 244);
    REQUIRE_EQUALS(d(2, 2), 268);
}

TEMPLATE_TEST_CASE_2("gpu/mmul_3", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<Z, 3, 3> b = {7, 8, 9, 9, 10, 11, 11, 12, 13};
    etl::fast_matrix<Z, 3, 3> d;

    d = 2.0 * (a * b);

    REQUIRE_EQUALS(d(0, 0), 2 * 58);
    REQUIRE_EQUALS(d(0, 1), 2 * 64);
    REQUIRE_EQUALS(d(0, 2), 2 * 70);
    REQUIRE_EQUALS(d(1, 0), 2 * 139);
    REQUIRE_EQUALS(d(1, 1), 2 * 154);
    REQUIRE_EQUALS(d(1, 2), 2 * 169);
    REQUIRE_EQUALS(d(2, 0), 2 * 220);
    REQUIRE_EQUALS(d(2, 1), 2 * 244);
    REQUIRE_EQUALS(d(2, 2), 2 * 268);
}

#endif //ETL_CUDA
