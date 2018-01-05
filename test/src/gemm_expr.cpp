//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

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

TEMPLATE_TEST_CASE_2("lvalue/mmul1", "[gemm]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 3> a = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
    etl::fast_matrix<Z, 2, 3, 2> b = {7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12};
    etl::fast_matrix<Z, 2, 2, 2> c;

    auto s = etl::sub(c, 0);

    static_assert(etl::is_etl_expr<decltype(s)>, "");

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
