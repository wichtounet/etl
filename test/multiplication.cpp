//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"
#include "etl/stop.hpp"

#include "mmul_test.hpp"

//{{{ Matrix Matrix multiplication tests

MMUL_TEST_CASE( "multiplication/mm_mul_1", "mmul") {
    etl::fast_matrix<T, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<T, 3, 2> b = {7,8,9,10,11,12};
    etl::fast_matrix<T, 2, 2> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

MMUL_TEST_CASE( "multiplication/mm_mul_2", "mmul") {
    etl::fast_matrix<T, 3, 3> a = {1,2,3,4,5,6,7,8,9};
    etl::fast_matrix<T, 3, 3> b = {7,8,9,9,10,11,11,12,13};
    etl::fast_matrix<T, 3, 3> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

MMUL_TEST_CASE( "multiplication/mm_mul_3", "mmul") {
    etl::dyn_matrix<T> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<T> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<T> c(4,4);

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 90);
    REQUIRE(c(0,1) == 100);
    REQUIRE(c(1,0) == 202);
    REQUIRE(c(1,1) == 228);
    REQUIRE(c(2,0) == 314);
    REQUIRE(c(2,1) == 356);
    REQUIRE(c(3,0) == 426);
    REQUIRE(c(3,1) == 484);
}

MMUL_TEST_CASE( "multiplication/mm_mul_4", "mmul") {
    etl::dyn_matrix<T> a(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<T> b(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<T> c(2,2);

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 7);
    REQUIRE(c(0,1) == 10);
    REQUIRE(c(1,0) == 15);
    REQUIRE(c(1,1) == 22);
}

MMUL_TEST_CASE( "multiplication/mm_mul_5", "mmul") {
    etl::dyn_matrix<T> a(3,3, std::initializer_list<T>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<T> b(3,3, std::initializer_list<T>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<T> c(3,3);

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

MMUL_TEST_CASE( "multiplication/mm_mul_6", "mmul") {
    etl::fast_matrix<T, 19, 19> a(etl::magic(19));
    etl::fast_matrix<T, 19, 19> b(etl::magic(19));
    etl::fast_matrix<T, 19, 19> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 828343);
    REQUIRE(c(1,1) == 825360);
    REQUIRE(c(2,2) == 826253);
    REQUIRE(c(3,3) == 824524);
    REQUIRE(c(18,18) == 828343);
}

//}}}

//{{{ Matrix vector and vector matrix multiplications

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_1", "auto_vmmul", Z, double, float) {
    etl::fast_matrix<Z, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_vector<Z, 3> b = {7,8,9};
    etl::fast_matrix<Z, 2> c;

    c = etl::mmul(a, b, c);

    REQUIRE(c(0) == 50);
    REQUIRE(c(1) == 122);
}

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_2", "auto_vmmul", Z, double, float) {
    etl::fast_matrix<Z, 2, 5> a = {1,2,3,4,5,6,7,8,9,10};
    etl::fast_vector<Z, 5> b = {7,8,9,10,11};
    etl::fast_matrix<Z, 2> c;

    c = etl::mmul(a, b);

    REQUIRE(c(0) == 145);
    REQUIRE(c(1) == 370);
}

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_3", "auto_vmmul", Z, double, float) {
    etl::fast_matrix<Z, 3, 2> a = {1,2,3,4,5,6};
    etl::fast_vector<Z, 3> b = {7,8,9};
    etl::fast_matrix<Z, 2> c;

    etl::force(etl::mmul(b, a, c));

    REQUIRE(c(0) == 76);
    REQUIRE(c(1) == 100);
}

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_4", "auto_vmmul", Z, double, float) {
    etl::dyn_matrix<Z> a(2,3, etl::values(1,2,3,4,5,6));
    etl::dyn_vector<Z> b(3, etl::values(7,8,9));
    etl::dyn_vector<Z> c(2);

    etl::force(etl::mmul(a, b, c));

    REQUIRE(c(0) == 50);
    REQUIRE(c(1) == 122);
}

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_5", "auto_vmmul", Z, double, float) {
    etl::dyn_matrix<Z> a(2, 5, etl::values(1,2,3,4,5,6,7,8,9,10));
    etl::dyn_vector<Z> b(5, etl::values(7,8,9,10,11));
    etl::dyn_vector<Z> c(2);

    etl::force(etl::mmul(a, b, c));

    REQUIRE(c(0) == 145);
    REQUIRE(c(1) == 370);
}

TEMPLATE_TEST_CASE_2( "multiplication/auto_vmmul_6", "auto_vmmul", Z, double, float) {
    etl::dyn_matrix<Z> a(3, 2, etl::values(1,2,3,4,5,6));
    etl::dyn_vector<Z> b(3, etl::values(7,8,9));
    etl::dyn_vector<Z> c(2);

    etl::force(etl::mmul(b, a, c));

    REQUIRE(c(0) == 76);
    REQUIRE(c(1) == 100);
}

//}}}

//Test using expressions directly
TEMPLATE_TEST_CASE_2( "multiplication/expression", "expression mmul", Z, double, float) {
    etl::fast_matrix<Z, 3, 3> a = {1,2,3,4,5,6,7,8,9};
    etl::fast_matrix<Z, 3, 3> b = {7,8,9,9,10,11,11,12,13};

    auto c = *etl::mmul(a, b);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

TEMPLATE_TEST_CASE_2( "multiplication/expr_mmul_1", "mmul", Z, double, float) {
    etl::dyn_matrix<Z> a(3,3, std::initializer_list<Z>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<Z> b(3,3, std::initializer_list<Z>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<Z> c(3,3);

    etl::force(etl::mmul(a + b - b, a + b - a, c));

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

TEMPLATE_TEST_CASE_2( "multiplication/expr_mmul_2", "mmul", Z, double, float) {
    etl::dyn_matrix<Z> a(3,3, std::initializer_list<Z>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<Z> b(3,3, std::initializer_list<Z>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<Z> c(3,3);

    etl::force(etl::mmul(abs(a), abs(b), c));

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

TEMPLATE_TEST_CASE_2( "multiplication/stop_mmul_1", "mmul", Z, double, float) {
    etl::dyn_matrix<Z> a(3,3, std::initializer_list<Z>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<Z> b(3,3, std::initializer_list<Z>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<Z> c(3,3);

    c = etl::mmul(s(abs(a)), s(abs(b)));

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(0,2) == 70);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
    REQUIRE(c(1,2) == 169);
    REQUIRE(c(2,0) == 220);
    REQUIRE(c(2,1) == 244);
    REQUIRE(c(2,2) == 268);
}

//}}}

//{{{ Expressions

TEMPLATE_TEST_CASE_2( "multiplication/expression_1", "expression", Z, double, float) {
    etl::dyn_matrix<Z> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<Z> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<Z> c(4,4);

    c = 2.0 * mmul(a,b) + mmul(a, b) / 1.1;

    REQUIRE(c(0,0) == Approx((2.0 + 1.0 / 1.1) * 90));
    REQUIRE(c(0,1) == Approx((2.0 + 1.0 / 1.1) * 100));
    REQUIRE(c(1,0) == Approx((2.0 + 1.0 / 1.1) * 202));
    REQUIRE(c(1,1) == Approx((2.0 + 1.0 / 1.1) * 228));
    REQUIRE(c(2,0) == Approx((2.0 + 1.0 / 1.1) * 314));
    REQUIRE(c(2,1) == Approx((2.0 + 1.0 / 1.1) * 356));
    REQUIRE(c(3,0) == Approx((2.0 + 1.0 / 1.1) * 426));
    REQUIRE(c(3,1) == Approx((2.0 + 1.0 / 1.1) * 484));
}

TEMPLATE_TEST_CASE_2( "multiplication/expression_2", "expression", Z, double, float) {
    etl::dyn_matrix<Z> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<Z> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<Z> c(4,4);

    c = 2.0 * etl::lazy_mmul(a,b) + etl::lazy_mmul(a, b) / 1.1;

    REQUIRE(c(0,0) == Approx((2.0 + 1.0 / 1.1) * 90));
    REQUIRE(c(0,1) == Approx((2.0 + 1.0 / 1.1) * 100));
    REQUIRE(c(1,0) == Approx((2.0 + 1.0 / 1.1) * 202));
    REQUIRE(c(1,1) == Approx((2.0 + 1.0 / 1.1) * 228));
    REQUIRE(c(2,0) == Approx((2.0 + 1.0 / 1.1) * 314));
    REQUIRE(c(2,1) == Approx((2.0 + 1.0 / 1.1) * 356));
    REQUIRE(c(3,0) == Approx((2.0 + 1.0 / 1.1) * 426));
    REQUIRE(c(3,1) == Approx((2.0 + 1.0 / 1.1) * 484));
}

//}}}
