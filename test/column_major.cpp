//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <cmath>

#include "catch.hpp"
#include "template_test.hpp"
#include "mmul_test.hpp"
#include "conv_test.hpp"

TEMPLATE_TEST_CASE_2( "column_major/1", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 2, 3> test_matrix(0);

    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix.template dim<0>() == 2);
    REQUIRE(test_matrix.template dim<1>() == 3);
    REQUIRE(test_matrix.dim(0) == 2);
    REQUIRE(test_matrix.dim(1) == 3);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0) == 1);
    REQUIRE(test_matrix(0, 1) == 3);
    REQUIRE(test_matrix(0, 2) == 5);
    REQUIRE(test_matrix(1, 0) == 2);
    REQUIRE(test_matrix(1, 1) == 4);
    REQUIRE(test_matrix(1, 2) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/2", "[dyn][cm]", Z, int, long ) {
    etl::dyn_matrix_cm<Z> test_matrix(2, 3);

    test_matrix = 0;

    REQUIRE(test_matrix.size() == 6);

    REQUIRE(test_matrix.template dim<0>() == 2);
    REQUIRE(test_matrix.template dim<1>() == 3);
    REQUIRE(test_matrix.dim(0) == 2);
    REQUIRE(test_matrix.dim(1) == 3);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0) == 1);
    REQUIRE(test_matrix(0, 1) == 3);
    REQUIRE(test_matrix(0, 2) == 5);
    REQUIRE(test_matrix(1, 0) == 2);
    REQUIRE(test_matrix(1, 1) == 4);
    REQUIRE(test_matrix(1, 2) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/3", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 2, 3, 4> test_matrix;

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0, 0) == 1);
    REQUIRE(test_matrix(0, 0, 1) == 7);
    REQUIRE(test_matrix(0, 0, 2) == 13);
    REQUIRE(test_matrix(0, 0, 3) == 19);
    REQUIRE(test_matrix(0, 1, 0) == 3);
    REQUIRE(test_matrix(0, 1, 1) == 9);
    REQUIRE(test_matrix(0, 1, 2) == 15);
    REQUIRE(test_matrix(0, 1, 3) == 21);
    REQUIRE(test_matrix(0, 2, 0) == 5);
    REQUIRE(test_matrix(0, 2, 1) == 11);
    REQUIRE(test_matrix(0, 2, 2) == 17);
    REQUIRE(test_matrix(0, 2, 3) == 23);
    REQUIRE(test_matrix(1, 0, 0) == 2);
    REQUIRE(test_matrix(1, 0, 1) == 8);
    REQUIRE(test_matrix(1, 0, 2) == 14);
    REQUIRE(test_matrix(1, 0, 3) == 20);
    REQUIRE(test_matrix(1, 1, 0) == 4);
    REQUIRE(test_matrix(1, 1, 1) == 10);
    REQUIRE(test_matrix(1, 1, 2) == 16);
    REQUIRE(test_matrix(1, 1, 3) == 22);
    REQUIRE(test_matrix(1, 2, 0) == 6);
    REQUIRE(test_matrix(1, 2, 1) == 12);
    REQUIRE(test_matrix(1, 2, 2) == 18);
    REQUIRE(test_matrix(1, 2, 3) == 24);
}

TEMPLATE_TEST_CASE_2( "column_major/4", "[dyn][cm]", Z, int, long ) {
    etl::dyn_matrix_cm<Z, 3> test_matrix(2, 3, 4);

    for(std::size_t i = 0; i < test_matrix.size(); ++i){
        test_matrix[i] = i+1;
    }

    REQUIRE(test_matrix(0, 0, 0) == 1);
    REQUIRE(test_matrix(0, 0, 1) == 7);
    REQUIRE(test_matrix(0, 0, 2) == 13);
    REQUIRE(test_matrix(0, 0, 3) == 19);
    REQUIRE(test_matrix(0, 1, 0) == 3);
    REQUIRE(test_matrix(0, 1, 1) == 9);
    REQUIRE(test_matrix(0, 1, 2) == 15);
    REQUIRE(test_matrix(0, 1, 3) == 21);
    REQUIRE(test_matrix(0, 2, 0) == 5);
    REQUIRE(test_matrix(0, 2, 1) == 11);
    REQUIRE(test_matrix(0, 2, 2) == 17);
    REQUIRE(test_matrix(0, 2, 3) == 23);
    REQUIRE(test_matrix(1, 0, 0) == 2);
    REQUIRE(test_matrix(1, 0, 1) == 8);
    REQUIRE(test_matrix(1, 0, 2) == 14);
    REQUIRE(test_matrix(1, 0, 3) == 20);
    REQUIRE(test_matrix(1, 1, 0) == 4);
    REQUIRE(test_matrix(1, 1, 1) == 10);
    REQUIRE(test_matrix(1, 1, 2) == 16);
    REQUIRE(test_matrix(1, 1, 3) == 22);
    REQUIRE(test_matrix(1, 2, 0) == 6);
    REQUIRE(test_matrix(1, 2, 1) == 12);
    REQUIRE(test_matrix(1, 2, 2) == 18);
    REQUIRE(test_matrix(1, 2, 3) == 24);
}

TEMPLATE_TEST_CASE_2( "column_major/transpose/1", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 2, 3> a(etl::sequence_generator(1));

    REQUIRE(a(0, 0) == 1);
    REQUIRE(a(0, 1) == 3);
    REQUIRE(a(0, 2) == 5);
    REQUIRE(a(1, 0) == 2);
    REQUIRE(a(1, 1) == 4);
    REQUIRE(a(1, 2) == 6);

    etl::fast_matrix_cm<Z, 3, 2> b(etl::transpose(a));

    REQUIRE(b(0, 0) == 1);
    REQUIRE(b(1, 0) == 3);
    REQUIRE(b(2, 0) == 5);
    REQUIRE(b(0, 1) == 2);
    REQUIRE(b(1, 1) == 4);
    REQUIRE(b(2, 1) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/tranpose/2", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 3, 2> a(etl::sequence_generator(1));

    REQUIRE(a(0, 0) == 1);
    REQUIRE(a(0, 1) == 4);
    REQUIRE(a(1, 0) == 2);
    REQUIRE(a(1, 1) == 5);
    REQUIRE(a(2, 0) == 3);
    REQUIRE(a(2, 1) == 6);

    etl::fast_matrix_cm<Z, 2, 3> b(etl::transpose(a));

    REQUIRE(b(0, 0) == 1);
    REQUIRE(b(0, 1) == 2);
    REQUIRE(b(0, 2) == 3);
    REQUIRE(b(1, 0) == 4);
    REQUIRE(b(1, 1) == 5);
    REQUIRE(b(1, 2) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/binary/1", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 3, 2> a(etl::sequence_generator(1));
    etl::fast_matrix_cm<Z, 3, 2> b(a + a - a + a);

    REQUIRE(a(0, 0) == 1);
    REQUIRE(a(0, 1) == 4);
    REQUIRE(a(1, 0) == 2);
    REQUIRE(a(1, 1) == 5);
    REQUIRE(a(2, 0) == 3);
    REQUIRE(a(2, 1) == 6);
}

TEMPLATE_TEST_CASE_2( "column_major/sub/1", "[fast][cm]", Z, int, long ) {
    etl::fast_matrix_cm<Z, 3, 3, 2> a;

    a(0) = etl::sequence_generator(1);
    a(1) = etl::sequence_generator(7);
    a(2) = etl::sequence_generator(13);

    etl::fast_matrix_cm<Z, 3, 2> b(a(1));

    REQUIRE(a(0)(0, 0) == 1);
    REQUIRE(a(0)(0, 1) == 4);
    REQUIRE(a(0)(1, 0) == 2);
    REQUIRE(a(0)(1, 1) == 5);
    REQUIRE(a(0)(2, 0) == 3);
    REQUIRE(a(0)(2, 1) == 6);

    REQUIRE(b(0, 0) == 7);
    REQUIRE(b(0, 1) == 10);
    REQUIRE(b(1, 0) == 8);
    REQUIRE(b(1, 1) == 11);
    REQUIRE(b(2, 0) == 9);
    REQUIRE(b(2, 1) == 12);
}

MMUL_TEST_CASE( "column_major/mul/1", "mmul") {
    etl::fast_matrix_cm<T, 2, 3> a = {1,4,2,5,3,6};
    etl::fast_matrix_cm<T, 3, 2> b = {7,9,11,8,10,12};
    etl::fast_matrix_cm<T, 2, 2> c;

    REQUIRE(etl::rows(a) == 2);
    REQUIRE(etl::columns(a) == 3);
    REQUIRE(etl::rows(b) == 3);
    REQUIRE(etl::columns(b) == 2);

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

TEMPLATE_TEST_CASE_2( "column_major/vmul/1", "vmmul", Z, double, float) {
    etl::fast_matrix_cm<Z, 2, 3> a = {1,4,2,5,3,6};
    etl::fast_vector_cm<Z, 3> b = {7,8,9};
    etl::fast_matrix_cm<Z, 2> c;

    c = etl::mul(a, b, c);

    REQUIRE(c(0) == 50);
    REQUIRE(c(1) == 122);
}

CONV1_FULL_TEST_CASE( "column_major/conv/full_1", "[cm][conv]" ) {
    etl::fast_vector_cm<T, 3> a = {1.0, 2.0, 3.0};
    etl::fast_vector_cm<T, 3> b = {0.0, 1.0, 0.5};
    etl::fast_vector_cm<T, 5> c;

    Impl::apply(a, b, c);

    REQUIRE(c[0] == Approx(0.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(2.5));
    REQUIRE(c[3] == Approx(4.0));
    REQUIRE(c[4] == Approx(1.5));
}

CONV2_FULL_TEST_CASE( "column_major/conv2/full_1", "[cm][conv2]" ) {
    etl::fast_matrix_cm<T, 3, 3> a = {1.0, 0.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0};
    etl::fast_matrix_cm<T, 2, 2> b = {2.0, 0.5, 0.0, 0.5};
    etl::fast_matrix_cm<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0,0) == Approx(T(2.0)));
    REQUIRE(c(0,1) == Approx(T(4.0)));
    REQUIRE(c(0,2) == Approx(T(6.0)));
    REQUIRE(c(0,3) == Approx(T(0.0)));

    REQUIRE(c(1,0) == Approx(T(0.5)));
    REQUIRE(c(1,1) == Approx(T(3.5)));
    REQUIRE(c(1,2) == Approx(T(4.5)));
    REQUIRE(c(1,3) == Approx(T(1.5)));

    REQUIRE(c(2,0) == Approx(T(6.0)));
    REQUIRE(c(2,1) == Approx(T(4.5)));
    REQUIRE(c(2,2) == Approx(T(3.0)));
    REQUIRE(c(2,3) == Approx(T(0.5)));

    REQUIRE(c(3,0) == Approx(T(1.5)));
    REQUIRE(c(3,1) == Approx(T(2.5)));
    REQUIRE(c(3,2) == Approx(T(1.5)));
    REQUIRE(c(3,3) == Approx(T(0.5)));
}

CONV2_FULL_TEST_CASE( "column_major/conv2/full_2", "[cm][conv2]" ) {
    //TODO magic(X) is not compatible with column major matrices

    etl::fast_matrix<T, 3, 3> aa(etl::magic(3));
    etl::fast_matrix<T, 2, 2> bb(etl::magic(2));

    etl::fast_matrix_cm<T, 3, 3> a(aa);
    etl::fast_matrix_cm<T, 2, 2> b(bb);
    etl::fast_matrix_cm<T, 4, 4> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(8)));
    REQUIRE(c(0, 1) == Approx(T(25)));
    REQUIRE(c(0, 2) == Approx(T(9)));
    REQUIRE(c(0, 3) == Approx(T(18)));

    REQUIRE(c(1, 0) == Approx(T(35)));
    REQUIRE(c(1, 1) == Approx(T(34)));
    REQUIRE(c(1, 2) == Approx(T(48)));
    REQUIRE(c(1, 3) == Approx(T(33)));

    REQUIRE(c(2, 0) == Approx(T(16)));
    REQUIRE(c(2, 1) == Approx(T(47)));
    REQUIRE(c(2, 2) == Approx(T(67)));
    REQUIRE(c(2, 3) == Approx(T(20)));

    REQUIRE(c(3, 0) == Approx(T(16)));
    REQUIRE(c(3, 1) == Approx(T(44)));
    REQUIRE(c(3, 2) == Approx(T(26)));
    REQUIRE(c(3, 3) == Approx(T(4)));
}

CONV2_FULL_TEST_CASE( "column_major/conv2/full_3", "[cm][conv2]" ) {
    etl::fast_matrix_cm<T, 2, 6> a = {1,7,2,8,3,9,4,10,5,11,6,12};
    etl::fast_matrix_cm<T, 2, 2> b = {1,3,2,4};
    etl::fast_matrix_cm<T, 3, 7> c;

    Impl::apply(a, b, c);

    REQUIRE(c(0, 0) == Approx(T(1)));
    REQUIRE(c(0, 1) == Approx(T(4)));
    REQUIRE(c(0, 2) == Approx(T(7)));
    REQUIRE(c(0, 3) == Approx(T(10)));
    REQUIRE(c(0, 4) == Approx(T(13)));
    REQUIRE(c(0, 5) == Approx(T(16)));
    REQUIRE(c(0, 6) == Approx(T(12)));

    REQUIRE(c(1, 0) == Approx(T(10)));
    REQUIRE(c(1, 1) == Approx(T(32)));
    REQUIRE(c(1, 2) == Approx(T(42)));
    REQUIRE(c(1, 3) == Approx(T(52)));
    REQUIRE(c(1, 4) == Approx(T(62)));
    REQUIRE(c(1, 5) == Approx(T(72)));
    REQUIRE(c(1, 6) == Approx(T(48)));

    REQUIRE(c(2, 0) == Approx(T(21)));
    REQUIRE(c(2, 1) == Approx(T(52)));
    REQUIRE(c(2, 2) == Approx(T(59)));
    REQUIRE(c(2, 3) == Approx(T(66)));
    REQUIRE(c(2, 4) == Approx(T(73)));
    REQUIRE(c(2, 5) == Approx(T(80)));
    REQUIRE(c(2, 6) == Approx(T(48)));
}
