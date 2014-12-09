//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/etl.hpp"
#include "etl/multiplication.hpp"
#include "etl/stop.hpp"

//{{{ mmul

TEST_CASE( "multiplication/mmul1", "mmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<double, 3, 2> b = {7,8,9,10,11,12};
    etl::fast_matrix<double, 2, 2> c;

    etl::mmul(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

TEST_CASE( "multiplication/mmul2", "mmul" ) {
    etl::fast_matrix<double, 3, 3> a = {1,2,3,4,5,6,7,8,9};
    etl::fast_matrix<double, 3, 3> b = {7,8,9,9,10,11,11,12,13};
    etl::fast_matrix<double, 3, 3> c;

    etl::mmul(a, b, c);

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

TEST_CASE( "multiplication/auto_vmmul_1", "auto_vmmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_vector<double, 3> b = {7,8,9};
    etl::fast_matrix<double, 2, 1> c;

    etl::auto_vmmul(a, b, c);

    REQUIRE(c(0,0) == 50);
    REQUIRE(c(1,0) == 122);
}

TEST_CASE( "multiplication/auto_vmmul_2", "auto_vmmul" ) {
    etl::fast_matrix<double, 2, 5> a = {1,2,3,4,5,6,7,8,9,10};
    etl::fast_vector<double, 5> b = {7,8,9,10,11};
    etl::fast_matrix<double, 2, 1> c;

    etl::auto_vmmul(a, b, c);

    REQUIRE(c(0,0) == 145);
    REQUIRE(c(1,0) == 370);
}

TEST_CASE( "multiplication/auto_vmmul_3", "auto_vmmul" ) {
    etl::fast_matrix<double, 3, 2> a = {1,2,3,4,5,6};
    etl::fast_vector<double, 3> b = {7,8,9};
    etl::fast_matrix<double, 1, 2> c;

    etl::auto_vmmul(b, a, c);

    REQUIRE(c(0,0) == 76);
    REQUIRE(c(0,1) == 100);
}

TEST_CASE( "multiplication/auto_vmmul_4", "auto_vmmul" ) {
    etl::dyn_matrix<double> a(2,3, etl::values(1,2,3,4,5,6));
    etl::dyn_vector<double> b(3, etl::values(7,8,9));
    etl::dyn_matrix<double> c(2,1);

    etl::auto_vmmul(a, b, c);

    REQUIRE(c(0,0) == 50);
    REQUIRE(c(1,0) == 122);
}

TEST_CASE( "multiplication/auto_vmmul_5", "auto_vmmul" ) {
    etl::dyn_matrix<double> a(2, 5, etl::values(1,2,3,4,5,6,7,8,9,10));
    etl::dyn_vector<double> b(5, etl::values(7,8,9,10,11));
    etl::dyn_matrix<double> c(2, 1);

    etl::auto_vmmul(a, b, c);

    REQUIRE(c(0,0) == 145);
    REQUIRE(c(1,0) == 370);
}

TEST_CASE( "multiplication/auto_vmmul_6", "auto_vmmul" ) {
    etl::dyn_matrix<double> a(3, 2, etl::values(1,2,3,4,5,6));
    etl::dyn_vector<double> b(3, etl::values(7,8,9));
    etl::dyn_matrix<double> c(1, 2);

    etl::auto_vmmul(b, a, c);

    REQUIRE(c(0,0) == 76);
    REQUIRE(c(0,1) == 100);
}

TEST_CASE( "multiplication/dyn_mmul", "mmul" ) {
    etl::dyn_matrix<double> a(3,3, std::initializer_list<double>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<double> b(3,3, std::initializer_list<double>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<double> c(3,3);

    etl::mmul(a, b, c);

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

TEST_CASE( "multiplication/expr_mmul_1", "mmul" ) {
    etl::dyn_matrix<double> a(3,3, std::initializer_list<double>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<double> b(3,3, std::initializer_list<double>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<double> c(3,3);

    etl::mmul(a + b - b, a + b - a, c);

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

TEST_CASE( "multiplication/expr_mmul_2", "mmul" ) {
    etl::dyn_matrix<double> a(3,3, std::initializer_list<double>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<double> b(3,3, std::initializer_list<double>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<double> c(3,3);

    etl::mmul(abs(a), abs(b), c);

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

TEST_CASE( "multiplication/stop_mmul_1", "mmul" ) {
    etl::dyn_matrix<double> a(3,3, std::initializer_list<double>({1,2,3,4,5,6,7,8,9}));
    etl::dyn_matrix<double> b(3,3, std::initializer_list<double>({7,8,9,9,10,11,11,12,13}));
    etl::dyn_matrix<double> c(3,3);

    etl::mmul(s(abs(a)), s(abs(b)), c);

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

TEST_CASE( "multiplication/mmul5", "mmul" ) {
    etl::dyn_matrix<double> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> c(4,4);

    etl::mmul(a, b, c);

    REQUIRE(c(0,0) == 90);
    REQUIRE(c(0,1) == 100);
    REQUIRE(c(1,0) == 202);
    REQUIRE(c(1,1) == 228);
    REQUIRE(c(2,0) == 314);
    REQUIRE(c(2,1) == 356);
    REQUIRE(c(3,0) == 426);
    REQUIRE(c(3,1) == 484);
}

TEST_CASE( "multiplication/mmul6", "mmul" ) {
    etl::dyn_matrix<double> a(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> b(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> c(2,2);

    etl::mmul(a, b, c);

    REQUIRE(c(0,0) == 7);
    REQUIRE(c(0,1) == 10);
    REQUIRE(c(1,0) == 15);
    REQUIRE(c(1,1) == 22);
}

//}}}

//{{{ mmul_2

TEST_CASE( "multiplication/mmul_2_1", "mmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<double, 3, 2> b = {7,8,9,10,11,12};
    etl::fast_matrix<double, 2, 2> c;

    c = etl::mmul(a, b);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

TEST_CASE( "multiplication/mmul_2_2", "mmul" ) {
    etl::fast_matrix<double, 3, 3> a = {1,2,3,4,5,6,7,8,9};
    etl::fast_matrix<double, 3, 3> b = {7,8,9,9,10,11,11,12,13};
    etl::fast_matrix<double, 3, 3> c;

    c = etl::mmul(a, b);

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

TEST_CASE( "multiplication/mmul_2_3", "mmul" ) {
    etl::dyn_matrix<double> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> c(4,4);

    c = etl::mmul(a, b);

    REQUIRE(c(0,0) == 90);
    REQUIRE(c(0,1) == 100);
    REQUIRE(c(1,0) == 202);
    REQUIRE(c(1,1) == 228);
    REQUIRE(c(2,0) == 314);
    REQUIRE(c(2,1) == 356);
    REQUIRE(c(3,0) == 426);
    REQUIRE(c(3,1) == 484);
}

TEST_CASE( "multiplication/mmul_2_4", "mmul" ) {
    etl::dyn_matrix<double> a(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> b(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> c(2,2);

    c = etl::mmul(a, b);

    REQUIRE(c(0,0) == 7);
    REQUIRE(c(0,1) == 10);
    REQUIRE(c(1,0) == 15);
    REQUIRE(c(1,1) == 22);
}

TEST_CASE( "multiplication/mmul_2_5", "mmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<double, 3, 2> b = {7,8,9,10,11,12};

    auto c = etl::mmul(a, b);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

//}}}

//{{{ mmul_2

TEST_CASE( "multiplication/lazy_mmul_1", "mmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<double, 3, 2> b = {7,8,9,10,11,12};
    etl::fast_matrix<double, 2, 2> c;

    c = etl::lazy_mmul(a, b);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

TEST_CASE( "multiplication/lazy_mmul_2", "mmul" ) {
    etl::fast_matrix<double, 3, 3> a = {1,2,3,4,5,6,7,8,9};
    etl::fast_matrix<double, 3, 3> b = {7,8,9,9,10,11,11,12,13};
    etl::fast_matrix<double, 3, 3> c;

    c = etl::lazy_mmul(a, b);

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

TEST_CASE( "multiplication/lazy_mmul_3", "mmul" ) {
    etl::dyn_matrix<double> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> c(4,4);

    c = etl::lazy_mmul(a, b);

    REQUIRE(c(0,0) == 90);
    REQUIRE(c(0,1) == 100);
    REQUIRE(c(1,0) == 202);
    REQUIRE(c(1,1) == 228);
    REQUIRE(c(2,0) == 314);
    REQUIRE(c(2,1) == 356);
    REQUIRE(c(3,0) == 426);
    REQUIRE(c(3,1) == 484);
}

TEST_CASE( "multiplication/lazy_mmul_4", "mmul" ) {
    etl::dyn_matrix<double> a(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> b(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> c(2,2);

    c = etl::lazy_mmul(a, b);

    REQUIRE(c(0,0) == 7);
    REQUIRE(c(0,1) == 10);
    REQUIRE(c(1,0) == 15);
    REQUIRE(c(1,1) == 22);
}

TEST_CASE( "multiplication/lazy_mmul_5", "mmul" ) {
    etl::fast_matrix<double, 2, 3> a = {1,2,3,4,5,6};
    etl::fast_matrix<double, 3, 2> b = {7,8,9,10,11,12};

    auto c = etl::lazy_mmul(a, b);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

//}}}

//{{{ Strassen mmul

TEST_CASE( "multiplication/strassen_mmul_1", "strassen_mmul" ) {
    etl::dyn_matrix<double> a(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> b(4,4, etl::values(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    etl::dyn_matrix<double> c(4,4);

    etl::strassen_mmul(a, b, c);

    REQUIRE(c(0,0) == 90);
    REQUIRE(c(0,1) == 100);
    REQUIRE(c(1,0) == 202);
    REQUIRE(c(1,1) == 228);
    REQUIRE(c(2,0) == 314);
    REQUIRE(c(2,1) == 356);
    REQUIRE(c(3,0) == 426);
    REQUIRE(c(3,1) == 484);
}

TEST_CASE( "multiplication/strassen_mmul_2", "strassen_mmul" ) {
    etl::dyn_matrix<double> a(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> b(2,2, etl::values(1,2,3,4,5,6,7,8));
    etl::dyn_matrix<double> c(2,2);

    etl::strassen_mmul(a, b, c);

    REQUIRE(c(0,0) == 7);
    REQUIRE(c(0,1) == 10);
    REQUIRE(c(1,0) == 15);
    REQUIRE(c(1,1) == 22);
}

TEST_CASE( "multiplication/strassen_mmul3", "strassen_mmul" ) {
    etl::dyn_matrix<double> a(3,3,etl::values(1,2,3,4,5,6,7,8,9));
    etl::dyn_matrix<double> b(3,3,etl::values(7,8,9,9,10,11,11,12,13));
    etl::dyn_matrix<double> c(3,3);

    etl::strassen_mmul(a, b, c);

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

TEST_CASE( "multiplication/strassen_mmul4", "mmul" ) {
    etl::dyn_matrix<double> a(2,3, etl::values(1,2,3,4,5,6));
    etl::dyn_matrix<double> b(3,2, etl::values(7,8,9,10,11,12));
    etl::dyn_matrix<double> c(2,2);

    etl::strassen_mmul(a, b, c);

    REQUIRE(c(0,0) == 58);
    REQUIRE(c(0,1) == 64);
    REQUIRE(c(1,0) == 139);
    REQUIRE(c(1,1) == 154);
}

//}}}
