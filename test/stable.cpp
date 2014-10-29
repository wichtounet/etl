//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/etl.hpp"

//{{{ Tests for rep

TEST_CASE( "rep/fast_matrix_1", "rep" ) {
    etl::fast_matrix<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<double, 3, 3> b(etl::rep<3>(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

TEST_CASE( "rep/fast_matrix_2", "rep" ) {
    etl::fast_matrix<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<double, 3, 3> b;

    b = etl::rep<3>(a);

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

TEST_CASE( "rep/fast_matrix_3", "rep" ) {
    etl::fast_matrix<double, 3> a({1.0, -2.0, 3.0});
    etl::fast_matrix<double, 3, 3, 2> b;

    b = etl::rep<3, 2>(a);

    REQUIRE(b(0,0,0) == 1.0);
    REQUIRE(b(0,0,1) == 1.0);
    REQUIRE(b(0,1,0) == 1.0);
    REQUIRE(b(0,1,1) == 1.0);
    REQUIRE(b(0,2,0) == 1.0);
    REQUIRE(b(0,2,1) == 1.0);
    REQUIRE(b(1,0,0) == -2.0);
    REQUIRE(b(1,0,1) == -2.0);
    REQUIRE(b(1,1,0) == -2.0);
    REQUIRE(b(1,1,1) == -2.0);
    REQUIRE(b(1,2,0) == -2.0);
    REQUIRE(b(1,2,1) == -2.0);
    REQUIRE(b(2,0,0) == 3.0);
    REQUIRE(b(2,0,1) == 3.0);
    REQUIRE(b(2,1,0) == 3.0);
    REQUIRE(b(2,1,1) == 3.0);
    REQUIRE(b(2,2,0) == 3.0);
    REQUIRE(b(2,2,1) == 3.0);
}

TEST_CASE( "rep/fast_matrix_4", "rep" ) {
    etl::fast_matrix<double, 1> a({1.0});
    etl::fast_matrix<double, 1, 3, 2, 5, 7> b;

    b = etl::rep<3, 2, 5, 7>(a);

    for(auto v : b){
        REQUIRE(v == 1.0);
    }
}

TEST_CASE( "rep/dyn_matrix_1", "rep" ) {
    etl::dyn_vector<double> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<double> b(etl::rep<3>(a));

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

TEST_CASE( "rep/dyn_matrix_2", "rep" ) {
    etl::dyn_vector<double> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<double> b(3,3);

    b = etl::rep<3>(a);

    REQUIRE(b(0,0) == 1.0);
    REQUIRE(b(0,1) == 1.0);
    REQUIRE(b(0,2) == 1.0);
    REQUIRE(b(1,0) == -2.0);
    REQUIRE(b(1,1) == -2.0);
    REQUIRE(b(1,2) == -2.0);
    REQUIRE(b(2,0) == 3.0);
    REQUIRE(b(2,1) == 3.0);
    REQUIRE(b(2,2) == 3.0);
}

TEST_CASE( "rep/dyn_matrix_3", "rep" ) {
    etl::dyn_vector<double> a(3, etl::values(1.0, -2.0, 3.0));
    etl::dyn_matrix<double, 3> b(3,3,2);

    b = etl::rep<3, 2>(a);

    REQUIRE(b(0,0,0) == 1.0);
    REQUIRE(b(0,0,1) == 1.0);
    REQUIRE(b(0,1,0) == 1.0);
    REQUIRE(b(0,1,1) == 1.0);
    REQUIRE(b(0,2,0) == 1.0);
    REQUIRE(b(0,2,1) == 1.0);
    REQUIRE(b(1,0,0) == -2.0);
    REQUIRE(b(1,0,1) == -2.0);
    REQUIRE(b(1,1,0) == -2.0);
    REQUIRE(b(1,1,1) == -2.0);
    REQUIRE(b(1,2,0) == -2.0);
    REQUIRE(b(1,2,1) == -2.0);
    REQUIRE(b(2,0,0) == 3.0);
    REQUIRE(b(2,0,1) == 3.0);
    REQUIRE(b(2,1,0) == 3.0);
    REQUIRE(b(2,1,1) == 3.0);
    REQUIRE(b(2,2,0) == 3.0);
    REQUIRE(b(2,2,1) == 3.0);
}

TEST_CASE( "rep/dyn_matrix_4", "rep" ) {
    etl::dyn_vector<double> a(1, 1.0);
    etl::dyn_matrix<double, 5> b(1, 3, 2, 5, 7);

    b = etl::rep<3, 2, 5, 7>(a);

    for(auto v : b){
        REQUIRE(v == 1.0);
    }
}

//}}}

//{{{ Tests for sum_r

TEST_CASE( "sum_r/fast_matrix_1", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b(etl::sum_r(a));

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEST_CASE( "sum_r/fast_matrix_2", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::sum_r(a);

    REQUIRE(b(0) == 10);
    REQUIRE(b(1) == 2);
    REQUIRE(b(2) == 22);
}

TEST_CASE( "sum_r/fast_matrix_3", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::sum_r(a) * 2.5;

    REQUIRE(b(0) == 25.0);
    REQUIRE(b(1) == 5.0);
    REQUIRE(b(2) == 55.0);
}

TEST_CASE( "sum_r/fast_matrix_4", "sum_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = (etl::sum_r(a) - etl::sum_r(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

//}}}

//{{{ Tests for mean_r

TEST_CASE( "mean_r/fast_matrix_1", "mean_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b(etl::mean_r(a));

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEST_CASE( "mean_r/fast_matrix_2", "mean_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEST_CASE( "mean_r/fast_matrix_3", "mean_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::mean_r(a) * 2.5;

    REQUIRE(b(0) == 6.25);
    REQUIRE(b(1) == 1.25);
    REQUIRE(b(2) == 13.75);
}

TEST_CASE( "mean_r/fast_matrix_4", "mean_r" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = (etl::mean_r(a) - etl::mean_r(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

TEST_CASE( "mean_r/fast_matrix_5", "mean_r" ) {
    etl::fast_matrix<double, 3, 4, 1, 1> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEST_CASE( "mean_r/fast_matrix_6", "mean_r" ) {
    etl::fast_matrix<double, 3, 2, 2> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 3> b;

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEST_CASE( "mean_r/dyn_matrix_1", "mean_r" ) {
    etl::dyn_matrix<double> a(3,4, etl::values(1,2,3,4,0,0,1,1,4,5,6,7));
    etl::dyn_vector<double> b(etl::mean_r(a));

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

TEST_CASE( "mean_r/dyn_matrix_2", "mean_r" ) {
    etl::dyn_matrix<double> a(3,4, etl::values(1,2,3,4,0,0,1,1,4,5,6,7));
    etl::dyn_vector<double> b(3);

    b = etl::mean_r(a);

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 0.5);
    REQUIRE(b(2) == 5.5);
}

//}}}

//{{{ Tests for mean_l

TEST_CASE( "mean_l/fast_matrix_1", "mean_l" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4, 0,0,1,1, 4,5,6,7});
    etl::fast_matrix<double, 4> b(etl::mean_l(a));

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEST_CASE( "mean_l/fast_matrix_2", "mean_l" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 4> b;

    b = etl::mean_l(a);

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEST_CASE( "mean_l/fast_matrix_3", "mean_l" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 4> b;

    b = etl::mean_l(a) * 2.5;

    REQUIRE(b(0) == Approx(4.1666666));
    REQUIRE(b(1) == Approx(5.8333333));
    REQUIRE(b(2) == Approx(8.333333));
}

TEST_CASE( "mean_l/fast_matrix_4", "mean_l" ) {
    etl::fast_matrix<double, 3, 4> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 4> b;

    b = (etl::mean_l(a) - etl::mean_l(a)) + 2.5;

    REQUIRE(b(0) == 2.5);
    REQUIRE(b(1) == 2.5);
    REQUIRE(b(2) == 2.5);
}

TEST_CASE( "mean_l/fast_matrix_5", "mean_l" ) {
    etl::fast_matrix<double, 3, 4, 1> a({1,2,3,4,0,0,1,1,4,5,6,7});
    etl::fast_matrix<double, 4, 1> b;

    b = etl::mean_l(a);

    REQUIRE(b(0,0) == Approx(1.666666));
    REQUIRE(b(1,0) == Approx(2.333333));
    REQUIRE(b(2,0) == Approx(3.333333));
}

TEST_CASE( "mean_l/fast_matrix_6", "mean_l" ) {
    etl::fast_matrix<double, 3, 4, 2> a({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24});
    etl::fast_matrix<double, 4, 2> b;

    b = etl::mean_l(a);

    REQUIRE(b(0,0) == Approx(9.0));
    REQUIRE(b(0,1) == Approx(10.0));
    REQUIRE(b(1,0) == Approx(11.0));
    REQUIRE(b(1,1) == Approx(12.0));
    REQUIRE(b(2,0) == Approx(13.0));
    REQUIRE(b(2,1) == Approx(14.0));
    REQUIRE(b(3,0) == Approx(15.0));
    REQUIRE(b(3,1) == Approx(16.0));
}

TEST_CASE( "mean_l/dyn_matrix_1", "mean_l" ) {
    etl::dyn_matrix<double> a(3,4, etl::values(1,2,3,4,0,0,1,1,4,5,6,7));
    etl::dyn_vector<double> b(etl::mean_l(a));

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

TEST_CASE( "mean_l/dyn_matrix_2", "mean_l" ) {
    etl::dyn_matrix<double> a(3,4, etl::values(1,2,3,4,0,0,1,1,4,5,6,7));
    etl::dyn_vector<double> b(4);

    b = etl::mean_l(a);

    REQUIRE(b(0) == Approx(1.666666));
    REQUIRE(b(1) == Approx(2.333333));
    REQUIRE(b(2) == Approx(3.333333));
}

//}}}
