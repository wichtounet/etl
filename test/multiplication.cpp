#include "catch.hpp"

#include "etl/fast_matrix.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/multiplication.hpp"

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

TEST_CASE( "multiplication/dyn_mmul", "mmul" ) {
    etl::dyn_matrix<double> a(3,3,{1,2,3,4,5,6,7,8,9});
    etl::dyn_matrix<double> b(3,3,{7,8,9,9,10,11,11,12,13});
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