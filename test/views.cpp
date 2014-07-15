#include "catch.hpp"

#include "etl/fast_matrix.hpp"
#include "etl/fast_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/dyn_vector.hpp"

TEST_CASE( "dim/fast_matrix_1", "dim<1>" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b(etl::dim<1>(a, 0));
    etl::fast_vector<double, 3> c(etl::dim<1>(a, 1));

    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/fast_matrix_2", "dim<2>" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 2> b(etl::dim<2>(a, 0));
    etl::fast_vector<double, 2> c(etl::dim<2>(a, 1));
    etl::fast_vector<double, 2> d(etl::dim<2>(a, 2));

    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/fast_matrix_3", "row" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b(etl::row(a, 0));
    etl::fast_vector<double, 3> c(etl::row(a, 1));

    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/fast_matrix_4", "col" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 2> b(etl::col(a, 0));
    etl::fast_vector<double, 2> c(etl::col(a, 1));
    etl::fast_vector<double, 2> d(etl::col(a, 2));

    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/dyn_matrix_1", "dim<1>" ) {
    etl::dyn_matrix<double> a(2, 3, {1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::dyn_vector<double> b(etl::dim<1>(a, 0));
    etl::dyn_vector<double> c(etl::dim<1>(a, 1));

    REQUIRE(!etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(!etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == -2.0);
    REQUIRE(b[2] == 4.0);

    REQUIRE(c[0] == 3.0);
    REQUIRE(c[1] == 0.5);
    REQUIRE(c[2] == -0.1);
}

TEST_CASE( "dim/dyn_matrix_2", "dim<2>" ) {
    etl::dyn_matrix<double> a(2, 3, {1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::dyn_vector<double> b(etl::dim<2>(a, 0));
    etl::dyn_vector<double> c(etl::dim<2>(a, 1));
    etl::dyn_vector<double> d(etl::dim<2>(a, 2));

    REQUIRE(!etl_traits<remove_cv_t<remove_reference_t<decltype(b)>>>::is_fast);
    REQUIRE(!etl_traits<remove_cv_t<remove_reference_t<decltype(c)>>>::is_fast);
    REQUIRE(!etl_traits<remove_cv_t<remove_reference_t<decltype(d)>>>::is_fast);

    REQUIRE(b[0] == 1.0);
    REQUIRE(b[1] == 3.0);

    REQUIRE(c[0] == -2.0);
    REQUIRE(c[1] == 0.5);

    REQUIRE(d[0] == 4.0);
    REQUIRE(d[1] == -0.1);
}

TEST_CASE( "dim/mix", "dim" ) {
    etl::fast_matrix<double, 2, 3> a({1.0, -2.0, 4.0, 3.0, 0.5, -0.1});
    etl::fast_vector<double, 3> b({0.1, 0.2, 0.3});
    etl::fast_vector<double, 3> c(b * row(a,1));

    REQUIRE(c[0] == Approx(0.3));
    REQUIRE(c[1] == Approx(0.1));
    REQUIRE(c[2] == Approx(-0.03));
}