//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

//{{{ fft_1d (real)

TEMPLATE_TEST_CASE_2( "fft_1d_r/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 5> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::fast_matrix<std::complex<Z>, 5> c;

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(15.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(2).real() == Approx(Z(-2.5)));
    REQUIRE(c(2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(3).real() == Approx(Z(-2.5)));
    REQUIRE(c(3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(4).real() == Approx(Z(-2.5)));
    REQUIRE(c(4).imag() == Approx(Z(-3.440955)));
}

TEMPLATE_TEST_CASE_2( "fft_1d_r/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 6> a({0.5, 1.5, 3.5, -1.5, 3.9, -5.5});
    etl::fast_matrix<std::complex<Z>, 6> c;

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(2.4)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(-3.7)));
    REQUIRE(c(1).imag() == Approx(Z(-5.71577)));
    REQUIRE(c(2).real() == Approx(Z(-2.7)));
    REQUIRE(c(2).imag() == Approx(Z(-6.40859)));
    REQUIRE(c(3).real() == Approx(Z(13.4)));
    REQUIRE(c(3).imag() == Approx(Z(0)));
    REQUIRE(c(4).real() == Approx(Z(-2.7)));
    REQUIRE(c(4).imag() == Approx(Z(6.40859)));
    REQUIRE(c(5).real() == Approx(Z(-3.7)));
    REQUIRE(c(5).imag() == Approx(Z(5.71577)));
}

//}}}

//{{{ fft_1d (complex)

TEMPLATE_TEST_CASE_2( "fft_1d_c/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 5> a;
    etl::fast_matrix<std::complex<Z>, 5> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(0.5)));
    REQUIRE(c(0).imag() == Approx(Z(5.5)));
    REQUIRE(c(1).real() == Approx(Z(5.626178)));
    REQUIRE(c(1).imag() == Approx(Z(0.376206)));
    REQUIRE(c(2).real() == Approx(Z(-1.067916)));
    REQUIRE(c(2).imag() == Approx(Z(-2.571198)));
    REQUIRE(c(3).real() == Approx(Z(-2.831271)));
    REQUIRE(c(3).imag() == Approx(Z(-2.709955)));
    REQUIRE(c(4).real() == Approx(Z(2.773009)));
    REQUIRE(c(4).imag() == Approx(Z(4.404947)));
}

TEMPLATE_TEST_CASE_2( "fft_1d_c/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 6> a;
    etl::fast_matrix<std::complex<Z>, 6> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);
    a[5] = std::complex<Z>(0.5, 0.5);

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(1.0)));
    REQUIRE(c(0).imag() == Approx(Z(6.0)));
    REQUIRE(c(1).real() == Approx(Z(5.366025)));
    REQUIRE(c(1).imag() == Approx(Z(2.0)));
    REQUIRE(c(2).real() == Approx(Z(1.464102)));
    REQUIRE(c(2).imag() == Approx(Z(-4.098076)));
    REQUIRE(c(3).real() == Approx(Z(0)));
    REQUIRE(c(3).imag() == Approx(Z(-1.0)));
    REQUIRE(c(4).real() == Approx(Z(-5.464102)));
    REQUIRE(c(4).imag() == Approx(Z(1.098076)));
    REQUIRE(c(5).real() == Approx(Z(3.633975)));
    REQUIRE(c(5).imag() == Approx(Z(2.0)));
}

//}}}
