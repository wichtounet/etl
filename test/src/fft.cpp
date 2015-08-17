//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#define MC(a,b) std::complex<Z>(a,b)

//fft_1d (real)

TEMPLATE_TEST_CASE_2( "fft_1d_r/0", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 8> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<std::complex<Z>, 8> c;

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(4.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(1.0)));
    REQUIRE(c(1).imag() == Approx(Z(-2.41421)));
    REQUIRE(c(2).real() == Approx(Z(0.0)));
    REQUIRE(c(2).imag() == Approx(Z(0.0)));
    REQUIRE(c(3).real() == Approx(Z(1.0)));
    REQUIRE(c(3).imag() == Approx(Z(-0.41421)));
    REQUIRE(c(4).real() == Approx(Z(0.0)));
    REQUIRE(c(4).imag() == Approx(Z(0.0)));
    REQUIRE(c(5).real() == Approx(Z(1.0)));
    REQUIRE(c(5).imag() == Approx(Z(0.41421)));
    REQUIRE(c(6).real() == Approx(Z(0.0)));
    REQUIRE(c(6).imag() == Approx(Z(0.0)));
    REQUIRE(c(7).real() == Approx(Z(1.0)));
    REQUIRE(c(7).imag() == Approx(Z(2.41421)));
}

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

TEMPLATE_TEST_CASE_2( "fft_1d_r/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 12> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0});
    etl::fast_matrix<std::complex<Z>, 12> c;

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(8.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(3.23205)));
    REQUIRE(c(1).imag() == Approx(Z(0.86603)));
    REQUIRE(c(2).real() == Approx(Z(-1.5)));
    REQUIRE(c(2).imag() == Approx(Z(-0.86603)));
    REQUIRE(c(3).real() == Approx(Z(0.0)));
    REQUIRE(c(3).imag() == Approx(Z(0.0)));
    REQUIRE(c(4).real() == Approx(Z(0.5)));
    REQUIRE(c(4).imag() == Approx(Z(0.86603)));
    REQUIRE(c(5).real() == Approx(Z(-0.23205)));
    REQUIRE(c(5).imag() == Approx(Z(-0.86603)));
    REQUIRE(c(6).real() == Approx(Z(0.0)));
    REQUIRE(c(6).imag() == Approx(Z(0.0)));

    REQUIRE(c(7).real() == Approx(Z(-0.23205)));
    REQUIRE(c(7).imag() == Approx(Z(0.86603)));
    REQUIRE(c(8).real() == Approx(Z(0.5)));
    REQUIRE(c(8).imag() == Approx(Z(-0.86603)));
    REQUIRE(c(9).real() == Approx(Z(0.0)));
    REQUIRE(c(9).imag() == Approx(Z(0.0)));

    REQUIRE(c(10).real() == Approx(Z(-1.5)));
    REQUIRE(c(10).imag() == Approx(Z(0.86603)));
    REQUIRE(c(11).real() == Approx(Z(3.23205)));
    REQUIRE(c(11).imag() == Approx(Z(-.86603)));
}

//fft_1d (complex)

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

TEMPLATE_TEST_CASE_2( "fft_1d_c/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 5> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);

    a.fft_inplace();

    REQUIRE(a(0).real() == Approx(Z(0.5)));
    REQUIRE(a(0).imag() == Approx(Z(5.5)));
    REQUIRE(a(1).real() == Approx(Z(5.626178)));
    REQUIRE(a(1).imag() == Approx(Z(0.376206)));
    REQUIRE(a(2).real() == Approx(Z(-1.067916)));
    REQUIRE(a(2).imag() == Approx(Z(-2.571198)));
    REQUIRE(a(3).real() == Approx(Z(-2.831271)));
    REQUIRE(a(3).imag() == Approx(Z(-2.709955)));
    REQUIRE(a(4).real() == Approx(Z(2.773009)));
    REQUIRE(a(4).imag() == Approx(Z(4.404947)));
}

//ifft_1d (complex)

TEMPLATE_TEST_CASE_2( "ifft_1d_c/0", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 4> a;
    etl::fast_matrix<std::complex<Z>, 4> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 2.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(3.0, -3.0);

    c = etl::ifft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(1.25)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(-0.75)));
    REQUIRE(c(1).imag() == Approx(Z(0.0)));
    REQUIRE(c(2).real() == Approx(Z(-1.25)));
    REQUIRE(c(2).imag() == Approx(Z(0.5)));
    REQUIRE(c(3).real() == Approx(Z(1.75)));
    REQUIRE(c(3).imag() == Approx(Z(0.5)));
}

TEMPLATE_TEST_CASE_2( "ifft_1d_c/1", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 5> a;
    etl::fast_matrix<std::complex<Z>, 5> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);

    c = etl::ifft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(0.1)));
    REQUIRE(c(0).imag() == Approx(Z(1.1)));
    REQUIRE(c(1).real() == Approx(Z(0.554602)));
    REQUIRE(c(1).imag() == Approx(Z(0.880989)));
    REQUIRE(c(2).real() == Approx(Z(-0.566254)));
    REQUIRE(c(2).imag() == Approx(Z(-0.541991)));
    REQUIRE(c(3).real() == Approx(Z(-0.213583)));
    REQUIRE(c(3).imag() == Approx(Z(-0.51424)));
    REQUIRE(c(4).real() == Approx(Z(1.125236)));
    REQUIRE(c(4).imag() == Approx(Z(0.075241)));
}

TEMPLATE_TEST_CASE_2( "ifft_1d_c/2", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 6> a;
    etl::fast_matrix<std::complex<Z>, 6> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);
    a[5] = std::complex<Z>(0.5, 0.5);

    c = etl::ifft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(0.166666)));
    REQUIRE(c(0).imag() == Approx(Z(1.0)));
    REQUIRE(c(1).real() == Approx(Z(0.605662)));
    REQUIRE(c(1).imag() == Approx(Z(0.333333)));
    REQUIRE(c(2).real() == Approx(Z(-0.910684)));
    REQUIRE(c(2).imag() == Approx(Z(0.183013)));
    REQUIRE(c(3).real() == Approx(Z(0)));
    REQUIRE(c(3).imag() == Approx(Z(-0.166667)));
    REQUIRE(c(4).real() == Approx(Z(0.244017)));
    REQUIRE(c(4).imag() == Approx(Z(-0.683013)));
    REQUIRE(c(5).real() == Approx(Z(0.894338)));
    REQUIRE(c(5).imag() == Approx(Z(0.333333)));
}

TEMPLATE_TEST_CASE_2( "ifft_1d_c/3", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 4> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 2.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(3.0, -3.0);

    a.ifft_inplace();

    REQUIRE(a(0).real() == Approx(Z(1.25)));
    REQUIRE(a(0).imag() == Approx(Z(0.0)));
    REQUIRE(a(1).real() == Approx(Z(-0.75)));
    REQUIRE(a(1).imag() == Approx(Z(0.0)));
    REQUIRE(a(2).real() == Approx(Z(-1.25)));
    REQUIRE(a(2).imag() == Approx(Z(0.5)));
    REQUIRE(a(3).real() == Approx(Z(1.75)));
    REQUIRE(a(3).imag() == Approx(Z(0.5)));
}

//ifft_1d (real)

TEMPLATE_TEST_CASE_2( "ifft_1d_real/1", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 5> a;
    etl::fast_matrix<Z, 5> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);

    c = etl::ifft_1d_real(a);

    REQUIRE(c(0) == Approx(Z(0.1)));
    REQUIRE(c(1) == Approx(Z(0.554602)));
    REQUIRE(c(2) == Approx(Z(-0.566254)));
    REQUIRE(c(3) == Approx(Z(-0.213583)));
    REQUIRE(c(4) == Approx(Z(1.125236)));
}

TEMPLATE_TEST_CASE_2( "ifft_1d_real/2", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 6> a;
    etl::fast_matrix<Z, 6> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);
    a[4] = std::complex<Z>(0.5, 1.5);
    a[5] = std::complex<Z>(0.5, 0.5);

    c = etl::ifft_1d_real(a);

    REQUIRE(c(0) == Approx(Z(0.166666)));
    REQUIRE(c(1) == Approx(Z(0.605662)));
    REQUIRE(c(2) == Approx(Z(-0.910684)));
    REQUIRE(c(3) == Approx(Z(0)));
    REQUIRE(c(4) == Approx(Z(0.244017)));
    REQUIRE(c(5) == Approx(Z(0.894338)));
}

//fft_2d (real)

TEMPLATE_TEST_CASE_2( "fft_2d_r/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    etl::fast_matrix<std::complex<Z>, 2, 3> c;

    c = etl::fft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(21.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1).real() == Approx(Z(-3.0)));
    REQUIRE(c(0,1).imag() == Approx(Z(1.73205)));
    REQUIRE(c(0,2).real() == Approx(Z(-3.0)));
    REQUIRE(c(0,2).imag() == Approx(Z(-1.73205)));
    REQUIRE(c(1,0).real() == Approx(Z(-9.0)));
    REQUIRE(c(1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1).real() == Approx(Z(0.0)));
    REQUIRE(c(1,1).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,2).real() == Approx(Z(0.0)));
    REQUIRE(c(1,2).imag() == Approx(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "fft_2d_r/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 3, 2> a({1.0, -2.0, 3.5, -4.0, 5.0, 6.5});
    etl::fast_matrix<std::complex<Z>, 3, 2> c;

    c = etl::fft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(10.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1).real() == Approx(Z(9.0)));
    REQUIRE(c(0,1).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,0).real() == Approx(Z(-6.5)));
    REQUIRE(c(1,0).imag() == Approx(Z(10.3923)));
    REQUIRE(c(1,1).real() == Approx(Z(0.0)));
    REQUIRE(c(1,1).imag() == Approx(Z(-7.7942)));
    REQUIRE(c(2,0).real() == Approx(Z(-6.5)));
    REQUIRE(c(2,0).imag() == Approx(Z(-10.39231)));
    REQUIRE(c(2,1).real() == Approx(Z(0.0)));
    REQUIRE(c(2,1).imag() == Approx(Z(7.7942)));
}

//fft_2d (complex)

TEMPLATE_TEST_CASE_2( "fft_2d_c/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;
    etl::fast_matrix<std::complex<Z>, 3, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    c = etl::fft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(10.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.25)));
    REQUIRE(c(0,1).real() == Approx(Z(9.0)));
    REQUIRE(c(0,1).imag() == Approx(Z(5.75)));
    REQUIRE(c(1,0).real() == Approx(Z(-10.1806)));
    REQUIRE(c(1,0).imag() == Approx(Z(11.7673)));
    REQUIRE(c(1,1).real() == Approx(Z(5.4127)));
    REQUIRE(c(1,1).imag() == Approx(Z(-9.1692)));
    REQUIRE(c(2,0).real() == Approx(Z(-2.8194)));
    REQUIRE(c(2,0).imag() == Approx(Z(-9.0173)));
    REQUIRE(c(2,1).real() == Approx(Z(-5.4127)));
    REQUIRE(c(2,1).imag() == Approx(Z(6.4192)));
}

TEMPLATE_TEST_CASE_2( "fft_2d_c/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2, 2> a;
    etl::fast_matrix<std::complex<Z>, 2, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);

    c = etl::fft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(4.0)));
    REQUIRE(c(0,1).real() == Approx(Z(0.0)));
    REQUIRE(c(0,1).imag() == Approx(Z(-2.0)));
    REQUIRE(c(1,0).real() == Approx(Z(6.0)));
    REQUIRE(c(1,0).imag() == Approx(Z(4.0)));
    REQUIRE(c(1,1).real() == Approx(Z(-2.0)));
    REQUIRE(c(1,1).imag() == Approx(Z(-2.0)));
}

TEMPLATE_TEST_CASE_2( "fft_2d_c/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    a.fft2_inplace();

    REQUIRE(a(0,0).real() == Approx(Z(10.0)));
    REQUIRE(a(0,0).imag() == Approx(Z(0.25)));
    REQUIRE(a(0,1).real() == Approx(Z(9.0)));
    REQUIRE(a(0,1).imag() == Approx(Z(5.75)));
    REQUIRE(a(1,0).real() == Approx(Z(-10.1806)));
    REQUIRE(a(1,0).imag() == Approx(Z(11.7673)));
    REQUIRE(a(1,1).real() == Approx(Z(5.4127)));
    REQUIRE(a(1,1).imag() == Approx(Z(-9.1692)));
    REQUIRE(a(2,0).real() == Approx(Z(-2.8194)));
    REQUIRE(a(2,0).imag() == Approx(Z(-9.0173)));
    REQUIRE(a(2,1).real() == Approx(Z(-5.4127)));
    REQUIRE(a(2,1).imag() == Approx(Z(6.4192)));
}

//ifft_2d (complex)

TEMPLATE_TEST_CASE_2( "ifft_2d_c/1", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;
    etl::fast_matrix<std::complex<Z>, 3, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    c = etl::ifft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(1.66667)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.04167)));
    REQUIRE(c(0,1).real() == Approx(Z(1.5)));
    REQUIRE(c(0,1).imag() == Approx(Z(0.95833)));
    REQUIRE(c(1,0).real() == Approx(Z(-0.4699)));
    REQUIRE(c(1,0).imag() == Approx(Z(-1.5029)));
    REQUIRE(c(1,1).real() == Approx(Z(-0.9021)));
    REQUIRE(c(1,1).imag() == Approx(Z(1.06987)));
    REQUIRE(c(2,0).real() == Approx(Z(-1.6968)));
    REQUIRE(c(2,0).imag() == Approx(Z(1.9612)));
    REQUIRE(c(2,1).real() == Approx(Z(0.9021)));
    REQUIRE(c(2,1).imag() == Approx(Z(-1.5282)));
}

TEMPLATE_TEST_CASE_2( "ifft_2d_c/2", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2, 2> a;
    etl::fast_matrix<std::complex<Z>, 2, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);

    c = etl::ifft_2d(a);

    REQUIRE(c(0,0).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(1.0)));
    REQUIRE(c(0,1).real() == Approx(Z(0.0)));
    REQUIRE(c(0,1).imag() == Approx(Z(-0.5)));
    REQUIRE(c(1,0).real() == Approx(Z(1.5)));
    REQUIRE(c(1,0).imag() == Approx(Z(1.0)));
    REQUIRE(c(1,1).real() == Approx(Z(-0.5)));
    REQUIRE(c(1,1).imag() == Approx(Z(-0.5)));
}

TEMPLATE_TEST_CASE_2( "ifft_2d_c/3", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    a.ifft2_inplace();

    REQUIRE(a(0,0).real() == Approx(Z(1.66667)));
    REQUIRE(a(0,0).imag() == Approx(Z(0.04167)));
    REQUIRE(a(0,1).real() == Approx(Z(1.5)));
    REQUIRE(a(0,1).imag() == Approx(Z(0.95833)));
    REQUIRE(a(1,0).real() == Approx(Z(-0.4699)));
    REQUIRE(a(1,0).imag() == Approx(Z(-1.5029)));
    REQUIRE(a(1,1).real() == Approx(Z(-0.9021)));
    REQUIRE(a(1,1).imag() == Approx(Z(1.06987)));
    REQUIRE(a(2,0).real() == Approx(Z(-1.6968)));
    REQUIRE(a(2,0).imag() == Approx(Z(1.9612)));
    REQUIRE(a(2,1).real() == Approx(Z(0.9021)));
    REQUIRE(a(2,1).imag() == Approx(Z(-1.5282)));
}

//ifft_2d (real)

TEMPLATE_TEST_CASE_2( "ifft_2d_c_real/1", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;
    etl::fast_matrix<Z, 3, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    c = etl::ifft_2d_real(a);

    REQUIRE(c(0,0) == Approx(Z(1.66667)));
    REQUIRE(c(0,1) == Approx(Z(1.5)));
    REQUIRE(c(1,0) == Approx(Z(-0.4699)));
    REQUIRE(c(1,1) == Approx(Z(-0.9021)));
    REQUIRE(c(2,0) == Approx(Z(-1.6968)));
    REQUIRE(c(2,1) == Approx(Z(0.9021)));
}

TEMPLATE_TEST_CASE_2( "ifft_2d_real_c/2", "[fast][ifft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2, 2> a;
    etl::fast_matrix<Z, 2, 2> c;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(-2.0, 0.0);

    c = etl::ifft_2d_real(a);

    REQUIRE(c(0,0) == Approx(Z(0.0)));
    REQUIRE(c(0,1) == Approx(Z(0.0)));
    REQUIRE(c(1,0) == Approx(Z(1.5)));
    REQUIRE(c(1,1) == Approx(Z(-0.5)));
}

/* fft_many tests */

TEMPLATE_TEST_CASE_2( "fft_1d_many/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 5> a({1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    etl::fast_matrix<std::complex<Z>, 2, 5> c;

    c = etl::fft_1d_many(a);

    REQUIRE(c(0,0).real() == Approx(Z(15.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(0,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(0,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(0,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,4).imag() == Approx(Z(-3.440955)));

    REQUIRE(c(1,0).real() == Approx(Z(15.0)));
    REQUIRE(c(1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(1,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(1,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(1,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,4).imag() == Approx(Z(-3.440955)));
}

TEMPLATE_TEST_CASE_2( "fft_1d_many/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 5> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    etl::fast_matrix<std::complex<Z>, 2, 5> c;

    c = etl::fft_1d_many(a);

    REQUIRE(c(0,0).real() == Approx(Z(15.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(0,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(0,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(0,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,4).imag() == Approx(Z(-3.440955)));

    REQUIRE(c(1,0).real() == Approx(Z(40.0)));
    REQUIRE(c(1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(1,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(1,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(1,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,4).imag() == Approx(Z(-3.440955)));
}

TEMPLATE_TEST_CASE_2( "fft_1d_many/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2, 5> a({MC(1.0,0.0), MC(2.0,0.0), MC(3.0,0.0), MC(4.0,0.0), MC(5.0,0.0), MC(6.0,0.0), MC(7.0,0.0), MC(8.0,0.0), MC(9.0,0.0), MC(10.0,0.0)});
    etl::fast_matrix<std::complex<Z>, 2, 5> c;

    c = etl::fft_1d_many(a);

    REQUIRE(c(0,0).real() == Approx(Z(15.0)));
    REQUIRE(c(0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(0,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(0,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(0,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(0,4).imag() == Approx(Z(-3.440955)));

    REQUIRE(c(1,0).real() == Approx(Z(40.0)));
    REQUIRE(c(1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(1,2).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(1,3).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(1,4).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,4).imag() == Approx(Z(-3.440955)));
}

//fft_2d_many

TEMPLATE_TEST_CASE_2( "fft_2d_many/0", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2, 2> a({1.0, 2.0, -1.0, -2.0, 1.0, 2.0, -1.0, 0.0});
    etl::fast_matrix<std::complex<Z>, 2, 2, 2> c;

    c = etl::fft_2d_many(a);

    REQUIRE(c(0,0,0).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,0,1).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0,1).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1,0).real() == Approx(Z(6.0)));
    REQUIRE(c(0,1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(0,1,1).real() == Approx(Z(-2.0)));
    REQUIRE(c(0,1,1).imag() == Approx(Z(0.0)));

    REQUIRE(c(1,0,0).real() == Approx(Z(2.0)));
    REQUIRE(c(1,0,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,0,1).real() == Approx(Z(-2.0)));
    REQUIRE(c(1,0,1).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1,0).real() == Approx(Z(4.0)));
    REQUIRE(c(1,1,0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1,1,1).real() == Approx(Z(0.0)));
    REQUIRE(c(1,1,1).imag() == Approx(Z(0.0)));
}

TEMPLATE_TEST_CASE_2( "fft_2d_many/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2, 2, 2> a({
        MC(1.0, 1.0), MC(2.0, 3.0), MC(-1.0, 0.0), MC(-2.0, 0.0),
        MC(-1.0, 1.0), MC(2.0, 1.0), MC(-0.5, 0.0), MC(-1.0, 1.0)});
    etl::fast_matrix<std::complex<Z>, 2, 2, 2> c;

    c = etl::fft_2d_many(a);

    REQUIRE(c(0,0,0).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0,0).imag() == Approx(Z(4.0)));
    REQUIRE(c(0,0,1).real() == Approx(Z(0.0)));
    REQUIRE(c(0,0,1).imag() == Approx(Z(-2.0)));
    REQUIRE(c(0,1,0).real() == Approx(Z(6.0)));
    REQUIRE(c(0,1,0).imag() == Approx(Z(4.0)));
    REQUIRE(c(0,1,1).real() == Approx(Z(-2.0)));
    REQUIRE(c(0,1,1).imag() == Approx(Z(-2.0)));

    REQUIRE(c(1,0,0).real() == Approx(Z(-0.5)));
    REQUIRE(c(1,0,0).imag() == Approx(Z(3.0)));
    REQUIRE(c(1,0,1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1,0,1).imag() == Approx(Z(-1.0)));
    REQUIRE(c(1,1,0).real() == Approx(Z(2.5)));
    REQUIRE(c(1,1,0).imag() == Approx(Z(1.0)));
    REQUIRE(c(1,1,1).real() == Approx(Z(-3.5)));
    REQUIRE(c(1,1,1).imag() == Approx(Z(1.0)));
}
