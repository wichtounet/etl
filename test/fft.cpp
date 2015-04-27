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

//{{{ ifft_1d (complex)

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

//}}}

//{{{ ifft_1d (real)

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

//}}}

//{{{ fft_1d (real)

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

//}}}

//{{{ fft_2d (complex)

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

//}}}

//{{{ ifft_2d (complex)

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

//}}}
