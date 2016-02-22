//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "catch_complex_approx.hpp"
#include "fft_test.hpp"

#define MC(a, b) std::complex<T>(a, b)
#define MZ(a, b) std::complex<Z>(a, b)

//fft_2d (real)

FFT2_TEST_CASE("fft_2d_r/1", "[fast][fft]") {
    etl::fast_matrix<T, 2, 3> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    etl::fast_matrix<std::complex<T>, 2, 3> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(21.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(-3.0)));
    REQUIRE(c(0, 1).imag() == Approx(T(1.73205)));
    REQUIRE(c(0, 2).real() == Approx(T(-3.0)));
    REQUIRE(c(0, 2).imag() == Approx(T(-1.73205)));
    REQUIRE(c(1, 0).real() == Approx(T(-9.0)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(0.0)));
    REQUIRE(c(1, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 2).real() == Approx(T(0.0)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.0)));
}

FFT2_TEST_CASE("fft_2d_r/2", "[fast][fft]") {
    etl::fast_matrix<T, 3, 2> a({1.0, -2.0, 3.5, -4.0, 5.0, 6.5});
    etl::fast_matrix<std::complex<T>, 3, 2> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(10.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(9.0)));
    REQUIRE(c(0, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 0).real() == Approx(T(-6.5)));
    REQUIRE(c(1, 0).imag() == Approx(T(10.3923)));
    REQUIRE(c(1, 1).real() == Approx(T(0.0)));
    REQUIRE(c(1, 1).imag() == Approx(T(-7.7942)));
    REQUIRE(c(2, 0).real() == Approx(T(-6.5)));
    REQUIRE(c(2, 0).imag() == Approx(T(-10.39231)));
    REQUIRE(c(2, 1).real() == Approx(T(0.0)));
    REQUIRE(c(2, 1).imag() == Approx(T(7.7942)));
}

//fft_2d (complex)

FFT2_TEST_CASE("fft_2d_c/1", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 3, 2> a;
    etl::fast_matrix<std::complex<T>, 3, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(10.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.25)));
    REQUIRE(c(0, 1).real() == Approx(T(9.0)));
    REQUIRE(c(0, 1).imag() == Approx(T(5.75)));
    REQUIRE(c(1, 0).real() == Approx(T(-10.1806)));
    REQUIRE(c(1, 0).imag() == Approx(T(11.7673)));
    REQUIRE(c(1, 1).real() == Approx(T(5.4127)));
    REQUIRE(c(1, 1).imag() == Approx(T(-9.1692)));
    REQUIRE(c(2, 0).real() == Approx(T(-2.8194)));
    REQUIRE(c(2, 0).imag() == Approx(T(-9.0173)));
    REQUIRE(c(2, 1).real() == Approx(T(-5.4127)));
    REQUIRE(c(2, 1).imag() == Approx(T(6.4192)));
}

FFT2_TEST_CASE("fft_2d_c/2", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2, 2> a;
    etl::fast_matrix<std::complex<T>, 2, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(4.0)));
    REQUIRE(c(0, 1).real() == Approx(T(0.0)));
    REQUIRE(c(0, 1).imag() == Approx(T(-2.0)));
    REQUIRE(c(1, 0).real() == Approx(T(6.0)));
    REQUIRE(c(1, 0).imag() == Approx(T(4.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-2.0)));
    REQUIRE(c(1, 1).imag() == Approx(T(-2.0)));
}

FFT2_TEST_CASE("fft_2d_c/4", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 3, 2> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, a);

    REQUIRE(a(0, 0).real() == Approx(T(10.0)));
    REQUIRE(a(0, 0).imag() == Approx(T(0.25)));
    REQUIRE(a(0, 1).real() == Approx(T(9.0)));
    REQUIRE(a(0, 1).imag() == Approx(T(5.75)));
    REQUIRE(a(1, 0).real() == Approx(T(-10.1806)));
    REQUIRE(a(1, 0).imag() == Approx(T(11.7673)));
    REQUIRE(a(1, 1).real() == Approx(T(5.4127)));
    REQUIRE(a(1, 1).imag() == Approx(T(-9.1692)));
    REQUIRE(a(2, 0).real() == Approx(T(-2.8194)));
    REQUIRE(a(2, 0).imag() == Approx(T(-9.0173)));
    REQUIRE(a(2, 1).real() == Approx(T(-5.4127)));
    REQUIRE(a(2, 1).imag() == Approx(T(6.4192)));
}

//fft_2d_many

FFT2_MANY_TEST_CASE("fft_2d_many/0", "[fast][fft]") {
    etl::fast_matrix<T, 2, 2, 2> a({1.0, 2.0, -1.0, -2.0, 1.0, 2.0, -1.0, 0.0});
    etl::fast_matrix<std::complex<T>, 2, 2, 2> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0, 0).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 0, 1).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1, 0).real() == Approx(T(6.0)));
    REQUIRE(c(0, 1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1, 1).real() == Approx(T(-2.0)));
    REQUIRE(c(0, 1, 1).imag() == Approx(T(0.0)));

    REQUIRE(c(1, 0, 0).real() == Approx(T(2.0)));
    REQUIRE(c(1, 0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 0, 1).real() == Approx(T(-2.0)));
    REQUIRE(c(1, 0, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1, 0).real() == Approx(T(4.0)));
    REQUIRE(c(1, 1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1, 1).real() == Approx(T(0.0)));
    REQUIRE(c(1, 1, 1).imag() == Approx(T(0.0)));
}

FFT2_MANY_TEST_CASE("fft_2d_many/1", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2, 2, 2> a({MC(1.0, 1.0), MC(2.0, 3.0), MC(-1.0, 0.0), MC(-2.0, 0.0),
                                                  MC(-1.0, 1.0), MC(2.0, 1.0), MC(-0.5, 0.0), MC(-1.0, 1.0)});
    etl::fast_matrix<std::complex<T>, 2, 2, 2> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0, 0).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0, 0).imag() == Approx(T(4.0)));
    REQUIRE(c(0, 0, 1).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0, 1).imag() == Approx(T(-2.0)));
    REQUIRE(c(0, 1, 0).real() == Approx(T(6.0)));
    REQUIRE(c(0, 1, 0).imag() == Approx(T(4.0)));
    REQUIRE(c(0, 1, 1).real() == Approx(T(-2.0)));
    REQUIRE(c(0, 1, 1).imag() == Approx(T(-2.0)));

    REQUIRE(c(1, 0, 0).real() == Approx(T(-0.5)));
    REQUIRE(c(1, 0, 0).imag() == Approx(T(3.0)));
    REQUIRE(c(1, 0, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 0, 1).imag() == Approx(T(-1.0)));
    REQUIRE(c(1, 1, 0).real() == Approx(T(2.5)));
    REQUIRE(c(1, 1, 0).imag() == Approx(T(1.0)));
    REQUIRE(c(1, 1, 1).real() == Approx(T(-3.5)));
    REQUIRE(c(1, 1, 1).imag() == Approx(T(1.0)));
}

// In place operations

TEMPLATE_TEST_CASE_2("fft_2d_c/3", "[fast][fft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    a.fft2_inplace();

    REQUIRE(a(0, 0).real() == Approx(Z(10.0)));
    REQUIRE(a(0, 0).imag() == Approx(Z(0.25)));
    REQUIRE(a(0, 1).real() == Approx(Z(9.0)));
    REQUIRE(a(0, 1).imag() == Approx(Z(5.75)));
    REQUIRE(a(1, 0).real() == Approx(Z(-10.1806)));
    REQUIRE(a(1, 0).imag() == Approx(Z(11.7673)));
    REQUIRE(a(1, 1).real() == Approx(Z(5.4127)));
    REQUIRE(a(1, 1).imag() == Approx(Z(-9.1692)));
    REQUIRE(a(2, 0).real() == Approx(Z(-2.8194)));
    REQUIRE(a(2, 0).imag() == Approx(Z(-9.0173)));
    REQUIRE(a(2, 1).real() == Approx(Z(-5.4127)));
    REQUIRE(a(2, 1).imag() == Approx(Z(6.4192)));
}

TEMPLATE_TEST_CASE_2("fft_2d_many/2", "[fast][fft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 2, 2, 2> a({MZ(1.0, 1.0), MZ(2.0, 3.0), MZ(-1.0, 0.0), MZ(-2.0, 0.0),
                                                  MZ(-1.0, 1.0), MZ(2.0, 1.0), MZ(-0.5, 0.0), MZ(-1.0, 1.0)});

    a.fft2_many_inplace();

    REQUIRE(a(0, 0, 0).real() == Approx(Z(0.0)));
    REQUIRE(a(0, 0, 0).imag() == Approx(Z(4.0)));
    REQUIRE(a(0, 0, 1).real() == Approx(Z(0.0)));
    REQUIRE(a(0, 0, 1).imag() == Approx(Z(-2.0)));
    REQUIRE(a(0, 1, 0).real() == Approx(Z(6.0)));
    REQUIRE(a(0, 1, 0).imag() == Approx(Z(4.0)));
    REQUIRE(a(0, 1, 1).real() == Approx(Z(-2.0)));
    REQUIRE(a(0, 1, 1).imag() == Approx(Z(-2.0)));

    REQUIRE(a(1, 0, 0).real() == Approx(Z(-0.5)));
    REQUIRE(a(1, 0, 0).imag() == Approx(Z(3.0)));
    REQUIRE(a(1, 0, 1).real() == Approx(Z(-2.5)));
    REQUIRE(a(1, 0, 1).imag() == Approx(Z(-1.0)));
    REQUIRE(a(1, 1, 0).real() == Approx(Z(2.5)));
    REQUIRE(a(1, 1, 0).imag() == Approx(Z(1.0)));
    REQUIRE(a(1, 1, 1).real() == Approx(Z(-3.5)));
    REQUIRE(a(1, 1, 1).imag() == Approx(Z(1.0)));
}
