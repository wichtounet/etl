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

//fft_1d (real)

FFT1_TEST_CASE("fft_1d_r/0", "[fast][fft]") {
    etl::fast_matrix<T, 8> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<std::complex<T>, 8> c;

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(4.0)));
    REQUIRE(c(0).imag() == Approx(T(0.0)));
    REQUIRE(c(1).real() == Approx(T(1.0)));
    REQUIRE(c(1).imag() == Approx(T(-2.41421)));
    REQUIRE(c(2).real() == Approx(T(0.0)));
    REQUIRE(c(2).imag() == Approx(T(0.0)));
    REQUIRE(c(3).real() == Approx(T(1.0)));
    REQUIRE(c(3).imag() == Approx(T(-0.41421)));
    REQUIRE(c(4).real() == Approx(T(0.0)));
    REQUIRE(c(4).imag() == Approx(T(0.0)));
    REQUIRE(c(5).real() == Approx(T(1.0)));
    REQUIRE(c(5).imag() == Approx(T(0.41421)));
    REQUIRE(c(6).real() == Approx(T(0.0)));
    REQUIRE(c(6).imag() == Approx(T(0.0)));
    REQUIRE(c(7).real() == Approx(T(1.0)));
    REQUIRE(c(7).imag() == Approx(T(2.41421)));
}

FFT1_TEST_CASE("fft_1d_r/1", "[fast][fft]") {
    etl::fast_matrix<T, 5> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::fast_matrix<std::complex<T>, 5> c;

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(15.0)));
    REQUIRE(c(0).imag() == Approx(T(0.0)));
    REQUIRE(c(1).real() == Approx(T(-2.5)));
    REQUIRE(c(1).imag() == Approx(T(3.440955)));
    REQUIRE(c(2).real() == Approx(T(-2.5)));
    REQUIRE(c(2).imag() == Approx(T(0.8123)));
    REQUIRE(c(3).real() == Approx(T(-2.5)));
    REQUIRE(c(3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(4).real() == Approx(T(-2.5)));
    REQUIRE(c(4).imag() == Approx(T(-3.440955)));
}

FFT1_TEST_CASE("fft_1d_r/2", "[fast][fft]") {
    etl::fast_matrix<T, 6> a({0.5, 1.5, 3.5, -1.5, 3.9, -5.5});
    etl::fast_matrix<std::complex<T>, 6> c;

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(2.4)));
    REQUIRE(c(0).imag() == Approx(T(0.0)));
    REQUIRE(c(1).real() == Approx(T(-3.7)));
    REQUIRE(c(1).imag() == Approx(T(-5.71577)));
    REQUIRE(c(2).real() == Approx(T(-2.7)));
    REQUIRE(c(2).imag() == Approx(T(-6.40859)));
    REQUIRE(c(3).real() == Approx(T(13.4)));
    REQUIRE(c(3).imag() == Approx(T(0)));
    REQUIRE(c(4).real() == Approx(T(-2.7)));
    REQUIRE(c(4).imag() == Approx(T(6.40859)));
    REQUIRE(c(5).real() == Approx(T(-3.7)));
    REQUIRE(c(5).imag() == Approx(T(5.71577)));
}

FFT1_TEST_CASE("fft_1d_r/3", "[fast][fft]") {
    etl::fast_matrix<T, 12> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0});
    etl::fast_matrix<std::complex<T>, 12> c;

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(8.0)));
    REQUIRE(c(0).imag() == Approx(T(0.0)));
    REQUIRE(c(1).real() == Approx(T(3.23205)));
    REQUIRE(c(1).imag() == Approx(T(0.86603)));
    REQUIRE(c(2).real() == Approx(T(-1.5)));
    REQUIRE(c(2).imag() == Approx(T(-0.86603)));
    REQUIRE(c(3).real() == Approx(T(0.0)));
    REQUIRE(c(3).imag() == Approx(T(0.0)));
    REQUIRE(c(4).real() == Approx(T(0.5)));
    REQUIRE(c(4).imag() == Approx(T(0.86603)));
    REQUIRE(c(5).real() == Approx(T(-0.23205)));
    REQUIRE(c(5).imag() == Approx(T(-0.86603)));
    REQUIRE(c(6).real() == Approx(T(0.0)));
    REQUIRE(c(6).imag() == Approx(T(0.0)));

    REQUIRE(c(7).real() == Approx(T(-0.23205)));
    REQUIRE(c(7).imag() == Approx(T(0.86603)));
    REQUIRE(c(8).real() == Approx(T(0.5)));
    REQUIRE(c(8).imag() == Approx(T(-0.86603)));
    REQUIRE(c(9).real() == Approx(T(0.0)));
    REQUIRE(c(9).imag() == Approx(T(0.0)));

    REQUIRE(c(10).real() == Approx(T(-1.5)));
    REQUIRE(c(10).imag() == Approx(T(0.86603)));
    REQUIRE(c(11).real() == Approx(T(3.23205)));
    REQUIRE(c(11).imag() == Approx(T(-.86603)));
}

//fft_1d (complex)

FFT1_TEST_CASE("fft_1d_c/1", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 5> a;
    etl::fast_matrix<std::complex<T>, 5> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(0.5)));
    REQUIRE(c(0).imag() == Approx(T(5.5)));
    REQUIRE(c(1).real() == Approx(T(5.626178)));
    REQUIRE(c(1).imag() == Approx(T(0.376206)));
    REQUIRE(c(2).real() == Approx(T(-1.067916)));
    REQUIRE(c(2).imag() == Approx(T(-2.571198)));
    REQUIRE(c(3).real() == Approx(T(-2.831271)));
    REQUIRE(c(3).imag() == Approx(T(-2.709955)));
    REQUIRE(c(4).real() == Approx(T(2.773009)));
    REQUIRE(c(4).imag() == Approx(T(4.404947)));
}

FFT1_TEST_CASE("fft_1d_c/2", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 6> a;
    etl::fast_matrix<std::complex<T>, 6> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);
    a[5] = std::complex<T>(0.5, 0.5);

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(1.0)));
    REQUIRE(c(0).imag() == Approx(T(6.0)));
    REQUIRE(c(1).real() == Approx(T(5.366025)));
    REQUIRE(c(1).imag() == Approx(T(2.0)));
    REQUIRE(c(2).real() == Approx(T(1.464102)));
    REQUIRE(c(2).imag() == Approx(T(-4.098076)));
    REQUIRE(c(3).real() == Approx(T(0)));
    REQUIRE(c(3).imag() == Approx(T(-1.0)));
    REQUIRE(c(4).real() == Approx(T(-5.464102)));
    REQUIRE(c(4).imag() == Approx(T(1.098076)));
    REQUIRE(c(5).real() == Approx(T(3.633975)));
    REQUIRE(c(5).imag() == Approx(T(2.0)));
}

FFT1_TEST_CASE("fft_1d_c/4", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 5> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);

    Impl::apply(a, a);

    REQUIRE(a(0).real() == Approx(T(0.5)));
    REQUIRE(a(0).imag() == Approx(T(5.5)));
    REQUIRE(a(1).real() == Approx(T(5.626178)));
    REQUIRE(a(1).imag() == Approx(T(0.376206)));
    REQUIRE(a(2).real() == Approx(T(-1.067916)));
    REQUIRE(a(2).imag() == Approx(T(-2.571198)));
    REQUIRE(a(3).real() == Approx(T(-2.831271)));
    REQUIRE(a(3).imag() == Approx(T(-2.709955)));
    REQUIRE(a(4).real() == Approx(T(2.773009)));
    REQUIRE(a(4).imag() == Approx(T(4.404947)));
}

FFT1_TEST_CASE("fft_1d/5", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2> a;
    etl::fast_matrix<std::complex<T>, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == std::complex<T>(3.0, 4.0));
    REQUIRE(c(1) == std::complex<T>(-1.0, -2.0));
}

FFT1_TEST_CASE("fft_1d/6", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 3> a;
    etl::fast_matrix<std::complex<T>, 3> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(6.0, 1.0));
    REQUIRE(c(1) == ComplexApprox<T>(3.69615, 1.86603));
    REQUIRE(c(2) == ComplexApprox<T>(-6.69615, 0.133975));
}

FFT1_TEST_CASE("fft_1d/7", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 4> a;
    etl::fast_matrix<std::complex<T>, 4> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(2.0, -1.0);
    a[3] = std::complex<T>(4.0, 3.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(9.0, 6.0));
    REQUIRE(c(1) == ComplexApprox<T>(-1.0, 4.0));
    REQUIRE(c(2) == ComplexApprox<T>(-3.0, -6.0));
    REQUIRE(c(3) == ComplexApprox<T>(-1.0, 0.0));
}

FFT1_TEST_CASE("fft_1d/8", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 8> a;
    etl::fast_matrix<std::complex<T>, 8> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(2.0, -1.0);
    a[3] = std::complex<T>(4.0, 3.0);
    a[4] = std::complex<T>(1.0, 1.0);
    a[5] = std::complex<T>(2.0, 3.0);
    a[6] = std::complex<T>(2.0, -1.0);
    a[7] = std::complex<T>(4.0, 3.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(18.0, 12.0));
    REQUIRE(c(1) == ComplexApprox<T>(0.0, 0.0));
    REQUIRE(c(2) == ComplexApprox<T>(-2.0, 8.0));
    REQUIRE(c(3) == ComplexApprox<T>(0.0, 0.0));
    REQUIRE(c(4) == ComplexApprox<T>(-6.0, -12.0));
    REQUIRE(c(5) == ComplexApprox<T>(0.0, 0.0));
    REQUIRE(c(6) == ComplexApprox<T>(-2.0, 0.0));
    REQUIRE(c(7) == ComplexApprox<T>(0.0, 0.0));
}

FFT1_TEST_CASE("fft_1d/9", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 6> a;
    etl::fast_matrix<std::complex<T>, 6> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(2.0, -1.0);
    a[3] = std::complex<T>(4.0, 3.0);
    a[4] = std::complex<T>(1.0, 1.0);
    a[5] = std::complex<T>(2.0, 3.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(12.0, 10.0));
    REQUIRE(c(1) == ComplexApprox<T>(-4.23205, 0.133975));
    REQUIRE(c(2) == ComplexApprox<T>(3.232051, 1.866025));
    REQUIRE(c(3) == ComplexApprox<T>(-4.0, -8.0));
    REQUIRE(c(4) == ComplexApprox<T>(-0.232051, 0.133975));
    REQUIRE(c(5) == ComplexApprox<T>(-0.767949, 1.866025));
}

FFT1_TEST_CASE("fft_1d/10", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 11> a;
    etl::fast_matrix<std::complex<T>, 11> c;

    a[0]  = std::complex<T>(1.0, 1.0);
    a[1]  = std::complex<T>(2.0, 3.0);
    a[2]  = std::complex<T>(2.0, -1.0);
    a[3]  = std::complex<T>(4.0, 3.0);
    a[4]  = std::complex<T>(1.0, 1.0);
    a[5]  = std::complex<T>(2.0, 3.0);
    a[6]  = std::complex<T>(1.0, 1.0);
    a[7]  = std::complex<T>(2.0, 3.0);
    a[8]  = std::complex<T>(2.0, -1.0);
    a[9]  = std::complex<T>(4.0, 3.0);
    a[10] = std::complex<T>(1.0, 1.0);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(22.0, 17.0));
    REQUIRE(c(1) == ComplexApprox<T>(0.773306, -1.773203));
    REQUIRE(c(2) == ComplexApprox<T>(-6.775364, 2.944859));
    REQUIRE(c(3) == ComplexApprox<T>(-2.233971, +0.139025));
    REQUIRE(c(4) == ComplexApprox<T>(6.847435, -5.023187));
    REQUIRE(c(5) == ComplexApprox<T>(9.607112, -6.146752));
    REQUIRE(c(6) == ComplexApprox<T>(-9.488755, 3.401181));
    REQUIRE(c(7) == ComplexApprox<T>(-3.653803, 0.227432));
    REQUIRE(c(8) == ComplexApprox<T>(-2.030497, 0.037287));
    REQUIRE(c(9) == ComplexApprox<T>(-3.910758, 1.512556));
    REQUIRE(c(10) == ComplexApprox<T>(-0.134705, -1.319198));
}

FFT1_TEST_CASE("fft_1d/11", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 7> a;
    etl::fast_matrix<std::complex<T>, 7> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);
    a[5] = std::complex<T>(0.5, 0.5);
    a[6] = std::complex<T>(1.5, 1.5);

    Impl::apply(a, c);

    REQUIRE(c(0) == ComplexApprox<T>(2.5, 7.5));
    REQUIRE(c(1) == ComplexApprox<T>(4.679386, 4.499176));
    REQUIRE(c(2) == ComplexApprox<T>(2.588507, -2.609462));
    REQUIRE(c(3) == ComplexApprox<T>(-2.552005, -2.028766));
    REQUIRE(c(4) == ComplexApprox<T>(-1.710704, -4.124027));
    REQUIRE(c(5) == ComplexApprox<T>(-3.115654, 3.576274));
    REQUIRE(c(6) == ComplexApprox<T>(4.61047, 0.186805));
}

//ifft_1d (complex)

IFFT1_TEST_CASE("ifft_1d_c/0", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 4> a;
    etl::fast_matrix<std::complex<T>, 4> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 2.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(1.25)));
    REQUIRE(c(0).imag() == Approx(T(0.0)));
    REQUIRE(c(1).real() == Approx(T(-0.75)));
    REQUIRE(c(1).imag() == Approx(T(0.0)));
    REQUIRE(c(2).real() == Approx(T(-1.25)));
    REQUIRE(c(2).imag() == Approx(T(0.5)));
    REQUIRE(c(3).real() == Approx(T(1.75)));
    REQUIRE(c(3).imag() == Approx(T(0.5)));
}

IFFT1_TEST_CASE("ifft_1d_c/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 5> a;
    etl::fast_matrix<std::complex<T>, 5> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(0.1)));
    REQUIRE(c(0).imag() == Approx(T(1.1)));
    REQUIRE(c(1).real() == Approx(T(0.554602)));
    REQUIRE(c(1).imag() == Approx(T(0.880989)));
    REQUIRE(c(2).real() == Approx(T(-0.566254)));
    REQUIRE(c(2).imag() == Approx(T(-0.541991)));
    REQUIRE(c(3).real() == Approx(T(-0.213583)));
    REQUIRE(c(3).imag() == Approx(T(-0.51424)));
    REQUIRE(c(4).real() == Approx(T(1.125236)));
    REQUIRE(c(4).imag() == Approx(T(0.075241)));
}

IFFT1_TEST_CASE("ifft_1d_c/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 6> a;
    etl::fast_matrix<std::complex<T>, 6> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);
    a[5] = std::complex<T>(0.5, 0.5);

    Impl::apply(a, c);

    REQUIRE(c(0).real() == Approx(T(0.166666)));
    REQUIRE(c(0).imag() == Approx(T(1.0)));
    REQUIRE(c(1).real() == Approx(T(0.605662)));
    REQUIRE(c(1).imag() == Approx(T(0.333333)));
    REQUIRE(c(2).real() == Approx(T(-0.910684)));
    REQUIRE(c(2).imag() == Approx(T(0.183013)));
    REQUIRE(c(3).real() == Approx(T(0)));
    REQUIRE(c(3).imag() == Approx(T(-0.166667)));
    REQUIRE(c(4).real() == Approx(T(0.244017)));
    REQUIRE(c(4).imag() == Approx(T(-0.683013)));
    REQUIRE(c(5).real() == Approx(T(0.894338)));
    REQUIRE(c(5).imag() == Approx(T(0.333333)));
}

IFFT1_TEST_CASE("ifft_1d_c/4", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 4> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 2.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, a);

    REQUIRE(a(0).real() == Approx(T(1.25)));
    REQUIRE(a(0).imag() == Approx(T(0.0)));
    REQUIRE(a(1).real() == Approx(T(-0.75)));
    REQUIRE(a(1).imag() == Approx(T(0.0)));
    REQUIRE(a(2).real() == Approx(T(-1.25)));
    REQUIRE(a(2).imag() == Approx(T(0.5)));
    REQUIRE(a(3).real() == Approx(T(1.75)));
    REQUIRE(a(3).imag() == Approx(T(0.5)));
}

//ifft_1d (real)

IFFT1_REAL_TEST_CASE("ifft_1d_real/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 5> a;
    etl::fast_matrix<T, 5> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);

    Impl::apply(a, c);

    REQUIRE(c(0) == Approx(T(0.1)));
    REQUIRE(c(1) == Approx(T(0.554602)));
    REQUIRE(c(2) == Approx(T(-0.566254)));
    REQUIRE(c(3) == Approx(T(-0.213583)));
    REQUIRE(c(4) == Approx(T(1.125236)));
}

IFFT1_REAL_TEST_CASE("ifft_1d_real/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 6> a;
    etl::fast_matrix<T, 6> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);
    a[5] = std::complex<T>(0.5, 0.5);

    Impl::apply(a, c);

    REQUIRE(c(0) == Approx(T(0.166666)));
    REQUIRE(c(1) == Approx(T(0.605662)));
    REQUIRE(c(2) == Approx(T(-0.910684)));
    REQUIRE(c(3) == Approx(T(0)));
    REQUIRE(c(4) == Approx(T(0.244017)));
    REQUIRE(c(5) == Approx(T(0.894338)));
}

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

//ifft_2d (complex)

IFFT2_TEST_CASE("ifft_2d_c/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 3, 2> a;
    etl::fast_matrix<std::complex<T>, 3, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(1.66667)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(c(0, 1).real() == Approx(T(1.5)));
    REQUIRE(c(0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(c(1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(c(1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(c(1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(c(1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(c(2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(c(2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(c(2, 1).real() == Approx(T(0.9021)));
    REQUIRE(c(2, 1).imag() == Approx(T(-1.5282)));
}

IFFT2_TEST_CASE("ifft_2d_c/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 2> a;
    etl::fast_matrix<std::complex<T>, 2, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(0.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(1.0)));
    REQUIRE(c(0, 1).real() == Approx(T(0.0)));
    REQUIRE(c(0, 1).imag() == Approx(T(-0.5)));
    REQUIRE(c(1, 0).real() == Approx(T(1.5)));
    REQUIRE(c(1, 0).imag() == Approx(T(1.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-0.5)));
    REQUIRE(c(1, 1).imag() == Approx(T(-0.5)));
}

IFFT2_TEST_CASE("ifft_2d_c/4", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 3, 2> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, a);

    REQUIRE(a(0, 0).real() == Approx(T(1.66667)));
    REQUIRE(a(0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(a(0, 1).real() == Approx(T(1.5)));
    REQUIRE(a(0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(a(1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(a(1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(a(1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(a(1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(a(2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(a(2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(a(2, 1).real() == Approx(T(0.9021)));
    REQUIRE(a(2, 1).imag() == Approx(T(-1.5282)));
}

//ifft_2d (real)

IFFT2_REAL_TEST_CASE("ifft_2d_c_real/1", "[fast][ifft]"){
    etl::fast_matrix<std::complex<T>, 3, 2> a;
    etl::fast_matrix<T, 3, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, c);

    REQUIRE(c(0, 0) == Approx(T(1.66667)));
    REQUIRE(c(0, 1) == Approx(T(1.5)));
    REQUIRE(c(1, 0) == Approx(T(-0.4699)));
    REQUIRE(c(1, 1) == Approx(T(-0.9021)));
    REQUIRE(c(2, 0) == Approx(T(-1.6968)));
    REQUIRE(c(2, 1) == Approx(T(0.9021)));
}

IFFT2_REAL_TEST_CASE("ifft_2d_real_c/2", "[fast][ifft]"){
    etl::fast_matrix<std::complex<T>, 2, 2> a;
    etl::fast_matrix<T, 2, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);

    Impl::apply(a, c);

    REQUIRE(c(0, 0) == Approx(T(0.0)));
    REQUIRE(c(0, 1) == Approx(T(0.0)));
    REQUIRE(c(1, 0) == Approx(T(1.5)));
    REQUIRE(c(1, 1) == Approx(T(-0.5)));
}

/* fft_many tests */

FFT1_MANY_TEST_CASE("fft_1d_many/1", "[fast][fft]") {
    etl::fast_matrix<T, 2, 5> a({1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(15.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(0, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(0, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(0, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 4).imag() == Approx(T(-3.440955)));

    REQUIRE(c(1, 0).real() == Approx(T(15.0)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(1, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(1, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(1, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 4).imag() == Approx(T(-3.440955)));
}

FFT1_MANY_TEST_CASE("fft_1d_many/2", "[fast][fft]") {
    etl::fast_matrix<T, 2, 5> a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(15.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(0, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(0, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(0, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 4).imag() == Approx(T(-3.440955)));

    REQUIRE(c(1, 0).real() == Approx(T(40.0)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(1, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(1, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(1, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 4).imag() == Approx(T(-3.440955)));
}

FFT1_MANY_TEST_CASE("fft_1d_many/3", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2, 5> a({MC(1.0, 0.0), MC(2.0, 0.0), MC(3.0, 0.0), MC(4.0, 0.0), MC(5.0, 0.0), MC(6.0, 0.0), MC(7.0, 0.0), MC(8.0, 0.0), MC(9.0, 0.0), MC(10.0, 0.0)});
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(15.0)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(0, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(0, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(0, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(0, 4).imag() == Approx(T(-3.440955)));

    REQUIRE(c(1, 0).real() == Approx(T(40.0)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 1).imag() == Approx(T(3.440955)));
    REQUIRE(c(1, 2).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.8123)));
    REQUIRE(c(1, 3).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 3).imag() == Approx(T(-0.8123)));
    REQUIRE(c(1, 4).real() == Approx(T(-2.5)));
    REQUIRE(c(1, 4).imag() == Approx(T(-3.440955)));
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

TEMPLATE_TEST_CASE_2("fft_1d_c/3", "[fast][fft]", Z, float, double) {
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

TEMPLATE_TEST_CASE_2("ifft_1d_c/3", "[fast][ifft]", Z, float, double) {
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

TEMPLATE_TEST_CASE_2("ifft_2d_c/3", "[fast][ifft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 3, 2> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(-2.0, 0.0);
    a[2] = std::complex<Z>(3.5, 1.5);
    a[3] = std::complex<Z>(-4.0, -4.0);
    a[4] = std::complex<Z>(5.0, 0.5);
    a[5] = std::complex<Z>(6.5, 1.25);

    a.ifft2_inplace();

    REQUIRE(a(0, 0).real() == Approx(Z(1.66667)));
    REQUIRE(a(0, 0).imag() == Approx(Z(0.04167)));
    REQUIRE(a(0, 1).real() == Approx(Z(1.5)));
    REQUIRE(a(0, 1).imag() == Approx(Z(0.95833)));
    REQUIRE(a(1, 0).real() == Approx(Z(-0.4699)));
    REQUIRE(a(1, 0).imag() == Approx(Z(-1.5029)));
    REQUIRE(a(1, 1).real() == Approx(Z(-0.9021)));
    REQUIRE(a(1, 1).imag() == Approx(Z(1.06987)));
    REQUIRE(a(2, 0).real() == Approx(Z(-1.6968)));
    REQUIRE(a(2, 0).imag() == Approx(Z(1.9612)));
    REQUIRE(a(2, 1).real() == Approx(Z(0.9021)));
    REQUIRE(a(2, 1).imag() == Approx(Z(-1.5282)));
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

TEMPLATE_TEST_CASE_2("fft_1d_many/4", "[fast][fft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 2, 5> a({MZ(1.0, 0.0), MZ(2.0, 0.0), MZ(3.0, 0.0), MZ(4.0, 0.0), MZ(5.0, 0.0), MZ(6.0, 0.0), MZ(7.0, 0.0), MZ(8.0, 0.0), MZ(9.0, 0.0), MZ(10.0, 0.0)});

    a.fft_many_inplace();

    REQUIRE(a(0, 0).real() == Approx(Z(15.0)));
    REQUIRE(a(0, 0).imag() == Approx(Z(0.0)));
    REQUIRE(a(0, 1).real() == Approx(Z(-2.5)));
    REQUIRE(a(0, 1).imag() == Approx(Z(3.440955)));
    REQUIRE(a(0, 2).real() == Approx(Z(-2.5)));
    REQUIRE(a(0, 2).imag() == Approx(Z(0.8123)));
    REQUIRE(a(0, 3).real() == Approx(Z(-2.5)));
    REQUIRE(a(0, 3).imag() == Approx(Z(-0.8123)));
    REQUIRE(a(0, 4).real() == Approx(Z(-2.5)));
    REQUIRE(a(0, 4).imag() == Approx(Z(-3.440955)));

    REQUIRE(a(1, 0).real() == Approx(Z(40.0)));
    REQUIRE(a(1, 0).imag() == Approx(Z(0.0)));
    REQUIRE(a(1, 1).real() == Approx(Z(-2.5)));
    REQUIRE(a(1, 1).imag() == Approx(Z(3.440955)));
    REQUIRE(a(1, 2).real() == Approx(Z(-2.5)));
    REQUIRE(a(1, 2).imag() == Approx(Z(0.8123)));
    REQUIRE(a(1, 3).real() == Approx(Z(-2.5)));
    REQUIRE(a(1, 3).imag() == Approx(Z(-0.8123)));
    REQUIRE(a(1, 4).real() == Approx(Z(-2.5)));
    REQUIRE(a(1, 4).imag() == Approx(Z(-3.440955)));
}

//ifft many

IFFT1_MANY_TEST_CASE("ifft_1d_many/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 4> a;
    etl::fast_matrix<std::complex<T>, 2, 4> c;

    a(0, 0) = std::complex<T>(1.0, 1.0);
    a(0, 1) = std::complex<T>(2.0, 2.0);
    a(0, 2) = std::complex<T>(-1.0, 0.0);
    a(0, 3) = std::complex<T>(3.0, -3.0);

    a(1, 0) = std::complex<T>(1.0, 1.0);
    a(1, 1) = std::complex<T>(2.0, 2.0);
    a(1, 2) = std::complex<T>(-1.0, 0.0);
    a(1, 3) = std::complex<T>(3.0, -3.0);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(1.25)));
    REQUIRE(c(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 1).real() == Approx(T(-0.75)));
    REQUIRE(c(0, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(0, 2).real() == Approx(T(-1.25)));
    REQUIRE(c(0, 2).imag() == Approx(T(0.5)));
    REQUIRE(c(0, 3).real() == Approx(T(1.75)));
    REQUIRE(c(0, 3).imag() == Approx(T(0.5)));

    REQUIRE(c(1, 0).real() == Approx(T(1.25)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(-0.75)));
    REQUIRE(c(1, 1).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 2).real() == Approx(T(-1.25)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.5)));
    REQUIRE(c(1, 3).real() == Approx(T(1.75)));
    REQUIRE(c(1, 3).imag() == Approx(T(0.5)));
}

IFFT1_MANY_TEST_CASE("ifft_1d_many/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 4> a;
    etl::fast_matrix<std::complex<T>, 2, 4> c;

    a(0, 0) = std::complex<T>(2.0, 1.0);
    a(0, 1) = std::complex<T>(2.0, 3.0);
    a(0, 2) = std::complex<T>(0.0, 4.0);
    a(0, 3) = std::complex<T>(1.0, -2.0);

    a(1, 0) = std::complex<T>(2.0, 0.0);
    a(1, 1) = std::complex<T>(3.0, 1.0);
    a(1, 2) = std::complex<T>(-2.0, 0.0);
    a(1, 3) = std::complex<T>(1.5, -1.0);

    Impl::apply(a, c);

    REQUIRE(c(0, 0).real() == Approx(T(1.25)));
    REQUIRE(c(0, 0).imag() == Approx(T(1.5)));
    REQUIRE(c(0, 1).real() == Approx(T(-0.75)));
    REQUIRE(c(0, 1).imag() == Approx(T(-0.5)));
    REQUIRE(c(0, 2).real() == Approx(T(-0.25)));
    REQUIRE(c(0, 2).imag() == Approx(T(1.0)));
    REQUIRE(c(0, 3).real() == Approx(T(1.75)));
    REQUIRE(c(0, 3).imag() == Approx(T(-1.0)));

    REQUIRE(c(1, 0).real() == Approx(T(1.125)));
    REQUIRE(c(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 1).real() == Approx(T(0.5)));
    REQUIRE(c(1, 1).imag() == Approx(T(0.375)));
    REQUIRE(c(1, 2).real() == Approx(T(-1.125)));
    REQUIRE(c(1, 2).imag() == Approx(T(0.0)));
    REQUIRE(c(1, 3).real() == Approx(T(1.5)));
    REQUIRE(c(1, 3).imag() == Approx(T(-0.375)));
}

IFFT2_MANY_TEST_CASE("ifft_2d_many/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 3, 2> a;
    etl::fast_matrix<std::complex<T>, 2, 3, 2> c;

    a(0, 0, 0) = std::complex<T>(1.0, 1.0);
    a(0, 0, 1) = std::complex<T>(-2.0, 0.0);
    a(0, 1, 0) = std::complex<T>(3.5, 1.5);
    a(0, 1, 1) = std::complex<T>(-4.0, -4.0);
    a(0, 2, 0) = std::complex<T>(5.0, 0.5);
    a(0, 2, 1) = std::complex<T>(6.5, 1.25);

    a(1, 0, 0) = std::complex<T>(1.0, 1.0);
    a(1, 0, 1) = std::complex<T>(-2.0, 0.0);
    a(1, 1, 0) = std::complex<T>(3.5, 1.5);
    a(1, 1, 1) = std::complex<T>(-4.0, -4.0);
    a(1, 2, 0) = std::complex<T>(5.0, 0.5);
    a(1, 2, 1) = std::complex<T>(6.5, 1.25);

    Impl::apply(a, c);

    REQUIRE(c(0, 0, 0).real() == Approx(T(1.66667)));
    REQUIRE(c(0, 0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(c(0, 0, 1).real() == Approx(T(1.5)));
    REQUIRE(c(0, 0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(c(0, 1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(c(0, 1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(c(0, 1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(c(0, 1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(c(0, 2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(c(0, 2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(c(0, 2, 1).real() == Approx(T(0.9021)));
    REQUIRE(c(0, 2, 1).imag() == Approx(T(-1.5282)));

    REQUIRE(c(1, 0, 0).real() == Approx(T(1.66667)));
    REQUIRE(c(1, 0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(c(1, 0, 1).real() == Approx(T(1.5)));
    REQUIRE(c(1, 0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(c(1, 1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(c(1, 1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(c(1, 1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(c(1, 1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(c(1, 2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(c(1, 2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(c(1, 2, 1).real() == Approx(T(0.9021)));
    REQUIRE(c(1, 2, 1).imag() == Approx(T(-1.5282)));
}

IFFT2_MANY_TEST_CASE("ifft_2d_many/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 3, 2> a;
    etl::fast_matrix<std::complex<T>, 2, 3, 2> c;

    a(0, 0, 0) = std::complex<T>(2.0, 1.0);
    a(0, 0, 1) = std::complex<T>(-2.0, 1.0);
    a(0, 1, 0) = std::complex<T>(0.5, 1.5);
    a(0, 1, 1) = std::complex<T>(-4.0, -1.0);
    a(0, 2, 0) = std::complex<T>(5.5, 1.5);
    a(0, 2, 1) = std::complex<T>(2.5, 0.25);

    a(1, 0, 0) = std::complex<T>(1.0, -1.0);
    a(1, 0, 1) = std::complex<T>(2.0, 1.0);
    a(1, 1, 0) = std::complex<T>(-3.5, -1.5);
    a(1, 1, 1) = std::complex<T>(4.0, 4.0);
    a(1, 2, 0) = std::complex<T>(3.0, -1.5);
    a(1, 2, 1) = std::complex<T>(5.5, 0.25);

    Impl::apply(a, c);

    REQUIRE(c(0, 0, 0).real() == Approx(T(0.75)));
    REQUIRE(c(0, 0, 0).imag() == Approx(T(0.70833)));
    REQUIRE(c(0, 0, 1).real() == Approx(T(1.91666)));
    REQUIRE(c(0, 0, 1).imag() == Approx(T(0.625)));
    REQUIRE(c(0, 1, 0).real() == Approx(T(-0.194578)));
    REQUIRE(c(0, 1, 0).imag() == Approx(T(-1.51404)));
    REQUIRE(c(0, 1, 1).real() == Approx(T(-0.138755)));
    REQUIRE(c(0, 1, 1).imag() == Approx(T(-0.09599)));
    REQUIRE(c(0, 2, 0).real() == Approx(T(-0.55542)));
    REQUIRE(c(0, 2, 0).imag() == Approx(T(1.80571)));
    REQUIRE(c(0, 2, 1).real() == Approx(T(0.22208)));
    REQUIRE(c(0, 2, 1).imag() == Approx(T(-0.5290)));

    REQUIRE(c(1, 0, 0).real() == Approx(T(2.0)));
    REQUIRE(c(1, 0, 0).imag() == Approx(T(0.20833)));
    REQUIRE(c(1, 0, 1).real() == Approx(T(-1.83333)));
    REQUIRE(c(1, 0, 1).imag() == Approx(T(-1.54167)));
    REQUIRE(c(1, 1, 0).real() == Approx(T(-0.79127)));
    REQUIRE(c(1, 1, 0).imag() == Approx(T(-1.25887)));
    REQUIRE(c(1, 1, 1).real() == Approx(T(1.20793)));
    REQUIRE(c(1, 1, 1).imag() == Approx(T(-0.45085)));
    REQUIRE(c(1, 2, 0).real() == Approx(T(0.29127)));
    REQUIRE(c(1, 2, 0).imag() == Approx(T(1.05053)));
    REQUIRE(c(1, 2, 1).real() == Approx(T(0.1254)));
    REQUIRE(c(1, 2, 1).imag() == Approx(T(0.99252)));
}

//ifft many inplace

TEMPLATE_TEST_CASE_2("ifft_1d_many_inplace/1", "[fast][ifft]", T, double, float) {
    etl::fast_matrix<std::complex<T>, 2, 4> a;

    a(0, 0) = std::complex<T>(1.0, 1.0);
    a(0, 1) = std::complex<T>(2.0, 2.0);
    a(0, 2) = std::complex<T>(-1.0, 0.0);
    a(0, 3) = std::complex<T>(3.0, -3.0);

    a(1, 0) = std::complex<T>(1.0, 1.0);
    a(1, 1) = std::complex<T>(2.0, 2.0);
    a(1, 2) = std::complex<T>(-1.0, 0.0);
    a(1, 3) = std::complex<T>(3.0, -3.0);

    a.ifft_many_inplace();

    REQUIRE(a(0, 0).real() == Approx(T(1.25)));
    REQUIRE(a(0, 0).imag() == Approx(T(0.0)));
    REQUIRE(a(0, 1).real() == Approx(T(-0.75)));
    REQUIRE(a(0, 1).imag() == Approx(T(0.0)));
    REQUIRE(a(0, 2).real() == Approx(T(-1.25)));
    REQUIRE(a(0, 2).imag() == Approx(T(0.5)));
    REQUIRE(a(0, 3).real() == Approx(T(1.75)));
    REQUIRE(a(0, 3).imag() == Approx(T(0.5)));

    REQUIRE(a(1, 0).real() == Approx(T(1.25)));
    REQUIRE(a(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(a(1, 1).real() == Approx(T(-0.75)));
    REQUIRE(a(1, 1).imag() == Approx(T(0.0)));
    REQUIRE(a(1, 2).real() == Approx(T(-1.25)));
    REQUIRE(a(1, 2).imag() == Approx(T(0.5)));
    REQUIRE(a(1, 3).real() == Approx(T(1.75)));
    REQUIRE(a(1, 3).imag() == Approx(T(0.5)));
}

TEMPLATE_TEST_CASE_2("ifft_1d_many_inplace/2", "[fast][ifft]", T, double, float) {
    etl::fast_matrix<std::complex<T>, 2, 4> a;

    a(0, 0) = std::complex<T>(2.0, 1.0);
    a(0, 1) = std::complex<T>(2.0, 3.0);
    a(0, 2) = std::complex<T>(0.0, 4.0);
    a(0, 3) = std::complex<T>(1.0, -2.0);

    a(1, 0) = std::complex<T>(2.0, 0.0);
    a(1, 1) = std::complex<T>(3.0, 1.0);
    a(1, 2) = std::complex<T>(-2.0, 0.0);
    a(1, 3) = std::complex<T>(1.5, -1.0);

    a.ifft_many_inplace();

    REQUIRE(a(0, 0).real() == Approx(T(1.25)));
    REQUIRE(a(0, 0).imag() == Approx(T(1.5)));
    REQUIRE(a(0, 1).real() == Approx(T(-0.75)));
    REQUIRE(a(0, 1).imag() == Approx(T(-0.5)));
    REQUIRE(a(0, 2).real() == Approx(T(-0.25)));
    REQUIRE(a(0, 2).imag() == Approx(T(1.0)));
    REQUIRE(a(0, 3).real() == Approx(T(1.75)));
    REQUIRE(a(0, 3).imag() == Approx(T(-1.0)));

    REQUIRE(a(1, 0).real() == Approx(T(1.125)));
    REQUIRE(a(1, 0).imag() == Approx(T(0.0)));
    REQUIRE(a(1, 1).real() == Approx(T(0.5)));
    REQUIRE(a(1, 1).imag() == Approx(T(0.375)));
    REQUIRE(a(1, 2).real() == Approx(T(-1.125)));
    REQUIRE(a(1, 2).imag() == Approx(T(0.0)));
    REQUIRE(a(1, 3).real() == Approx(T(1.5)));
    REQUIRE(a(1, 3).imag() == Approx(T(-0.375)));
}

TEMPLATE_TEST_CASE_2("ifft_2d_many_inplace/1", "[fast][ifft]", T, double, float) {
    etl::fast_matrix<std::complex<T>, 2, 3, 2> a;

    a(0, 0, 0) = std::complex<T>(1.0, 1.0);
    a(0, 0, 1) = std::complex<T>(-2.0, 0.0);
    a(0, 1, 0) = std::complex<T>(3.5, 1.5);
    a(0, 1, 1) = std::complex<T>(-4.0, -4.0);
    a(0, 2, 0) = std::complex<T>(5.0, 0.5);
    a(0, 2, 1) = std::complex<T>(6.5, 1.25);

    a(1, 0, 0) = std::complex<T>(1.0, 1.0);
    a(1, 0, 1) = std::complex<T>(-2.0, 0.0);
    a(1, 1, 0) = std::complex<T>(3.5, 1.5);
    a(1, 1, 1) = std::complex<T>(-4.0, -4.0);
    a(1, 2, 0) = std::complex<T>(5.0, 0.5);
    a(1, 2, 1) = std::complex<T>(6.5, 1.25);

    a.ifft2_many_inplace();

    REQUIRE(a(0, 0, 0).real() == Approx(T(1.66667)));
    REQUIRE(a(0, 0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(a(0, 0, 1).real() == Approx(T(1.5)));
    REQUIRE(a(0, 0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(a(0, 1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(a(0, 1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(a(0, 1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(a(0, 1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(a(0, 2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(a(0, 2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(a(0, 2, 1).real() == Approx(T(0.9021)));
    REQUIRE(a(0, 2, 1).imag() == Approx(T(-1.5282)));

    REQUIRE(a(1, 0, 0).real() == Approx(T(1.66667)));
    REQUIRE(a(1, 0, 0).imag() == Approx(T(0.04167)));
    REQUIRE(a(1, 0, 1).real() == Approx(T(1.5)));
    REQUIRE(a(1, 0, 1).imag() == Approx(T(0.95833)));
    REQUIRE(a(1, 1, 0).real() == Approx(T(-0.4699)));
    REQUIRE(a(1, 1, 0).imag() == Approx(T(-1.5029)));
    REQUIRE(a(1, 1, 1).real() == Approx(T(-0.9021)));
    REQUIRE(a(1, 1, 1).imag() == Approx(T(1.06987)));
    REQUIRE(a(1, 2, 0).real() == Approx(T(-1.6968)));
    REQUIRE(a(1, 2, 0).imag() == Approx(T(1.9612)));
    REQUIRE(a(1, 2, 1).real() == Approx(T(0.9021)));
    REQUIRE(a(1, 2, 1).imag() == Approx(T(-1.5282)));
}

TEMPLATE_TEST_CASE_2("ifft_2d_many_inplace/2", "[fast][ifft]", T, double, float) {
    etl::fast_matrix<std::complex<T>, 2, 3, 2> a;

    a(0, 0, 0) = std::complex<T>(2.0, 1.0);
    a(0, 0, 1) = std::complex<T>(-2.0, 1.0);
    a(0, 1, 0) = std::complex<T>(0.5, 1.5);
    a(0, 1, 1) = std::complex<T>(-4.0, -1.0);
    a(0, 2, 0) = std::complex<T>(5.5, 1.5);
    a(0, 2, 1) = std::complex<T>(2.5, 0.25);

    a(1, 0, 0) = std::complex<T>(1.0, -1.0);
    a(1, 0, 1) = std::complex<T>(2.0, 1.0);
    a(1, 1, 0) = std::complex<T>(-3.5, -1.5);
    a(1, 1, 1) = std::complex<T>(4.0, 4.0);
    a(1, 2, 0) = std::complex<T>(3.0, -1.5);
    a(1, 2, 1) = std::complex<T>(5.5, 0.25);

    a.ifft2_many_inplace();

    REQUIRE(a(0, 0, 0).real() == Approx(T(0.75)));
    REQUIRE(a(0, 0, 0).imag() == Approx(T(0.70833)));
    REQUIRE(a(0, 0, 1).real() == Approx(T(1.91666)));
    REQUIRE(a(0, 0, 1).imag() == Approx(T(0.625)));
    REQUIRE(a(0, 1, 0).real() == Approx(T(-0.194578)));
    REQUIRE(a(0, 1, 0).imag() == Approx(T(-1.51404)));
    REQUIRE(a(0, 1, 1).real() == Approx(T(-0.138755)));
    REQUIRE(a(0, 1, 1).imag() == Approx(T(-0.09599)));
    REQUIRE(a(0, 2, 0).real() == Approx(T(-0.55542)));
    REQUIRE(a(0, 2, 0).imag() == Approx(T(1.80571)));
    REQUIRE(a(0, 2, 1).real() == Approx(T(0.22208)));
    REQUIRE(a(0, 2, 1).imag() == Approx(T(-0.5290)));

    REQUIRE(a(1, 0, 0).real() == Approx(T(2.0)));
    REQUIRE(a(1, 0, 0).imag() == Approx(T(0.20833)));
    REQUIRE(a(1, 0, 1).real() == Approx(T(-1.83333)));
    REQUIRE(a(1, 0, 1).imag() == Approx(T(-1.54167)));
    REQUIRE(a(1, 1, 0).real() == Approx(T(-0.79127)));
    REQUIRE(a(1, 1, 0).imag() == Approx(T(-1.25887)));
    REQUIRE(a(1, 1, 1).real() == Approx(T(1.20793)));
    REQUIRE(a(1, 1, 1).imag() == Approx(T(-0.45085)));
    REQUIRE(a(1, 2, 0).real() == Approx(T(0.29127)));
    REQUIRE(a(1, 2, 0).imag() == Approx(T(1.05053)));
    REQUIRE(a(1, 2, 1).real() == Approx(T(0.1254)));
    REQUIRE(a(1, 2, 1).imag() == Approx(T(0.99252)));
}
