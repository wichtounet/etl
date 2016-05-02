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

TEMPLATE_TEST_CASE_2("fft_1d_many/5", "[fast][fft]", Z, float, double) {
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1033> a;
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1033> c_1;
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1033> c_2;

    c_1 = etl::fft_1d_many(a);

    for(std::size_t i = 0; i < 18; ++i){
        c_2(i) = etl::fft_1d(a(i));
    }

    for(std::size_t i = 0; i < a.size(); ++i){
        REQUIRE(c_1[i] == c_2[i]);
    }
}

TEMPLATE_TEST_CASE_2("fft_1d_many/6", "[fast][fft]", Z, float, double) {
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1045> a;
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1045> c_1;
    etl::fast_dyn_matrix<std::complex<Z>, 18, 1045> c_2;

    c_1 = etl::fft_1d_many(a);

    for(std::size_t i = 0; i < 18; ++i){
        c_2(i) = etl::fft_1d(a(i));
    }

    for(std::size_t i = 0; i < a.size(); ++i){
        REQUIRE(c_1[i] == c_2[i]);
    }
}
