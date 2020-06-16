//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
    etl::fast_matrix<T, 8> a{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    etl::fast_matrix<std::complex<T>, 8> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(-2.41421));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-0.41421));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(5).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(5).imag(), T(0.41421));
    REQUIRE_EQUALS_APPROX(c(6).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(6).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(7).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(7).imag(), T(2.41421));
}

FFT1_TEST_CASE("fft_1d_r/1", "[fast][fft]") {
    etl::fast_matrix<T, 5> a{1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_matrix<std::complex<T>, 5> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), T(15.0));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(-3.440955));
}

FFT1_TEST_CASE("fft_1d_r/2", "[fast][fft]") {
    etl::fast_matrix<T, 6> a{0.5, 1.5, 3.5, -1.5, 3.9, -5.5};
    etl::fast_matrix<std::complex<T>, 6> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), T(2.4));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(-3.7));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(-5.71577));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-2.7));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(-6.40859));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(13.4));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(0));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(-2.7));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(6.40859));
    REQUIRE_EQUALS_APPROX(c(5).real(), T(-3.7));
    REQUIRE_EQUALS_APPROX(c(5).imag(), T(5.71577));
}

FFT1_TEST_CASE("fft_1d_r/3", "[fast][fft]") {
    etl::fast_matrix<T, 12> a{1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
    etl::fast_matrix<std::complex<T>, 12> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), T(8.0));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(3.23205));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(0.86603));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-1.5));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(-0.86603));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(0.86603));
    REQUIRE_EQUALS_APPROX(c(5).real(), T(-0.23205));
    REQUIRE_EQUALS_APPROX(c(5).imag(), T(-0.86603));
    REQUIRE_EQUALS_APPROX(c(6).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(6).imag(), T(0.0));

    REQUIRE_EQUALS_APPROX(c(7).real(), T(-0.23205));
    REQUIRE_EQUALS_APPROX(c(7).imag(), T(0.86603));
    REQUIRE_EQUALS_APPROX(c(8).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(8).imag(), T(-0.86603));
    REQUIRE_EQUALS_APPROX(c(9).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(9).imag(), T(0.0));

    REQUIRE_EQUALS_APPROX(c(10).real(), T(-1.5));
    REQUIRE_EQUALS_APPROX(c(10).imag(), T(0.86603));
    REQUIRE_EQUALS_APPROX(c(11).real(), T(3.23205));
    REQUIRE_EQUALS_APPROX(c(11).imag(), T(-.86603));
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

    REQUIRE_EQUALS_APPROX(c(0).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(5.5));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(5.626178));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(0.376206));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-1.067916));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(-2.571198));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(-2.831271));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-2.709955));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(2.773009));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(4.404947));
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

    REQUIRE_EQUALS_APPROX(c(0).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(6.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(5.366025));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(2.0));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(1.464102));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(-4.098076));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(0));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-1.0));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(-5.464102));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(1.098076));
    REQUIRE_EQUALS_APPROX(c(5).real(), T(3.633975));
    REQUIRE_EQUALS_APPROX(c(5).imag(), T(2.0));
}

FFT1_TEST_CASE("fft_1d_c/4", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 5> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);
    a[4] = std::complex<T>(0.5, 1.5);

    Impl::apply(a, a);

    REQUIRE_EQUALS_APPROX(a(0).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(0).imag(), T(5.5));
    REQUIRE_EQUALS_APPROX(a(1).real(), T(5.626178));
    REQUIRE_EQUALS_APPROX(a(1).imag(), T(0.376206));
    REQUIRE_EQUALS_APPROX(a(2).real(), T(-1.067916));
    REQUIRE_EQUALS_APPROX(a(2).imag(), T(-2.571198));
    REQUIRE_EQUALS_APPROX(a(3).real(), T(-2.831271));
    REQUIRE_EQUALS_APPROX(a(3).imag(), T(-2.709955));
    REQUIRE_EQUALS_APPROX(a(4).real(), T(2.773009));
    REQUIRE_EQUALS_APPROX(a(4).imag(), T(4.404947));
}

FFT1_TEST_CASE("fft_1d/5", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2> a;
    etl::fast_matrix<std::complex<T>, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS(c(0), std::complex<T>(3.0, 4.0));
    REQUIRE_EQUALS(c(1), std::complex<T>(-1.0, -2.0));
}

FFT1_TEST_CASE("fft_1d/6", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 3> a;
    etl::fast_matrix<std::complex<T>, 3> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(6.0, 1.0));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(3.69615, 1.86603));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(-6.69615, 0.133975));
}

FFT1_TEST_CASE("fft_1d/7", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 4> a;
    etl::fast_matrix<std::complex<T>, 4> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(2.0, -1.0);
    a[3] = std::complex<T>(4.0, 3.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(9.0, 6.0));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(-1.0, 4.0));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(-3.0, -6.0));
    REQUIRE_EQUALS(c(3), ComplexApprox<T>(-1.0, 0.0));
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

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(18.0, 12.0));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(0.0, 0.0));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(-2.0, 8.0));
    REQUIRE_EQUALS(c(3), ComplexApprox<T>(0.0, 0.0));
    REQUIRE_EQUALS(c(4), ComplexApprox<T>(-6.0, -12.0));
    REQUIRE_EQUALS(c(5), ComplexApprox<T>(0.0, 0.0));
    REQUIRE_EQUALS(c(6), ComplexApprox<T>(-2.0, 0.0));
    REQUIRE_EQUALS(c(7), ComplexApprox<T>(0.0, 0.0));
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

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(12.0, 10.0));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(-4.23205, 0.133975));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(3.232051, 1.866025));
    REQUIRE_EQUALS(c(3), ComplexApprox<T>(-4.0, -8.0));
    REQUIRE_EQUALS(c(4), ComplexApprox<T>(-0.232051, 0.133975));
    REQUIRE_EQUALS(c(5), ComplexApprox<T>(-0.767949, 1.866025));
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

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(22.0, 17.0));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(0.773306, -1.773203));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(-6.775364, 2.944859));
    REQUIRE_EQUALS(c(3), ComplexApprox<T>(-2.233971, +0.139025));
    REQUIRE_EQUALS(c(4), ComplexApprox<T>(6.847435, -5.023187));
    REQUIRE_EQUALS(c(5), ComplexApprox<T>(9.607112, -6.146752));
    REQUIRE_EQUALS(c(6), ComplexApprox<T>(-9.488755, 3.401181));
    REQUIRE_EQUALS(c(7), ComplexApprox<T>(-3.653803, 0.227432));
    REQUIRE_EQUALS(c(8), ComplexApprox<T>(-2.030497, 0.037287));
    REQUIRE_EQUALS(c(9), ComplexApprox<T>(-3.910758, 1.512556));
    REQUIRE_EQUALS(c(10), ComplexApprox<T>(-0.134705, -1.319198));
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

    REQUIRE_EQUALS(c(0), ComplexApprox<T>(2.5, 7.5));
    REQUIRE_EQUALS(c(1), ComplexApprox<T>(4.679386, 4.499176));
    REQUIRE_EQUALS(c(2), ComplexApprox<T>(2.588507, -2.609462));
    REQUIRE_EQUALS(c(3), ComplexApprox<T>(-2.552005, -2.028766));
    REQUIRE_EQUALS(c(4), ComplexApprox<T>(-1.710704, -4.124027));
    REQUIRE_EQUALS(c(5), ComplexApprox<T>(-3.115654, 3.576274));
    REQUIRE_EQUALS(c(6), ComplexApprox<T>(4.61047, 0.186805));
}

/* fft_many tests */

FFT1_MANY_TEST_CASE("fft_1d_many/1", "[fast][fft]") {
    etl::fast_matrix<T, 2, 5> a{1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(15.0));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(0, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 4).imag(), T(-3.440955));

    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(15.0));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(1, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 4).imag(), T(-3.440955));
}

FFT1_MANY_TEST_CASE("fft_1d_many/2", "[fast][fft]") {
    etl::fast_matrix<T, 2, 5> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(15.0));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(0, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 4).imag(), T(-3.440955));

    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(40.0));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(1, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 4).imag(), T(-3.440955));
}

FFT1_MANY_TEST_CASE("fft_1d_many/3", "[fast][fft]") {
    etl::fast_matrix<std::complex<T>, 2, 5> a{MC(1.0, 0.0), MC(2.0, 0.0), MC(3.0, 0.0), MC(4.0, 0.0), MC(5.0, 0.0), MC(6.0, 0.0), MC(7.0, 0.0), MC(8.0, 0.0), MC(9.0, 0.0), MC(10.0, 0.0)};
    etl::fast_matrix<std::complex<T>, 2, 5> c;

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(15.0));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(0, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(0, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(0, 4).imag(), T(-3.440955));

    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(40.0));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(3.440955));
    REQUIRE_EQUALS_APPROX(c(1, 2).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 2).imag(), T(0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 3).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 3).imag(), T(-0.8123));
    REQUIRE_EQUALS_APPROX(c(1, 4).real(), T(-2.5));
    REQUIRE_EQUALS_APPROX(c(1, 4).imag(), T(-3.440955));
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

    REQUIRE_EQUALS_APPROX(a(0).real(), Z(0.5));
    REQUIRE_EQUALS_APPROX(a(0).imag(), Z(5.5));
    REQUIRE_EQUALS_APPROX(a(1).real(), Z(5.626178));
    REQUIRE_EQUALS_APPROX(a(1).imag(), Z(0.376206));
    REQUIRE_EQUALS_APPROX(a(2).real(), Z(-1.067916));
    REQUIRE_EQUALS_APPROX(a(2).imag(), Z(-2.571198));
    REQUIRE_EQUALS_APPROX(a(3).real(), Z(-2.831271));
    REQUIRE_EQUALS_APPROX(a(3).imag(), Z(-2.709955));
    REQUIRE_EQUALS_APPROX(a(4).real(), Z(2.773009));
    REQUIRE_EQUALS_APPROX(a(4).imag(), Z(4.404947));
}

TEMPLATE_TEST_CASE_2("fft_1d_many/4", "[fast][fft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 2, 5> a{MZ(1.0, 0.0), MZ(2.0, 0.0), MZ(3.0, 0.0), MZ(4.0, 0.0), MZ(5.0, 0.0), MZ(6.0, 0.0), MZ(7.0, 0.0), MZ(8.0, 0.0), MZ(9.0, 0.0), MZ(10.0, 0.0)};

    a.fft_many_inplace();

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), Z(15.0));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), Z(3.440955));
    REQUIRE_EQUALS_APPROX(a(0, 2).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(0, 2).imag(), Z(0.8123));
    REQUIRE_EQUALS_APPROX(a(0, 3).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(0, 3).imag(), Z(-0.8123));
    REQUIRE_EQUALS_APPROX(a(0, 4).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(0, 4).imag(), Z(-3.440955));

    REQUIRE_EQUALS_APPROX(a(1, 0).real(), Z(40.0));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), Z(3.440955));
    REQUIRE_EQUALS_APPROX(a(1, 2).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(1, 2).imag(), Z(0.8123));
    REQUIRE_EQUALS_APPROX(a(1, 3).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(1, 3).imag(), Z(-0.8123));
    REQUIRE_EQUALS_APPROX(a(1, 4).real(), Z(-2.5));
    REQUIRE_EQUALS_APPROX(a(1, 4).imag(), Z(-3.440955));
}

TEMPLATE_TEST_CASE_2("fft_1d_many/5", "[fast][fft]", Z, float, double) {
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1033> a;
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1033> c_1;
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1033> c_2;

    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 1033; ++j) {
            a(i, j) = std::complex<Z>(Z(i * j / 2.0), Z(0));
        }
    }

    c_1 = etl::fft_1d_many(a);

    SELECTED_SECTION(etl::fft_impl::STD) {
        for (size_t i = 0; i < 7; ++i) {
            c_2(i) = etl::fft_1d(a(i));
        }
    }

#ifdef ETL_CUFFT_MODE
    auto eps = base_eps_etl_large * 2;
#else
    auto eps = base_eps * 10;
#endif

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(c_1[i].real(), c_2[i].real(), eps);
        REQUIRE_EQUALS_APPROX_E(c_1[i].imag(), c_2[i].imag(), eps);
    }
}

TEMPLATE_TEST_CASE_2("fft_1d_many/6", "[fast][fft]", Z, float, double) {
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1045> a;
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1045> c_1;
    etl::fast_dyn_matrix<std::complex<Z>, 7, 1045> c_2;

    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 1045; ++j) {
            a(i, j) = std::complex<Z>(Z(i * j), Z(0));
        }
    }

    c_1 = etl::fft_1d_many(a);

    SELECTED_SECTION(etl::fft_impl::STD) {
        for (size_t i = 0; i < 7; ++i) {
            c_2(i) = etl::fft_1d(a(i));
        }
    }

#ifdef ETL_CUFFT_MODE
    auto eps = base_eps_etl_large * 2;
#else
    auto eps = base_eps * 10;
#endif

    for(size_t i = 0; i < a.size(); ++i){
        REQUIRE_EQUALS_APPROX_E(c_1[i].real(), c_2[i].real(), eps);
        REQUIRE_EQUALS_APPROX_E(c_1[i].imag(), c_2[i].imag(), eps);
    }
}
