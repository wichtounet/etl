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

//ifft_1d (complex)

IFFT1_TEST_CASE("ifft_1d_c/0", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 4> a;
    etl::fast_matrix<std::complex<T>, 4> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 2.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(0.5));
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

    REQUIRE_EQUALS_APPROX(c(0).real(), T(0.1));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(1.1));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(0.554602));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(0.880989));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-0.566254));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(-0.541991));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(-0.213583));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-0.51424));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(1.125236));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(0.075241));
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

    REQUIRE_EQUALS_APPROX(c(0).real(), T(0.166666));
    REQUIRE_EQUALS_APPROX(c(0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(1).real(), T(0.605662));
    REQUIRE_EQUALS_APPROX(c(1).imag(), T(0.333333));
    REQUIRE_EQUALS_APPROX(c(2).real(), T(-0.910684));
    REQUIRE_EQUALS_APPROX(c(2).imag(), T(0.183013));
    REQUIRE_EQUALS_APPROX(c(3).real(), T(0));
    REQUIRE_EQUALS_APPROX(c(3).imag(), T(-0.166667));
    REQUIRE_EQUALS_APPROX(c(4).real(), T(0.244017));
    REQUIRE_EQUALS_APPROX(c(4).imag(), T(-0.683013));
    REQUIRE_EQUALS_APPROX(c(5).real(), T(0.894338));
    REQUIRE_EQUALS_APPROX(c(5).imag(), T(0.333333));
}

IFFT1_TEST_CASE("ifft_1d_c/4", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 4> a;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 2.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(3.0, -3.0);

    Impl::apply(a, a);

    REQUIRE_EQUALS_APPROX(a(0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(a(0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(a(1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(a(2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(a(3).imag(), T(0.5));
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

    REQUIRE_EQUALS_APPROX(a(0).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(1).real(), T(2.0));
    REQUIRE_EQUALS_APPROX(a(1).imag(), T(3.0));
    REQUIRE_EQUALS_APPROX(a(2).real(), T(-1.0));
    REQUIRE_EQUALS_APPROX(a(2).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(3).real(), T(-2.0));
    REQUIRE_EQUALS_APPROX(a(3).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(4).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(4).imag(), T(1.5));

    REQUIRE_EQUALS_APPROX(c(0), T(0.1));
    REQUIRE_EQUALS_APPROX(c(1), T(0.554602));
    REQUIRE_EQUALS_APPROX(c(2), T(-0.566254));
    REQUIRE_EQUALS_APPROX(c(3), T(-0.213583));
    REQUIRE_EQUALS_APPROX(c(4), T(1.125236));
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

    REQUIRE_EQUALS_APPROX(a(0).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(1).real(), T(2.0));
    REQUIRE_EQUALS_APPROX(a(1).imag(), T(3.0));
    REQUIRE_EQUALS_APPROX(a(2).real(), T(-1.0));
    REQUIRE_EQUALS_APPROX(a(2).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(3).real(), T(-2.0));
    REQUIRE_EQUALS_APPROX(a(3).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(4).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(4).imag(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(5).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(5).imag(), T(0.5));

    REQUIRE_EQUALS_APPROX(c(0), T(0.166666));
    REQUIRE_EQUALS_APPROX(c(1), T(0.605662));
    REQUIRE_EQUALS_APPROX(c(2), T(-0.910684));
    REQUIRE_EQUALS_APPROX(c(3), T(0));
    REQUIRE_EQUALS_APPROX(c(4), T(0.244017));
    REQUIRE_EQUALS_APPROX(c(5), T(0.894338));
}

TEMPLATE_TEST_CASE_2("ifft_1d_c/3", "[fast][ifft]", Z, float, double) {
    etl::fast_matrix<std::complex<Z>, 4> a;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 2.0);
    a[2] = std::complex<Z>(-1.0, 0.0);
    a[3] = std::complex<Z>(3.0, -3.0);

    a.ifft_inplace();

    REQUIRE_EQUALS_APPROX(a(0).real(), Z(1.25));
    REQUIRE_EQUALS_APPROX(a(0).imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a(1).real(), Z(-0.75));
    REQUIRE_EQUALS_APPROX(a(1).imag(), Z(0.0));
    REQUIRE_EQUALS_APPROX(a(2).real(), Z(-1.25));
    REQUIRE_EQUALS_APPROX(a(2).imag(), Z(0.5));
    REQUIRE_EQUALS_APPROX(a(3).real(), Z(1.75));
    REQUIRE_EQUALS_APPROX(a(3).imag(), Z(0.5));
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

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(c(0, 2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(0, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(c(0, 3).imag(), T(0.5));

    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(c(1, 2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(c(1, 3).imag(), T(0.5));
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

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(-0.5));
    REQUIRE_EQUALS_APPROX(c(0, 2).real(), T(-0.25));
    REQUIRE_EQUALS_APPROX(c(0, 2).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(c(0, 3).imag(), T(-1.0));

    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(1.125));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(0.375));
    REQUIRE_EQUALS_APPROX(c(1, 2).real(), T(-1.125));
    REQUIRE_EQUALS_APPROX(c(1, 2).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 3).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 3).imag(), T(-0.375));
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

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(0, 2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(a(0, 2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(0, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(a(0, 3).imag(), T(0.5));

    REQUIRE_EQUALS_APPROX(a(1, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 2).real(), T(-1.25));
    REQUIRE_EQUALS_APPROX(a(1, 2).imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(1, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(a(1, 3).imag(), T(0.5));
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

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), T(1.25));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), T(-0.75));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), T(-0.5));
    REQUIRE_EQUALS_APPROX(a(0, 2).real(), T(-0.25));
    REQUIRE_EQUALS_APPROX(a(0, 2).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(0, 3).real(), T(1.75));
    REQUIRE_EQUALS_APPROX(a(0, 3).imag(), T(-1.0));

    REQUIRE_EQUALS_APPROX(a(1, 0).real(), T(1.125));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), T(0.5));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), T(0.375));
    REQUIRE_EQUALS_APPROX(a(1, 2).real(), T(-1.125));
    REQUIRE_EQUALS_APPROX(a(1, 2).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 3).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(1, 3).imag(), T(-0.375));
}
