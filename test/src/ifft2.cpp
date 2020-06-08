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

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(c(2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(c(2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(c(2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(c(2, 1).imag(), T(-1.5282));
}

IFFT2_TEST_CASE("ifft_2d_c/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 2> a;
    etl::fast_matrix<std::complex<T>, 2, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(c(0, 0).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).real(), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1).imag(), T(-0.5));
    REQUIRE_EQUALS_APPROX(c(1, 0).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(c(1, 1).real(), T(-0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1).imag(), T(-0.5));

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), T(2.0));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), T(3.0));
    REQUIRE_EQUALS_APPROX(a(1, 0).real(), T(-1.0));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), T(-2.0));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), T(0.0));
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

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(a(1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(a(2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(a(2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(a(2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(a(2, 1).imag(), T(-1.5282));
}

//ifft_2d (real)

IFFT2_REAL_TEST_CASE("ifft_2d_c_real/1", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 3, 2> a;
    etl::fast_matrix<T, 3, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(-2.0, 0.0);
    a[2] = std::complex<T>(3.5, 1.5);
    a[3] = std::complex<T>(-4.0, -4.0);
    a[4] = std::complex<T>(5.0, 0.5);
    a[5] = std::complex<T>(6.5, 1.25);

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(a[0].real(), T(1.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a[1].real(), T(-2.0));
    REQUIRE_EQUALS_APPROX(a[1].imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), T(3.5));
    REQUIRE_EQUALS_APPROX(a[2].imag(), T(1.5));
    REQUIRE_EQUALS_APPROX(a[3].real(), T(-4.0));
    REQUIRE_EQUALS_APPROX(a[3].imag(), T(-4.0));
    REQUIRE_EQUALS_APPROX(a[4].real(), T(5.0));
    REQUIRE_EQUALS_APPROX(a[4].imag(), T(0.5));
    REQUIRE_EQUALS_APPROX(a[5].real(), T(6.5));
    REQUIRE_EQUALS_APPROX(a[5].imag(), T(1.25));

    REQUIRE_EQUALS_APPROX(c(0, 0), T(1.66667));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(-0.4699));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(-0.9021));
    REQUIRE_EQUALS_APPROX(c(2, 0), T(-1.6968));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(0.9021));
}

IFFT2_REAL_TEST_CASE("ifft_2d_real_c/2", "[fast][ifft]") {
    etl::fast_matrix<std::complex<T>, 2, 2> a;
    etl::fast_matrix<T, 2, 2> c;

    a[0] = std::complex<T>(1.0, 1.0);
    a[1] = std::complex<T>(2.0, 3.0);
    a[2] = std::complex<T>(-1.0, 0.0);
    a[3] = std::complex<T>(-2.0, 0.0);

    Impl::apply(a, c);

    REQUIRE_EQUALS_APPROX(a[0].real(), T(1.0));
    REQUIRE_EQUALS_APPROX(a[0].imag(), T(1.0));
    REQUIRE_EQUALS_APPROX(a[1].real(), T(2.0));
    REQUIRE_EQUALS_APPROX(a[1].imag(), T(3.0));
    REQUIRE_EQUALS_APPROX(a[2].real(), T(-1.0));
    REQUIRE_EQUALS_APPROX(a[2].imag(), T(0.0));
    REQUIRE_EQUALS_APPROX(a[3].real(), T(-2.0));
    REQUIRE_EQUALS_APPROX(a[3].imag(), T(0.0));

    REQUIRE_EQUALS_APPROX(c(0, 0), T(0.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(0.0));
    REQUIRE_EQUALS_APPROX(c(1, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(-0.5));
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

    REQUIRE_EQUALS_APPROX(a(0, 0).real(), Z(1.66667));
    REQUIRE_EQUALS_APPROX(a(0, 0).imag(), Z(0.04167));
    REQUIRE_EQUALS_APPROX(a(0, 1).real(), Z(1.5));
    REQUIRE_EQUALS_APPROX(a(0, 1).imag(), Z(0.95833));
    REQUIRE_EQUALS_APPROX(a(1, 0).real(), Z(-0.4699));
    REQUIRE_EQUALS_APPROX(a(1, 0).imag(), Z(-1.5029));
    REQUIRE_EQUALS_APPROX(a(1, 1).real(), Z(-0.9021));
    REQUIRE_EQUALS_APPROX(a(1, 1).imag(), Z(1.06987));
    REQUIRE_EQUALS_APPROX(a(2, 0).real(), Z(-1.6968));
    REQUIRE_EQUALS_APPROX(a(2, 0).imag(), Z(1.9612));
    REQUIRE_EQUALS_APPROX(a(2, 1).real(), Z(0.9021));
    REQUIRE_EQUALS_APPROX(a(2, 1).imag(), Z(-1.5282));
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

    REQUIRE_EQUALS_APPROX(c(0, 0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(c(0, 0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(c(0, 2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(c(0, 2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1).imag(), T(-1.5282));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(c(1, 0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(c(1, 2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(c(1, 2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1).imag(), T(-1.5282));
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

    REQUIRE_EQUALS_APPROX(c(0, 0, 0).real(), T(0.75));
    REQUIRE_EQUALS_APPROX(c(0, 0, 0).imag(), T(0.70833));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1).real(), T(1.91666));
    REQUIRE_EQUALS_APPROX(c(0, 0, 1).imag(), T(0.625));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0).real(), T(-0.194578));
    REQUIRE_EQUALS_APPROX(c(0, 1, 0).imag(), T(-1.51404));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1).real(), T(-0.138755));
    REQUIRE_EQUALS_APPROX(c(0, 1, 1).imag(), T(-0.09599));
    REQUIRE_EQUALS_APPROX(c(0, 2, 0).real(), T(-0.55542));
    REQUIRE_EQUALS_APPROX(c(0, 2, 0).imag(), T(1.80571));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1).real(), T(0.22208));
    REQUIRE_EQUALS_APPROX(c(0, 2, 1).imag(), T(-0.5290));

    REQUIRE_EQUALS_APPROX(c(1, 0, 0).real(), T(2.0));
    REQUIRE_EQUALS_APPROX(c(1, 0, 0).imag(), T(0.20833));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1).real(), T(-1.83333));
    REQUIRE_EQUALS_APPROX(c(1, 0, 1).imag(), T(-1.54167));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0).real(), T(-0.79127));
    REQUIRE_EQUALS_APPROX(c(1, 1, 0).imag(), T(-1.25887));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1).real(), T(1.20793));
    REQUIRE_EQUALS_APPROX(c(1, 1, 1).imag(), T(-0.45085));
    REQUIRE_EQUALS_APPROX(c(1, 2, 0).real(), T(0.29127));
    REQUIRE_EQUALS_APPROX(c(1, 2, 0).imag(), T(1.05053));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1).real(), T(0.1254));
    REQUIRE_EQUALS_APPROX(c(1, 2, 1).imag(), T(0.99252));
}

//ifft many inplace

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

    REQUIRE_EQUALS_APPROX(a(0, 0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(a(0, 0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(a(0, 0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(0, 0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(a(0, 1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(a(0, 1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(a(0, 1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(a(0, 1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(a(0, 2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(a(0, 2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(a(0, 2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(a(0, 2, 1).imag(), T(-1.5282));

    REQUIRE_EQUALS_APPROX(a(1, 0, 0).real(), T(1.66667));
    REQUIRE_EQUALS_APPROX(a(1, 0, 0).imag(), T(0.04167));
    REQUIRE_EQUALS_APPROX(a(1, 0, 1).real(), T(1.5));
    REQUIRE_EQUALS_APPROX(a(1, 0, 1).imag(), T(0.95833));
    REQUIRE_EQUALS_APPROX(a(1, 1, 0).real(), T(-0.4699));
    REQUIRE_EQUALS_APPROX(a(1, 1, 0).imag(), T(-1.5029));
    REQUIRE_EQUALS_APPROX(a(1, 1, 1).real(), T(-0.9021));
    REQUIRE_EQUALS_APPROX(a(1, 1, 1).imag(), T(1.06987));
    REQUIRE_EQUALS_APPROX(a(1, 2, 0).real(), T(-1.6968));
    REQUIRE_EQUALS_APPROX(a(1, 2, 0).imag(), T(1.9612));
    REQUIRE_EQUALS_APPROX(a(1, 2, 1).real(), T(0.9021));
    REQUIRE_EQUALS_APPROX(a(1, 2, 1).imag(), T(-1.5282));
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

    REQUIRE_EQUALS_APPROX(a(0, 0, 0).real(), T(0.75));
    REQUIRE_EQUALS_APPROX(a(0, 0, 0).imag(), T(0.70833));
    REQUIRE_EQUALS_APPROX(a(0, 0, 1).real(), T(1.91666));
    REQUIRE_EQUALS_APPROX(a(0, 0, 1).imag(), T(0.625));
    REQUIRE_EQUALS_APPROX(a(0, 1, 0).real(), T(-0.194578));
    REQUIRE_EQUALS_APPROX(a(0, 1, 0).imag(), T(-1.51404));
    REQUIRE_EQUALS_APPROX(a(0, 1, 1).real(), T(-0.138755));
    REQUIRE_EQUALS_APPROX(a(0, 1, 1).imag(), T(-0.09599));
    REQUIRE_EQUALS_APPROX(a(0, 2, 0).real(), T(-0.55542));
    REQUIRE_EQUALS_APPROX(a(0, 2, 0).imag(), T(1.80571));
    REQUIRE_EQUALS_APPROX(a(0, 2, 1).real(), T(0.22208));
    REQUIRE_EQUALS_APPROX(a(0, 2, 1).imag(), T(-0.5290));

    REQUIRE_EQUALS_APPROX(a(1, 0, 0).real(), T(2.0));
    REQUIRE_EQUALS_APPROX(a(1, 0, 0).imag(), T(0.20833));
    REQUIRE_EQUALS_APPROX(a(1, 0, 1).real(), T(-1.83333));
    REQUIRE_EQUALS_APPROX(a(1, 0, 1).imag(), T(-1.54167));
    REQUIRE_EQUALS_APPROX(a(1, 1, 0).real(), T(-0.79127));
    REQUIRE_EQUALS_APPROX(a(1, 1, 0).imag(), T(-1.25887));
    REQUIRE_EQUALS_APPROX(a(1, 1, 1).real(), T(1.20793));
    REQUIRE_EQUALS_APPROX(a(1, 1, 1).imag(), T(-0.45085));
    REQUIRE_EQUALS_APPROX(a(1, 2, 0).real(), T(0.29127));
    REQUIRE_EQUALS_APPROX(a(1, 2, 0).imag(), T(1.05053));
    REQUIRE_EQUALS_APPROX(a(1, 2, 1).real(), T(0.1254));
    REQUIRE_EQUALS_APPROX(a(1, 2, 1).imag(), T(0.99252));
}
