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
