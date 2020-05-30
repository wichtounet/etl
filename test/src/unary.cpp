//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

#include <cmath>

#define CZ(a, b) std::complex<Z>(a, b)
#define ECZ(a, b) etl::complex<Z>(a, b)

TEMPLATE_TEST_CASE_2("log/0", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log/1", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log/2", "[log]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {3.0, 2.0, 5.0, 1.0, 1.2, 2.2, 3.2, -4.2};

    etl::fast_matrix<Z, 2, 2, 2> d;

    d = log(a);

    REQUIRE_EQUALS_APPROX(d[0], std::log(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::log(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::log(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::log(Z(1.2)));
    REQUIRE_EQUALS_APPROX(d[5], std::log(Z(2.2)));
    REQUIRE_EQUALS_APPROX(d[6], std::log(Z(3.2)));
    REQUIRE_DIRECT(std::isnan(d[7]));
}

TEMPLATE_TEST_CASE_2("log/3", "[log]", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), std::log(a[0]).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), std::log(a[0]).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), std::log(a[1]).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), std::log(a[1]).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), std::log(a[2]).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), std::log(a[2]).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), std::log(a[3]).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), std::log(a[3]).imag());
}

TEMPLATE_TEST_CASE_2("log/4", "[log]", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::log(a[0]).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::log(a[0]).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::log(a[1]).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::log(a[1]).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::log(a[2]).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::log(a[2]).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::log(a[3]).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::log(a[3]).imag);
}

TEMPLATE_TEST_CASE_2("log2/0", "[log2]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = log2(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log2(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log2(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log2/1", "[log2]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log2(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log2(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log2(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log2/2", "[log2]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {3.0, 2.0, 5.0, 1.0, 1.2, 2.2, 3.2, -4.2};

    etl::fast_matrix<Z, 2, 2, 2> d;

    d = log2(a);

    REQUIRE_EQUALS_APPROX(d[0], std::log2(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::log2(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log2(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::log2(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::log2(Z(1.2)));
    REQUIRE_EQUALS_APPROX(d[5], std::log2(Z(2.2)));
    REQUIRE_EQUALS_APPROX(d[6], std::log2(Z(3.2)));
    REQUIRE_DIRECT(std::isnan(d[7]));
}

namespace {

template<typename T>
std::complex<T> my_log2(std::complex<T> z){
    return std::log(z) / std::log(std::complex<T>{T(2)});
}

template<typename T>
std::complex<T> my_cbrt(std::complex<T> z){
    auto z_abs = std::abs(z);
    auto z_arg = std::arg(z);

    auto new_abs = std::cbrt(z_abs);
    auto new_arg = z_arg / 3.0f;

    return {new_abs * std::cos(new_arg), new_abs * std::sin(new_arg)};
}

} // end of anonymous namespace

TEMPLATE_TEST_CASE_2("log2/3", "[log2]", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log2(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), my_log2(a[0]).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), my_log2(a[0]).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), my_log2(a[1]).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), my_log2(a[1]).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), my_log2(a[2]).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), my_log2(a[2]).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), my_log2(a[3]).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), my_log2(a[3]).imag());
}

TEMPLATE_TEST_CASE_2("log2/4", "[log2]", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log2(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::log2(a[0]).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::log2(a[0]).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::log2(a[1]).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::log2(a[1]).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::log2(a[2]).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::log2(a[2]).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::log2(a[3]).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::log2(a[3]).imag);
}

TEMPLATE_TEST_CASE_2("log10/0", "[log10]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = log10(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log10(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log10(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log10/1", "[log10]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log10(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::log10(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log10(Z(5.0)));
}

TEMPLATE_TEST_CASE_2("log10/2", "[log10]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {3.0, 2.0, 5.0, 1.0, 1.2, 2.2, 3.2, -4.2};

    etl::fast_matrix<Z, 2, 2, 2> d;

    d = log10(a);

    REQUIRE_EQUALS_APPROX(d[0], std::log10(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::log10(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::log10(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::log10(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::log10(Z(1.2)));
    REQUIRE_EQUALS_APPROX(d[5], std::log10(Z(2.2)));
    REQUIRE_EQUALS_APPROX(d[6], std::log10(Z(3.2)));
    REQUIRE_DIRECT(std::isnan(d[7]));
}

TEMPLATE_TEST_CASE_2("log10/3", "[log10]", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log10(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), std::log10(a[0]).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), std::log10(a[0]).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), std::log10(a[1]).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), std::log10(a[1]).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), std::log10(a[2]).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), std::log10(a[2]).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), std::log10(a[3]).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), std::log10(a[3]).imag());
}

TEMPLATE_TEST_CASE_2("log10/4", "[log10]", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = log10(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::log10(a[0]).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::log10(a[0]).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::log10(a[1]).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::log10(a[1]).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::log10(a[2]).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::log10(a[2]).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::log10(a[3]).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::log10(a[3]).imag);
}

TEMPLATE_TEST_CASE_2("sqrt/1", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("sqrt/2", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("sqrt/3", "fast_matrix::sqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], std::sqrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::sqrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::sqrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("sqrt/4", "fast_matrix::sqrt", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), std::sqrt(a[0]).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), std::sqrt(a[0]).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), std::sqrt(a[1]).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), std::sqrt(a[1]).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), std::sqrt(a[2]).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), std::sqrt(a[2]).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), std::sqrt(a[3]).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), std::sqrt(a[3]).imag());
}

TEMPLATE_TEST_CASE_2("sqrt/5", "fast_matrix::sqrt", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = sqrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::sqrt(a[0]).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::sqrt(a[0]).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::sqrt(a[1]).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::sqrt(a[1]).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::sqrt(a[2]).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::sqrt(a[2]).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::sqrt(a[3]).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::sqrt(a[3]).imag);
}

TEMPLATE_TEST_CASE_2("invsqrt/1", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = invsqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invsqrt/2", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a);

    REQUIRE_DIRECT(std::isnan(d[0]));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invsqrt/3", "fast_matrix::invsqrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::sqrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::sqrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::sqrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::sqrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invsqrt/4", "fast_matrix::sqrt", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), (Z(1.0) / std::sqrt(a[0])).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), (Z(1.0) / std::sqrt(a[0])).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), (Z(1.0) / std::sqrt(a[1])).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), (Z(1.0) / std::sqrt(a[1])).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), (Z(1.0) / std::sqrt(a[2])).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), (Z(1.0) / std::sqrt(a[2])).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), (Z(1.0) / std::sqrt(a[3])).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), (Z(1.0) / std::sqrt(a[3])).imag());
}

TEMPLATE_TEST_CASE_2("invsqrt/5", "fast_matrix::sqrt", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 1.0), Z(2.0, 3.0), Z(5.0, 2.0), Z(1.0, 1.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invsqrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::invsqrt(a[0]).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::invsqrt(a[0]).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::invsqrt(a[1]).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::invsqrt(a[1]).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::invsqrt(a[2]).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::invsqrt(a[2]).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::invsqrt(a[3]).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::invsqrt(a[3]).imag);
}

TEMPLATE_TEST_CASE_2("cbrt/1", "[cbrt]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("cbrt/2", "[cbrt]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("cbrt/3", "[cbrt]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], std::cbrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::cbrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::cbrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("cbrt/4", "[cbrt]", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 0.1), Z(2.0, 0.1), Z(5.0, 0.1), Z(1.0, 0.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), my_cbrt(Z(-1.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), my_cbrt(Z(-1.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), my_cbrt(Z(2.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), my_cbrt(Z(2.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), my_cbrt(Z(5.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), my_cbrt(Z(5.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), my_cbrt(Z(1.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), my_cbrt(Z(1.0, 0.1)).imag());
}

TEMPLATE_TEST_CASE_2("cbrt/5", "[cbrt]", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 0.1), Z(2.0, 0.1), Z(5.0, 0.1), Z(1.0, 0.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::cbrt(Z(-1.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::cbrt(Z(-1.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::cbrt(Z(2.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::cbrt(Z(2.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::cbrt(Z(5.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::cbrt(Z(5.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::cbrt(Z(1.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::cbrt(Z(1.0, 0.1)).imag);
}

TEMPLATE_TEST_CASE_2("invcbrt/1", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = invcbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invcbrt/2", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invcbrt(a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(5.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invcbrt/3", "fast_matrix::invcbrt", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 1> a = {-1.0, 2.0, 5.0, 1.0};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = invcbrt(a >> a);

    REQUIRE_EQUALS_APPROX(d[0], Z(1) / std::cbrt(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[1], Z(1) / std::cbrt(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[2], Z(1) / std::cbrt(Z(25.0)));
    REQUIRE_EQUALS_APPROX(d[3], Z(1) / std::cbrt(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("invcbrt/4", "[cbrt]", Z, std::complex<float>, std::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 0.1), Z(2.0, 0.1), Z(5.0, 0.1), Z(1.0, 0.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real(), my_cbrt(Z(-1.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[0].imag(), my_cbrt(Z(-1.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[1].real(), my_cbrt(Z(2.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[1].imag(), my_cbrt(Z(2.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[2].real(), my_cbrt(Z(5.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[2].imag(), my_cbrt(Z(5.0, 0.1)).imag());
    REQUIRE_EQUALS_APPROX(d[3].real(), my_cbrt(Z(1.0, 0.1)).real());
    REQUIRE_EQUALS_APPROX(d[3].imag(), my_cbrt(Z(1.0, 0.1)).imag());
}

TEMPLATE_TEST_CASE_2("invcbrt/5", "[cbrt]", Z, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 2, 2, 1> a = {Z(-1.0, 0.1), Z(2.0, 0.1), Z(5.0, 0.1), Z(1.0, 0.1)};

    etl::fast_matrix<Z, 2, 2, 1> d;
    d = cbrt(a);

    REQUIRE_EQUALS_APPROX(d[0].real, etl::cbrt(Z(-1.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[0].imag, etl::cbrt(Z(-1.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[1].real, etl::cbrt(Z(2.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[1].imag, etl::cbrt(Z(2.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[2].real, etl::cbrt(Z(5.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[2].imag, etl::cbrt(Z(5.0, 0.1)).imag);
    REQUIRE_EQUALS_APPROX(d[3].real, etl::cbrt(Z(1.0, 0.1)).real);
    REQUIRE_EQUALS_APPROX(d[3].imag, etl::cbrt(Z(1.0, 0.1)).imag);
}

TEMPLATE_TEST_CASE_2("abs/0", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 4> a = {-1.0, 2.0, 0.0, 1.0, 1.5, -3.2, 1.1, -2.3};

    etl::fast_matrix<Z, 2, 4> d;
    d = abs(a);

    REQUIRE_EQUALS(d[0], Z(1.0));
    REQUIRE_EQUALS(d[1], Z(2.0));
    REQUIRE_EQUALS(d[2], Z(0.0));
    REQUIRE_EQUALS(d[3], Z(1.0));
    REQUIRE_EQUALS(d[4], Z(1.5));
    REQUIRE_EQUALS(d[5], Z(3.2));
    REQUIRE_EQUALS(d[6], Z(1.1));
    REQUIRE_EQUALS(d[7], Z(2.3));
}

TEMPLATE_TEST_CASE_2("fast_matrix/sign", "fast_matrix::sign", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sign(a);

    REQUIRE_EQUALS(d[0], -1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_unary", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 3.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(sign(a));

    REQUIRE_EQUALS(d[0], 1.0);
    REQUIRE_EQUALS(d[1], 1.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_binary_1", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(a + a);

    REQUIRE_EQUALS(d[0], 2.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("fast_matrix/unary_binary_2", "fast_matrix::abs", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = abs(a) + a;

    REQUIRE_EQUALS(d[0], 0.0);
    REQUIRE_EQUALS(d[1], 4.0);
    REQUIRE_EQUALS(d[2], 0.0);
}

TEMPLATE_TEST_CASE_2("sigmoid/forward/0", "[fast][ml]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = sigmoid(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::logistic_sigmoid(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::logistic_sigmoid(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("sigmoid/forward/1", "[fast][ml]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    // Inplace should work!
    a = sigmoid(a);

    REQUIRE_EQUALS_APPROX(a[0], etl::math::logistic_sigmoid(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(a[1], etl::math::logistic_sigmoid(Z(2.0)));
    REQUIRE_EQUALS_APPROX(a[2], etl::math::logistic_sigmoid(Z(0.0)));
    REQUIRE_EQUALS_APPROX(a[3], etl::math::logistic_sigmoid(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("fast_matrix/softplus", "fast_matrix::softplus", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;

    d = softplus(a);

    REQUIRE_EQUALS_APPROX(d[0], etl::math::softplus(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], etl::math::softplus(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], etl::math::softplus(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], etl::math::softplus(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("exp/0", "[exp]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::exp(Z(1.0)));
}

TEMPLATE_TEST_CASE_2("exp/1", "[exp]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2, 2> a = {-1.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.1, 6.1};

    etl::fast_matrix<Z, 2, 2, 2> d;
    d = exp(a);

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)));
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)));
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(0.0)));
    REQUIRE_EQUALS_APPROX(d[3], std::exp(Z(1.0)));
    REQUIRE_EQUALS_APPROX(d[4], std::exp(Z(3.0)));
    REQUIRE_EQUALS_APPROX(d[5], std::exp(Z(4.0)));
    REQUIRE_EQUALS_APPROX(d[6], std::exp(Z(5.1)));
    REQUIRE_EQUALS_APPROX(d[7], std::exp(Z(6.1)));
}

constexpr bool binary(double a) {
    return a == 0.0 || a == 1.0;
}

TEMPLATE_TEST_CASE_2("fast_matrix/bernoulli/0", "fast_matrix::bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = etl::bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}

TEMPLATE_TEST_CASE_2("fast_matrix/bernoulli/1", "fast_matrix::bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = etl::state_bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}

TEMPLATE_TEST_CASE_2("fast_matrix/bernoulli/2", "fast_matrix::bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    auto states = std::make_shared<void*>();
    etl::fast_matrix<Z, 2, 2> d;
    d = etl::state_bernoulli(a, states);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}

TEMPLATE_TEST_CASE_2("fast_matrix/r_bernoulli", "fast_matrix::r_bernoulli", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a = {-1.0, 2.0, 0.0, 1.0};

    etl::fast_matrix<Z, 2, 2> d;
    d = etl::r_bernoulli(a);

    REQUIRE_DIRECT(binary(d[0]));
    REQUIRE_DIRECT(binary(d[1]));
    REQUIRE_DIRECT(binary(d[2]));
    REQUIRE_DIRECT(binary(d[3]));
}
