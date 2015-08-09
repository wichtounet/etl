//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include <functional>

TEMPLATE_TEST_CASE_2( "experimental/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2> a;
    etl::fast_matrix<std::complex<Z>, 2> c1;
    etl::fast_matrix<std::complex<Z>, 2> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/4", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3> a;
    etl::fast_matrix<std::complex<Z>, 3> c1;
    etl::fast_matrix<std::complex<Z>, 3> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(3.0, -3.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 4> a;
    etl::fast_matrix<std::complex<Z>, 4> c1;
    etl::fast_matrix<std::complex<Z>, 4> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 8> a;
    etl::fast_matrix<std::complex<Z>, 8> c1;
    etl::fast_matrix<std::complex<Z>, 8> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);
    a[6] = std::complex<Z>(2.0, -1.0);
    a[7] = std::complex<Z>(4.0, 3.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/5", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 6> a;
    etl::fast_matrix<std::complex<Z>, 6> c1;
    etl::fast_matrix<std::complex<Z>, 6> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/6", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 11> a;
    etl::fast_matrix<std::complex<Z>, 11> c1;
    etl::fast_matrix<std::complex<Z>, 11> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);
    a[6] = std::complex<Z>(1.0, 1.0);
    a[7] = std::complex<Z>(2.0, 3.0);
    a[8] = std::complex<Z>(2.0, -1.0);
    a[9] = std::complex<Z>(4.0, 3.0);
    a[10] = std::complex<Z>(1.0, 1.0);

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()).epsilon(0.01));
        CHECK(c1[i].imag() == Approx(c2[i].imag()).epsilon(0.01));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/7", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 7 * 11> a;
    etl::fast_matrix<std::complex<Z>, 7 * 11> c1;
    etl::fast_matrix<std::complex<Z>, 7 * 11> c2;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<Z> dist(-140.0, 250.0);
    auto d = std::bind(dist, gen);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        a[i].real(d());
        a[i].imag(d());
    }

    c1 = etl::fft_1d(a);

    etl::impl::standard::fftn1(a, c2);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()).epsilon(0.01));
        CHECK(c1[i].imag() == Approx(c2[i].imag()).epsilon(0.01));
    }
}
