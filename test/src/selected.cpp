//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEMPLATE_TEST_CASE_2("selected/1", "[selected]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = etl::selected<etl::sum_impl, etl::sum_impl::STD>(a + a);

    REQUIRE(b[0] == 2.0);
    REQUIRE(b[1] == -4.0);
    REQUIRE(b[2] == 6.0);
}

TEMPLATE_TEST_CASE_2("selected/2", "[selected]", Z, float, double) {
    etl::fast_matrix<Z, 8> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<std::complex<Z>, 8> c;

    c = etl::selected<etl::fft_impl, etl::fft_impl::STD>(etl::fft_1d(a));

    REQUIRE(c(0).real() == Approx(Z(4.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(1.0)));
    REQUIRE(c(1).imag() == Approx(Z(-2.41421)));
    REQUIRE(c(2).real() == Approx(Z(0.0)));
    REQUIRE(c(2).imag() == Approx(Z(0.0)));
    REQUIRE(c(3).real() == Approx(Z(1.0)));
    REQUIRE(c(3).imag() == Approx(Z(-0.41421)));
    REQUIRE(c(4).real() == Approx(Z(0.0)));
    REQUIRE(c(4).imag() == Approx(Z(0.0)));
    REQUIRE(c(5).real() == Approx(Z(1.0)));
    REQUIRE(c(5).imag() == Approx(Z(0.41421)));
    REQUIRE(c(6).real() == Approx(Z(0.0)));
    REQUIRE(c(6).imag() == Approx(Z(0.0)));
    REQUIRE(c(7).real() == Approx(Z(1.0)));
    REQUIRE(c(7).imag() == Approx(Z(2.41421)));
}

TEMPLATE_TEST_CASE_2("selected/3", "[selected]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    b = selected_helper(etl::conv_impl::SSE, a + a);

    REQUIRE(b[0] == 2.0);
    REQUIRE(b[1] == -4.0);
    REQUIRE(b[2] == 6.0);
}

TEMPLATE_TEST_CASE_2("selected/4", "[selected]", Z, float, double) {
    etl::fast_matrix<Z, 8> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<std::complex<Z>, 8> c;

    c = selected_helper(etl::fft_impl::STD, etl::fft_1d(a));

    REQUIRE(c(0).real() == Approx(Z(4.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(1.0)));
    REQUIRE(c(1).imag() == Approx(Z(-2.41421)));
    REQUIRE(c(2).real() == Approx(Z(0.0)));
    REQUIRE(c(2).imag() == Approx(Z(0.0)));
    REQUIRE(c(3).real() == Approx(Z(1.0)));
    REQUIRE(c(3).imag() == Approx(Z(-0.41421)));
    REQUIRE(c(4).real() == Approx(Z(0.0)));
    REQUIRE(c(4).imag() == Approx(Z(0.0)));
    REQUIRE(c(5).real() == Approx(Z(1.0)));
    REQUIRE(c(5).imag() == Approx(Z(0.41421)));
    REQUIRE(c(6).real() == Approx(Z(0.0)));
    REQUIRE(c(6).imag() == Approx(Z(0.0)));
    REQUIRE(c(7).real() == Approx(Z(1.0)));
    REQUIRE(c(7).imag() == Approx(Z(2.41421)));
}

TEMPLATE_TEST_CASE_2("selected/5", "[selected]", Z, float, double) {
    etl::fast_vector<Z, 3> a({1.0, -2.0, 3.0});
    etl::fast_vector<Z, 3> b;

    SELECTED_SECTION(etl::dot_impl::STD){
        b = a + a;
    }

    REQUIRE(b[0] == 2.0);
    REQUIRE(b[1] == -4.0);
    REQUIRE(b[2] == 6.0);
}

TEMPLATE_TEST_CASE_2("selected/6", "[selected]", Z, float, double) {
    etl::fast_matrix<Z, 8> a({1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    etl::fast_matrix<std::complex<Z>, 8> c;

    SELECTED_SECTION(etl::fft_impl::STD){
        c = etl::fft_1d(a);
    }

    REQUIRE(c(0).real() == Approx(Z(4.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(1.0)));
    REQUIRE(c(1).imag() == Approx(Z(-2.41421)));
    REQUIRE(c(2).real() == Approx(Z(0.0)));
    REQUIRE(c(2).imag() == Approx(Z(0.0)));
    REQUIRE(c(3).real() == Approx(Z(1.0)));
    REQUIRE(c(3).imag() == Approx(Z(-0.41421)));
    REQUIRE(c(4).real() == Approx(Z(0.0)));
    REQUIRE(c(4).imag() == Approx(Z(0.0)));
    REQUIRE(c(5).real() == Approx(Z(1.0)));
    REQUIRE(c(5).imag() == Approx(Z(0.41421)));
    REQUIRE(c(6).real() == Approx(Z(0.0)));
    REQUIRE(c(6).imag() == Approx(Z(0.0)));
    REQUIRE(c(7).real() == Approx(Z(1.0)));
    REQUIRE(c(7).imag() == Approx(Z(2.41421)));
}

TEMPLATE_TEST_CASE_2("selected/7", "[selected]", Z, float, double) {
    REQUIRE(!etl::local_context().fft_selector.forced);

    SELECTED_SECTION(etl::fft_impl::MKL) {
        REQUIRE(etl::local_context().fft_selector.forced);
        REQUIRE(etl::local_context().fft_selector.impl == etl::fft_impl::MKL);

        etl::local_context().fft_selector.forced = false;
        etl::local_context().fft_selector.impl = etl::fft_impl::CUFFT;;

        SELECTED_SECTION(etl::fft_impl::STD) {
            REQUIRE(etl::local_context().fft_selector.forced);
            REQUIRE(etl::local_context().fft_selector.impl == etl::fft_impl::STD);
        }

        REQUIRE(!etl::local_context().fft_selector.forced);
        REQUIRE(etl::local_context().fft_selector.impl == etl::fft_impl::CUFFT);
    }

    REQUIRE(!etl::local_context().fft_selector.forced);
}
