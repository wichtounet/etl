//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"
#include "template_test.hpp"

#include "etl/etl.hpp"

//{{{ fft_1d

TEMPLATE_TEST_CASE_2( "fft_1d/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<Z, 5> a({1.0, 2.0, 3.0, 4.0, 5.0});
    etl::fast_matrix<std::complex<Z>, 5> c;

    c = etl::fft_1d(a);

    REQUIRE(c(0).real() == Approx(Z(15.0)));
    REQUIRE(c(0).imag() == Approx(Z(0.0)));
    REQUIRE(c(1).real() == Approx(Z(-2.5)));
    REQUIRE(c(1).imag() == Approx(Z(3.440955)));
    REQUIRE(c(2).real() == Approx(Z(-2.5)));
    REQUIRE(c(2).imag() == Approx(Z(0.8123)));
    REQUIRE(c(3).real() == Approx(Z(-2.5)));
    REQUIRE(c(3).imag() == Approx(Z(-0.8123)));
    REQUIRE(c(4).real() == Approx(Z(-2.5)));
    REQUIRE(c(4).imag() == Approx(Z(-3.440955)));
}

//}}}
