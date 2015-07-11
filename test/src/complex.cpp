//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
//{{{ Complex content

TEMPLATE_TEST_CASE_2( "complex/1", "[std::complex]", Z, float, double ) {
    etl::fast_vector<std::complex<Z>, 3> a = {-1.0, 2.0, 5.0};

    REQUIRE(a[0].real() == Approx(Z(-1.0)));
    REQUIRE(a[0].imag() == Approx(Z(0.0)));
    REQUIRE(a[1].real() == Approx(Z(2.0)));
    REQUIRE(a[2].imag() == Approx(Z(0.0)));
    REQUIRE(a[2].real() == Approx(Z(5.0)));
    REQUIRE(a[2].imag() == Approx(Z(0.0)));

    a[0] = 33.0;

    REQUIRE(a[0].real() == Approx(Z(33.0)));
    REQUIRE(a[0].imag() == Approx(Z(0.0)));

    a[0].imag(12.0);

    REQUIRE(a[0].real() == Approx(Z(33.0)));
    REQUIRE(a[0].imag() == Approx(Z(12.0)));

    a[0] = std::complex<Z>(1.0, 2.0);

    REQUIRE(a[0].real() == Approx(Z(1.0)));
    REQUIRE(a[0].imag() == Approx(Z(2.0)));

    a = std::complex<Z>(3.0, -2.0);

    REQUIRE(a[0].real() == Approx(Z(3.0)));
    REQUIRE(a[0].imag() == Approx(Z(-2.0)));
    REQUIRE(a[1].real() == Approx(Z(3.0)));
    REQUIRE(a[1].imag() == Approx(Z(-2.0)));
    REQUIRE(a[2].real() == Approx(Z(3.0)));
    REQUIRE(a[2].imag() == Approx(Z(-2.0)));
}
