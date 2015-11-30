//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

#define CZ(a, b) std::complex<Z>(a, b)
#define ECZ(a, b) etl::complex<Z>(a, b)

TEMPLATE_TEST_CASE_2("sparse/complex/init/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<std::complex<Z>> a(3, 2, std::initializer_list<std::complex<Z>>({CZ(1.0, 0.0), CZ(0.0, 0.0), CZ(0.0, 0.0), CZ(0.0, 0.0), CZ(-1.0, 2.0), CZ(0.0, 1.0)    }));

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 6);
    REQUIRE(a.non_zeros() == 3);

    REQUIRE(a.get(0, 0).real() == Z(1.0));
    REQUIRE(a.get(0, 0).imag() == Z(0.0));
    REQUIRE(a.get(0, 1).real() == Z(0.0));
    REQUIRE(a.get(0, 1).imag() == Z(0.0));
    REQUIRE(a.get(1, 0).real() == Z(0.0));
    REQUIRE(a.get(1, 0).imag() == Z(0.0));
    REQUIRE(a.get(1, 1).real() == Z(0.0));
    REQUIRE(a.get(1, 1).imag() == Z(0.0));
    REQUIRE(a.get(2, 0).real() == Z(-1.0));
    REQUIRE(a.get(2, 0).imag() == Z( 2.0));
    REQUIRE(a.get(2, 1).real() == Z(0.0));
    REQUIRE(a.get(2, 1).imag() == Z(1.0));
}
