//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht // Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2( "compare/1", "[compare]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a(3.3);
    etl::fast_matrix<Z, 2, 2> b(3.3);
    etl::fast_matrix<Z, 2, 2> c(33.3);

    REQUIRE(a == a);
    REQUIRE(a == b);
    REQUIRE(b == a);
    REQUIRE(b == b);
    REQUIRE(!(a == c));
    REQUIRE(!(b == c));

    REQUIRE(!(a != a));
    REQUIRE(!(a != b));
    REQUIRE(!(b != a));
    REQUIRE(!(b != b));
    REQUIRE(a != c);
    REQUIRE(b != c);
}

TEMPLATE_TEST_CASE_2( "compare/2", "[compare]", Z, float, double ) {
    etl::dyn_matrix<Z> a(2, 2, 3.3);
    etl::dyn_matrix<Z> b(2, 2, 3.3);
    etl::dyn_matrix<Z> c(2, 2, 33.3);

    REQUIRE(a == a);
    REQUIRE(a == b);
    REQUIRE(b == a);
    REQUIRE(b == b);
    REQUIRE(!(a == c));
    REQUIRE(!(b == c));

    REQUIRE(!(a != a));
    REQUIRE(!(a != b));
    REQUIRE(!(b != a));
    REQUIRE(!(b != b));
    REQUIRE(a != c);
    REQUIRE(b != c);
}

TEMPLATE_TEST_CASE_2( "compare/3", "[compare]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> fa(3.3);
    etl::fast_matrix<Z, 2, 2> fc(33.3);
    etl::dyn_matrix<Z> da(2, 2, 3.3);
    etl::dyn_matrix<Z> dc(2, 2, 33.3);

    REQUIRE(da == fa);
    REQUIRE(fa == da);

    REQUIRE(da != fc);
    REQUIRE(fc != da);
}

TEMPLATE_TEST_CASE_2( "compare/4", "[compare]", Z, float, double ) {
    etl::fast_matrix<Z, 2, 2> a(3.3);
    etl::dyn_matrix<Z> b(2, 2, 3.3);

    REQUIRE((a + b) == (b + a));
    REQUIRE((2 * a) == (a * 2));
    REQUIRE(*(a * a) == *(a * b));

    REQUIRE(log(a + b) == log(b + a));
    REQUIRE(log(a + b) != exp(b + a));
}
