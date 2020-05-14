//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("compare/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(3.3);
    etl::fast_matrix<Z, 2, 2> b(3.3);
    etl::fast_matrix<Z, 2, 2> c(33.3);

    REQUIRE_EQUALS(a, a);
    REQUIRE_EQUALS(a, b);
    REQUIRE_EQUALS(b, a);
    REQUIRE_EQUALS(b, b);
    REQUIRE_DIRECT(!(a == c));
    REQUIRE_DIRECT(!(b == c));

    REQUIRE_DIRECT(!(a != a));
    REQUIRE_DIRECT(!(a != b));
    REQUIRE_DIRECT(!(b != a));
    REQUIRE_DIRECT(!(b != b));
    REQUIRE_DIRECT(a != c);
    REQUIRE_DIRECT(b != c);
}

TEMPLATE_TEST_CASE_2("compare/2", "[compare]", Z, float, double) {
    etl::dyn_matrix<Z> a(2, 2, 3.3);
    etl::dyn_matrix<Z> b(2, 2, 3.3);
    etl::dyn_matrix<Z> c(2, 2, 33.3);

    REQUIRE_EQUALS(a, a);
    REQUIRE_EQUALS(a, b);
    REQUIRE_EQUALS(b, a);
    REQUIRE_EQUALS(b, b);
    REQUIRE_DIRECT(!(a == c));
    REQUIRE_DIRECT(!(b == c));

    REQUIRE_DIRECT(!(a != a));
    REQUIRE_DIRECT(!(a != b));
    REQUIRE_DIRECT(!(b != a));
    REQUIRE_DIRECT(!(b != b));
    REQUIRE_DIRECT(a != c);
    REQUIRE_DIRECT(b != c);
}

TEMPLATE_TEST_CASE_2("compare/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> fa(3.3);
    etl::fast_matrix<Z, 2, 2> fc(33.3);
    etl::dyn_matrix<Z> da(2, 2, 3.3);
    etl::dyn_matrix<Z> dc(2, 2, 33.3);

    REQUIRE_EQUALS(da, fa);
    REQUIRE_EQUALS(fa, da);

    REQUIRE_DIRECT(da != fc);
    REQUIRE_DIRECT(fc != da);
}

TEMPLATE_TEST_CASE_2("compare/4", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(3.3);
    etl::dyn_matrix<Z> b(2, 2, 3.3);

    REQUIRE_EQUALS((a + b), (b + a));
    REQUIRE_EQUALS((2 * a), (a * 2));
    REQUIRE_EQUALS(*(a * a), *(a * b));

    REQUIRE_EQUALS(log(a + b), log(b + a));
    REQUIRE_DIRECT(log(a + b) != exp(b + a));
}

TEMPLATE_TEST_CASE_2("compare/5", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 2, 2> a(3.3);
    etl::dyn_matrix<Z> b(2, 2, 3.3);

    etl::fast_matrix<Z, 3, 2> c(3.3);
    etl::dyn_matrix<Z> d(3, 2, 3.3);

    REQUIRE_DIRECT(a != c);
    REQUIRE_DIRECT(b != d);

    REQUIRE_DIRECT((a + b) != (c + d));
    REQUIRE_DIRECT((2 * a) != (c * 2));

    REQUIRE_DIRECT(log(a + b) != log(c + d));
}
