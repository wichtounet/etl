//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

#include <cmath>

TEMPLATE_TEST_CASE_2("elt_compare/equal/1", "[compare]", Z, float, double) {
    etl::fast_dyn_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_dyn_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.0, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_dyn_matrix<bool, 3, 2, 2> c;

    c = equal(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/equal/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = equal(a, Z(-1.0));

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/equal/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = equal(Z(-1.0), a);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/not_equal/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.0, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = not_equal(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/not_equal/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = not_equal(a, 1.0);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/not_equal/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, 0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = not_equal(1.0, a);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/less/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.01, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/less/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less(a, 2.0);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/less/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less(2.0, a);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/less_equal/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.01, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less_equal(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/less_equal/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less_equal(a, Z(2.01));

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/less_equal/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = less_equal(Z(2.01), a);

    REQUIRE_EQUALS(c(0, 0, 0), 2.01 <= c(0,0,0));
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.01, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = greater(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = greater(a, 2.0);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater/3", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.01, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = greater(2.0, b);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater_equal/1", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<Z, 3, 2, 2> b{0.0, -1.1, 2.01, 3.3,  1.0, 2.1, 3.1, 4.0,  -1.0, 2.25, 3.4, 0.001};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = greater_equal(a, b);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), false);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), false);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater_equal/2", "[compare]", Z, float, double) {
    etl::fast_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_matrix<bool, 3, 2, 2> c;

    c = greater_equal(a, 2.0);

    REQUIRE_EQUALS(c(0, 0, 0), false);
    REQUIRE_EQUALS(c(0, 0, 1), false);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), true);

    REQUIRE_EQUALS(c(1, 0, 0), false);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), true);
    REQUIRE_EQUALS(c(1, 1, 1), true);

    REQUIRE_EQUALS(c(2, 0, 0), false);
    REQUIRE_EQUALS(c(2, 0, 1), true);
    REQUIRE_EQUALS(c(2, 1, 0), true);
    REQUIRE_EQUALS(c(2, 1, 1), false);
}

TEMPLATE_TEST_CASE_2("elt_compare/greater_equal/3", "[compare]", Z, float, double) {
    etl::fast_dyn_matrix<Z, 3, 2, 2> a{1.0, -1.0, 2.0, 3.3,  1.0, 2.0, 3.0, 4.0,  -1.0, 2.25, 3.4, -0.002};
    etl::fast_dyn_matrix<bool, 3, 2, 2> c;

    c = greater_equal(2.0, a);

    REQUIRE_EQUALS(c(0, 0, 0), true);
    REQUIRE_EQUALS(c(0, 0, 1), true);
    REQUIRE_EQUALS(c(0, 1, 0), true);
    REQUIRE_EQUALS(c(0, 1, 1), false);

    REQUIRE_EQUALS(c(1, 0, 0), true);
    REQUIRE_EQUALS(c(1, 0, 1), true);
    REQUIRE_EQUALS(c(1, 1, 0), false);
    REQUIRE_EQUALS(c(1, 1, 1), false);

    REQUIRE_EQUALS(c(2, 0, 0), true);
    REQUIRE_EQUALS(c(2, 0, 1), false);
    REQUIRE_EQUALS(c(2, 1, 0), false);
    REQUIRE_EQUALS(c(2, 1, 1), true);
}
