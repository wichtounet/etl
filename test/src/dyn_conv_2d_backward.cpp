//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include "conv_test.hpp"

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/1", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 3, 3> a = {1.0, 2.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0};
    etl::fast_matrix<T, 2, 2> b = {2.0, 0.0, 0.5, 0.5};
    etl::fast_matrix<T, 4, 4> c;

    c = etl::conv_2d_backward(a, b, 1, 1, 0, 0);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(2.0));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(4.0));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(6.0));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(0.0));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(0.5));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(3.5));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(4.5));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(1.5));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(6.0));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(4.5));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(3.0));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(0.5));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(2.5));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(1.5));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(0.5));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/2", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 5, 5> a;
    etl::fast_matrix<T, 3, 3> b;
    etl::fast_matrix<T, 7, 7> c;

    a = etl::magic(5);
    b = etl::magic(3);

    c = etl::conv_2d_backward(a, b, 1, 1, 0, 0);

    REQUIRE_EQUALS_APPROX(c(0, 0), T(136));
    REQUIRE_EQUALS_APPROX(c(0, 1), T(209));
    REQUIRE_EQUALS_APPROX(c(0, 2), T(134));
    REQUIRE_EQUALS_APPROX(c(0, 3), T(209));

    REQUIRE_EQUALS_APPROX(c(1, 0), T(235));
    REQUIRE_EQUALS_APPROX(c(1, 1), T(220));
    REQUIRE_EQUALS_APPROX(c(1, 2), T(441));
    REQUIRE_EQUALS_APPROX(c(1, 3), T(346));

    REQUIRE_EQUALS_APPROX(c(2, 0), T(169));
    REQUIRE_EQUALS_APPROX(c(2, 1), T(431));
    REQUIRE_EQUALS_APPROX(c(2, 2), T(595));
    REQUIRE_EQUALS_APPROX(c(2, 3), T(410));

    REQUIRE_EQUALS_APPROX(c(3, 0), T(184));
    REQUIRE_EQUALS_APPROX(c(3, 1), T(371));
    REQUIRE_EQUALS_APPROX(c(3, 2), T(440));
    REQUIRE_EQUALS_APPROX(c(3, 3), T(555));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/3", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 7, 7> a;
    etl::fast_matrix<T, 3, 3> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 9, 9> c;
    c = etl::conv_2d_backward(a, b, 1, 1, 0, 0);

    etl::fast_matrix<T, 9, 9> c_ref;
    c_ref = etl::conv_2d_full(a, b);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/4", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 7, 7> a;
    etl::fast_matrix<T, 3, 3> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 7, 7> c;
    c = etl::conv_2d_backward(a, b, 1, 1, 1, 1);

    etl::fast_matrix<T, 7, 7> c_ref;
    c_ref = etl::conv_2d_valid(a, b, 1, 1, 1, 1);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/5", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 7, 7> a;
    etl::fast_matrix<T, 3, 3> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 5, 5> c;
    c = etl::conv_2d_backward(a, b, 1, 1, 2, 2);

    etl::fast_matrix<T, 5, 5> c_ref;
    c_ref = etl::conv_2d_valid(a, b, 1, 1, 0, 0);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/6", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 32, 32> a;
    etl::fast_matrix<T, 7, 7> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 28, 28> c;
    c = etl::conv_2d_backward(a, b, 1, 1, 5, 5);

    etl::fast_matrix<T, 28, 28> c_ref;
    c_ref = etl::conv_2d_valid(a, b, 1, 1, 1, 1);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/7", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 4, 4> a;
    etl::fast_matrix<T, 5, 5> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 11, 11> c;
    c = etl::conv_2d_backward(a, b, 2, 2, 0, 0);

    etl::fast_matrix<T, 11, 11> c_ref;
    auto a_s = etl::impl::common::inner_pad(a, 2, 2);
    c_ref = etl::conv_2d_full(a_s, b);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}

TEMPLATE_TEST_CASE_2("conv/2d/dyn_backward/8", "[conv][conv2]", T, float, double) {
    etl::fast_matrix<T, 4, 4> a;
    etl::fast_matrix<T, 5, 5> b;

    a = 0.2 * etl::sequence_generator<T>(1.0);
    b = 0.1 * etl::sequence_generator<T>(2.0);

    etl::fast_matrix<T, 9, 9> c;
    c = etl::conv_2d_backward(a, b, 2, 2, 1, 1);

    etl::fast_matrix<T, 9, 9> c_ref;
    auto a_s = etl::impl::common::inner_pad(a, 2, 2);
    c_ref = etl::conv_2d_valid(a_s, b, 1, 1, 3, 3);

    REQUIRE_DIRECT(approx_equals(c_ref, c, base_eps_etl));
}
