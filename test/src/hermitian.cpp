//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_4("herm/fast_matrix/1", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::hermitian_matrix<etl::fast_matrix<Z, 2,2>> a(Z(0.0, 0.0));
    etl::hermitian_matrix<etl::fast_matrix<Z, 1,1>> b(Z(0.0, 0.0));
    etl::hermitian_matrix<etl::fast_dyn_matrix<Z, 1,1>> c(Z(0.0, 0.0));
    etl::hermitian_matrix<etl::dyn_matrix<Z>> d(3, Z(0.0, 0.0));

    using a_t = decltype(a);
    using b_t = decltype(b);
    using c_t = decltype(c);
    using d_t = decltype(d);

    REQUIRE_EQUALS(a.dimensions(), 2UL);
    REQUIRE_EQUALS(b.dimensions(), 2UL);
    REQUIRE_EQUALS(c.dimensions(), 2UL);
    REQUIRE_EQUALS(d.dimensions(), 2UL);

    REQUIRE_DIRECT(etl::etl_traits<a_t>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<b_t>::is_fast);
    REQUIRE_DIRECT(etl::etl_traits<c_t>::is_fast);
    REQUIRE_DIRECT(!etl::etl_traits<d_t>::is_fast);

    REQUIRE_EQUALS(etl::etl_traits<a_t>::size(a), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<b_t>::size(b), 1UL);
    REQUIRE_EQUALS(etl::etl_traits<c_t>::size(c), 1UL);
    REQUIRE_EQUALS(etl::etl_traits<d_t>::size(d), 9UL);

    REQUIRE_DIRECT(Z(0.0, 0.0) == etl::get_conj(Z(0.0, 0.0)));

    REQUIRE_DIRECT(etl::is_hermitian(a));
    REQUIRE_DIRECT(etl::is_hermitian(b));
    REQUIRE_DIRECT(etl::is_hermitian(c));
    REQUIRE_DIRECT(etl::is_hermitian(d));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/2", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::hermitian_matrix<etl::fast_matrix<Z, 3, 3>> a;

    REQUIRE_EQUALS(a(0, 0), Z(0.0, 0.0));
    REQUIRE_EQUALS(a(1, 2), Z(0.0, 0.0));

    a(2, 1) = Z(3.5, 1.0);

    REQUIRE_EQUALS(a(2, 1), Z(3.5, 1.0));
    REQUIRE_EQUALS(a(1, 2), Z(3.5, -1.0));

    a(1, 1) += Z(5.5, 5.5);

    REQUIRE_EQUALS(a(1, 1), Z(5.5, 5.5));

    a(2, 0) = Z(1.5, 1.5);

    REQUIRE_EQUALS(a(2, 0), Z(1.5, 1.5));
    REQUIRE_EQUALS(a(0, 2), Z(1.5, -1.5));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/3", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::hermitian_matrix<etl::dyn_matrix<Z>> a(3UL);

    REQUIRE_EQUALS(a(0, 0), Z(0.0, 0.0));
    REQUIRE_EQUALS(a(1, 2), Z(0.0, 0.0));

    a(2, 1) = Z(3.5, 2.0);

    REQUIRE_EQUALS(a(2, 1), Z(3.5, 2.0));
    REQUIRE_EQUALS(a(1, 2), Z(3.5, -2.0));

    a(1, 1) += Z(5.5, 1.0);

    REQUIRE_EQUALS(a(1, 1), Z(5.5, 1.0));

    a(2, 0) = Z(1.5, 1.0);

    REQUIRE_EQUALS(a(2, 0), Z(1.5, 1.0));
    REQUIRE_EQUALS(a(0, 2), Z(1.5, -1.0));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/4", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 3, 3> a = {Z(0.0, 1.0),  Z(1.0, 1.0), Z(2.0, 2.0),
                                   Z(1.0, -1.0), Z(2.0, 2.0), Z(3.0, 3.0),
                                   Z(2.0, -2.0),  Z(3.0, -3.0), Z(3.0, 3.0)};;
    etl::hermitian_matrix<etl::fast_matrix<Z, 3, 3>> b;
    b = a;

    REQUIRE_EQUALS(b(0, 0), Z(0.0, 1.0));
    REQUIRE_EQUALS(b(0, 1), Z(1.0, 1.0));
    REQUIRE_EQUALS(b(0, 2), Z(2.0, 2.0));

    REQUIRE_EQUALS(b(1, 0), Z(1.0, -1.0));
    REQUIRE_EQUALS(b(1, 1), Z(2.0, 2.0));
    REQUIRE_EQUALS(b(1, 2), Z(3.0, 3.0));

    REQUIRE_EQUALS(b(2, 0), Z(2.0, -2.0));
    REQUIRE_EQUALS(b(2, 1), Z(3.0, -3.0));
    REQUIRE_EQUALS(b(2, 2), Z(3.0, 3.0));

    REQUIRE_DIRECT(b == a);
    REQUIRE_DIRECT(a == b);
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/5", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 3, 3> a = {Z(0.0, 1.0),  Z(1.0, 1.0), Z(2.0, 2.0),
                                   Z(1.0, -1.0), Z(2.0, 2.0), Z(3.0, 3.0),
                                   Z(2.0, -2.0),  Z(3.0, -3.0), Z(3.0, 3.0)};;

    etl::hermitian_matrix<etl::dyn_matrix<Z>> b(3UL);
    b = a;

    etl::hermitian_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    c = b + c;

    REQUIRE_EQUALS(c(0, 0), Z(0.0, 2.0));
    REQUIRE_EQUALS(c(0, 1), Z(2.0, 2.0));
    REQUIRE_EQUALS(c(0, 2), Z(4.0, 4.0));

    REQUIRE_EQUALS(c(1, 0), Z(2.0, -2.0));
    REQUIRE_EQUALS(c(1, 1), Z(4.0, 4.0));
    REQUIRE_EQUALS(c(1, 2), Z(6.0, 6.0));

    REQUIRE_EQUALS(c(2, 0), Z(4.0, -4.0));
    REQUIRE_EQUALS(c(2, 1), Z(6.0, -6.0));
    REQUIRE_EQUALS(c(2, 2), Z(6.0, 6.0));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/6", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 3, 3> a = {Z(0.0, 1.0),  Z(1.0, 1.0), Z(2.0, 2.0),
                                   Z(1.0, -1.0), Z(2.0, 2.0), Z(3.0, 3.0),
                                   Z(2.0, -2.0),  Z(3.0, -3.0), Z(3.0, 3.0)};;

    etl::hermitian_matrix<etl::dyn_matrix<Z>> b(3UL);
    b = a;

    etl::hermitian_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    c += b;

    REQUIRE_EQUALS(c(0, 0), Z(0.0, 2.0));
    REQUIRE_EQUALS(c(0, 1), Z(2.0, 2.0));
    REQUIRE_EQUALS(c(0, 2), Z(4.0, 4.0));

    REQUIRE_EQUALS(c(1, 0), Z(2.0, -2.0));
    REQUIRE_EQUALS(c(1, 1), Z(4.0, 4.0));
    REQUIRE_EQUALS(c(1, 2), Z(6.0, 6.0));

    REQUIRE_EQUALS(c(2, 0), Z(4.0, -4.0));
    REQUIRE_EQUALS(c(2, 1), Z(6.0, -6.0));
    REQUIRE_EQUALS(c(2, 2), Z(6.0, 6.0));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/7", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 3, 3> a = {Z(0.0, 1.0),  Z(1.0, 1.0), Z(2.0, 2.0),
                                   Z(2.0, -1.0), Z(2.0, 2.0), Z(3.0, 3.0),
                                   Z(2.0, -2.0),  Z(3.0, -3.0), Z(3.0, 3.0)};;
    etl::hermitian_matrix<etl::fast_matrix<Z, 3, 3>> b;

    REQUIRE_THROWS(b = a);

    REQUIRE_EQUALS(b(0, 0), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(0, 1), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(0, 2), Z(0.0, 0.0));

    REQUIRE_EQUALS(b(1, 0), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(1, 1), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(1, 2), Z(0.0, 0.0));

    REQUIRE_EQUALS(b(2, 0), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(2, 1), Z(0.0, 0.0));
    REQUIRE_EQUALS(b(2, 2), Z(0.0, 0.0));
}

TEMPLATE_TEST_CASE_4("herm/fast_matrix/8", "[herm][fast]", Z, std::complex<float>, std::complex<double>, etl::complex<float>, etl::complex<double>) {
    etl::fast_matrix<Z, 3, 3> a = {Z(0.0, 1.0),  Z(1.0, 1.0), Z(2.0, 2.0),
                                   Z(1.0, -1.0), Z(2.0, 2.0), Z(3.0, 3.0),
                                   Z(2.0, -2.0),  Z(3.0, -3.0), Z(3.0, 3.0)};;

    etl::hermitian_matrix<etl::dyn_matrix<Z>> c(3UL);
    c = a;

    a(0,1) = Z(0.0, 0.0);

    REQUIRE_THROWS(c += a);

    REQUIRE_EQUALS(c(0, 0), Z(0.0, 1.0));
    REQUIRE_EQUALS(c(0, 1), Z(1.0, 1.0));
    REQUIRE_EQUALS(c(0, 2), Z(2.0, 2.0));

    REQUIRE_EQUALS(c(1, 0), Z(1.0, -1.0));
    REQUIRE_EQUALS(c(1, 1), Z(2.0, 2.0));
    REQUIRE_EQUALS(c(1, 2), Z(3.0, 3.0));

    REQUIRE_EQUALS(c(2, 0), Z(2.0, -2.0));
    REQUIRE_EQUALS(c(2, 1), Z(3.0, -3.0));
    REQUIRE_EQUALS(c(2, 2), Z(3.0, 3.0));
}
