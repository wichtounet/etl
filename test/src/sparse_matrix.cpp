//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

TEMPLATE_TEST_CASE_2("sparse_matrix/traits/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a;

    REQUIRE(etl::is_etl_expr<decltype(a)>::value);
    REQUIRE(etl::is_sparse_matrix<decltype(a)>::value);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/traits/2", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 4);

    REQUIRE(etl::is_etl_expr<decltype(a)>::value);
    REQUIRE(etl::is_sparse_matrix<decltype(a)>::value);
    REQUIRE(etl::rows(a) == 3);
    REQUIRE(etl::columns(a) == 4);
    REQUIRE(etl::size(a) == 12);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a;

    REQUIRE(a.rows() == 0);
    REQUIRE(a.columns() == 0);
    REQUIRE(a.size() == 0);
    REQUIRE(a.non_zeros() == 0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/2", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 6);
    REQUIRE(a.non_zeros() == 3);

    REQUIRE(a.get(0, 0) == Z(1.0));
    REQUIRE(a.get(0, 1) == Z(0.0));
    REQUIRE(a.get(1, 0) == Z(0.0));
    REQUIRE(a.get(1, 1) == Z(2.0));
    REQUIRE(a.get(2, 0) == Z(3.0));
    REQUIRE(a.get(2, 1) == Z(0.0));

    REQUIRE(a(0, 0) == Z(1.0));
    REQUIRE(a(0, 1) == Z(0.0));
    REQUIRE(a(1, 0) == Z(0.0));
    REQUIRE(a(1, 1) == Z(2.0));
    REQUIRE(a(2, 0) == Z(3.0));
    REQUIRE(a(2, 1) == Z(0.0));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/3", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, etl::values(0.0, 1.2, 0.0, 2.0, 0.0, 0.01));

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 2);
    REQUIRE(a.size() == 6);
    REQUIRE(a.non_zeros() == 3);

    REQUIRE(a.get(0, 0) == Z(0.0));
    REQUIRE(a.get(0, 1) == Z(1.2));
    REQUIRE(a.get(1, 0) == Z(0.0));
    REQUIRE(a.get(1, 1) == Z(2.0));
    REQUIRE(a.get(2, 0) == Z(0.0));
    REQUIRE(a.get(2, 1) == Z(0.01));

    REQUIRE(a(0, 0) == Z(0.0));
    REQUIRE(a(0, 1) == Z(1.2));
    REQUIRE(a(1, 0) == Z(0.0));
    REQUIRE(a(1, 1) == Z(2.0));
    REQUIRE(a(2, 0) == Z(0.0));
    REQUIRE(a(2, 1) == Z(0.01));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/set/1", "[mat][set][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 3);
    REQUIRE(a.size() == 9);
    REQUIRE(a.non_zeros() == 0);

    a.set(1, 1, 42);

    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.non_zeros() == 1);

    a.set(2, 2, 2);
    a.set(0, 0, 1);

    REQUIRE(a.get(0, 0) == 1);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 2);
    REQUIRE(a.non_zeros() == 3);

    a.set(2, 2, -2.0);

    REQUIRE(a.get(0, 0) == 1);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == -2.0);
    REQUIRE(a.non_zeros() == 3);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/set/2", "[mat][set][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 3);
    REQUIRE(a.size() == 9);
    REQUIRE(a.non_zeros() == 0);

    a.set(0, 0, 1.0);
    a.set(1, 1, 42);
    a.set(2, 2, 2);

    REQUIRE(a.get(0, 0) == 1.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 2.0);
    REQUIRE(a.non_zeros() == 3);

    a.set(0, 0, 0.0);

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 2.0);
    REQUIRE(a.non_zeros() == 2);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/reference/1", "[mat][reference][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 3);
    REQUIRE(a.size() == 9);
    REQUIRE(a.non_zeros() == 0);

    a(1, 1) = 42;

    REQUIRE(a.get(1, 1) == 42.0);
    REQUIRE(a.non_zeros() == 1);

    a(0, 0) = 1.0;
    a(2, 2) = 2.0;

    REQUIRE(a.get(0, 0) == 1.0);
    REQUIRE(a.get(1, 1) == 42.0);
    REQUIRE(a.get(2, 2) == 2.0);
    REQUIRE(a.non_zeros() == 3);

    a(2, 2) = -2.0;

    REQUIRE(a.get(0, 0) == 1.0);
    REQUIRE(a.get(1, 1) == 42.0);
    REQUIRE(a.get(2, 2) == -2.0);
    REQUIRE(a.non_zeros() == 3);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/reference/2", "[mat][reference][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE(a.rows() == 3);
    REQUIRE(a.columns() == 3);
    REQUIRE(a.size() == 9);
    REQUIRE(a.non_zeros() == 0);

    a(0, 0) = 1.0;
    a(1, 1) = 42;
    a(2, 2) = 2;

    REQUIRE(a.get(0, 0) == 1.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 2.0);
    REQUIRE(a.non_zeros() == 3);

    a(0, 0) = 0.0;

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 2.0);
    REQUIRE(a.non_zeros() == 2);

    a(2, 2) = 0.0;

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 42);
    REQUIRE(a.get(2, 2) == 0.0);
    REQUIRE(a.non_zeros() == 1);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/erase/1", "[mat][erase][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE(a.non_zeros() == 3);

    a.erase(0, 0);

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 2.0);
    REQUIRE(a.non_zeros() == 2);

    a.erase(0, 0);

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(1, 1) == 2.0);
    REQUIRE(a.non_zeros() == 2);

    a.erase(1, 1);
    a.erase(2, 0);

    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(0, 1) == 0.0);
    REQUIRE(a.get(0, 0) == 0.0);
    REQUIRE(a.get(2, 0) == 0.0);
    REQUIRE(a.non_zeros() == 0);

    a.set(2, 0, 3);

    REQUIRE(a.get(2, 0) == 3.0);
    REQUIRE(a.non_zeros() == 1);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/sequential/1", "[mat][erase][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE(a[0] == 1.0);
    REQUIRE(a[1] == 0.0);
    REQUIRE(a[2] == 0.0);
    REQUIRE(a[3] == 2.0);
    REQUIRE(a[4] == 3.0);
    REQUIRE(a[5] == 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/add/1", "[mat][add][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a + b;

    REQUIRE(c.get(0, 0) == 3.0);
    REQUIRE(c.get(0, 1) == 1.0);
    REQUIRE(c.get(1, 0) == 0.0);
    REQUIRE(c.get(1, 1) == 5.0);
    REQUIRE(c.get(2, 0) == 3.0);
    REQUIRE(c.get(2, 1) == 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/sub/1", "[mat][sub][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a - b;

    REQUIRE(c.get(0, 0) == -1.0);
    REQUIRE(c.get(0, 1) == -1.0);
    REQUIRE(c.get(1, 0) == 0.0);
    REQUIRE(c.get(1, 1) == -1.0);
    REQUIRE(c.get(2, 0) == 3.0);
    REQUIRE(c.get(2, 1) == 1.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/mul/1", "[mat][mul][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a >> b;

    REQUIRE(c.get(0, 0) == 2.0);
    REQUIRE(c.get(0, 1) == 0.0);
    REQUIRE(c.get(1, 0) == 0.0);
    REQUIRE(c.get(1, 1) == 6.0);
    REQUIRE(c.get(2, 0) == 0.0);
    REQUIRE(c.get(2, 1) == 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/div/1", "[mat][div][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.1, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.1, 3.0, 1.0, 3.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a / b;

    REQUIRE(c.get(0, 0) == Approx(Z(0.5)));
    REQUIRE(c.get(0, 1) == Approx(Z(0.0)));
    REQUIRE(c.get(1, 0) == Approx(Z(1.0)));
    REQUIRE(c.get(1, 1) == Approx(Z(0.666666)));
    REQUIRE(c.get(2, 0) == Approx(Z(3.0)));
    REQUIRE(c.get(2, 1) == Approx(Z(0.333333)));
}
