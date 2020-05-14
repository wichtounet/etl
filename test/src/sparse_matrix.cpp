//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

TEMPLATE_TEST_CASE_2("sparse_matrix/traits/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a;

    REQUIRE_DIRECT(etl::is_etl_expr<decltype(a)>);
    REQUIRE_DIRECT(etl::is_sparse_matrix<decltype(a)>);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/traits/2", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 4);

    REQUIRE_DIRECT(etl::is_etl_expr<decltype(a)>);
    REQUIRE_DIRECT(etl::is_sparse_matrix<decltype(a)>);
    REQUIRE_EQUALS(etl::rows(a), 3UL);
    REQUIRE_EQUALS(etl::columns(a), 4UL);
    REQUIRE_EQUALS(etl::size(a), 12UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/1", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a;

    REQUIRE_EQUALS(a.rows(), 0UL);
    REQUIRE_EQUALS(a.columns(), 0UL);
    REQUIRE_EQUALS(a.size(), 0UL);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/2", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 2UL);
    REQUIRE_EQUALS(a.size(), 6UL);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    REQUIRE_EQUALS(a.get(0, 0), Z(1.0));
    REQUIRE_EQUALS(a.get(0, 1), Z(0.0));
    REQUIRE_EQUALS(a.get(1, 0), Z(0.0));
    REQUIRE_EQUALS(a.get(1, 1), Z(2.0));
    REQUIRE_EQUALS(a.get(2, 0), Z(3.0));
    REQUIRE_EQUALS(a.get(2, 1), Z(0.0));

    REQUIRE_EQUALS(a(0, 0), Z(1.0));
    REQUIRE_EQUALS(a(0, 1), Z(0.0));
    REQUIRE_EQUALS(a(1, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 1), Z(2.0));
    REQUIRE_EQUALS(a(2, 0), Z(3.0));
    REQUIRE_EQUALS(a(2, 1), Z(0.0));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/init/3", "[mat][init][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, etl::values(0.0, 1.2, 0.0, 2.0, 0.0, 0.01));

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 2UL);
    REQUIRE_EQUALS(a.size(), 6UL);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    REQUIRE_EQUALS(a.get(0, 0), Z(0.0));
    REQUIRE_EQUALS(a.get(0, 1), Z(1.2));
    REQUIRE_EQUALS(a.get(1, 0), Z(0.0));
    REQUIRE_EQUALS(a.get(1, 1), Z(2.0));
    REQUIRE_EQUALS(a.get(2, 0), Z(0.0));
    REQUIRE_EQUALS(a.get(2, 1), Z(0.01));

    REQUIRE_EQUALS(a(0, 0), Z(0.0));
    REQUIRE_EQUALS(a(0, 1), Z(1.2));
    REQUIRE_EQUALS(a(1, 0), Z(0.0));
    REQUIRE_EQUALS(a(1, 1), Z(2.0));
    REQUIRE_EQUALS(a(2, 0), Z(0.0));
    REQUIRE_EQUALS(a(2, 1), Z(0.01));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/set/1", "[mat][set][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 3UL);
    REQUIRE_EQUALS(a.size(), 9UL);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);

    a.set(1, 1, 42);

    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.non_zeros(), 1UL);

    a.set(2, 2, 2);
    a.set(0, 0, 1);

    REQUIRE_EQUALS(a.get(0, 0), 1);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 2);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    a.set(2, 2, -2.0);

    REQUIRE_EQUALS(a.get(0, 0), 1);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), -2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/set/2", "[mat][set][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 3UL);
    REQUIRE_EQUALS(a.size(), 9UL);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);

    a.set(0, 0, 1.0);
    a.set(1, 1, 42);
    a.set(2, 2, 2);

    REQUIRE_EQUALS(a.get(0, 0), 1.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    a.set(0, 0, 0.0);

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 2UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/reference/1", "[mat][reference][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 3UL);
    REQUIRE_EQUALS(a.size(), 9UL);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);

    a(1, 1) = 42;

    REQUIRE_EQUALS(a.get(1, 1), 42.0);
    REQUIRE_EQUALS(a.non_zeros(), 1UL);

    a(0, 0) = 1.0;
    a(2, 2) = 2.0;

    REQUIRE_EQUALS(a.get(0, 0), 1.0);
    REQUIRE_EQUALS(a.get(1, 1), 42.0);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    a(2, 2) = -2.0;

    REQUIRE_EQUALS(a.get(0, 0), 1.0);
    REQUIRE_EQUALS(a.get(1, 1), 42.0);
    REQUIRE_EQUALS(a.get(2, 2), -2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/reference/2", "[mat][reference][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);

    REQUIRE_EQUALS(a.rows(), 3UL);
    REQUIRE_EQUALS(a.columns(), 3UL);
    REQUIRE_EQUALS(a.size(), 9UL);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);

    a(0, 0) = 1.0;
    a(1, 1) = 42;
    a(2, 2) = 2;

    REQUIRE_EQUALS(a.get(0, 0), 1.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    a(0, 0) = 0.0;

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 2UL);

    a(2, 2) = 0.0;

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 42);
    REQUIRE_EQUALS(a.get(2, 2), 0.0);
    REQUIRE_EQUALS(a.non_zeros(), 1UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/erase/1", "[mat][erase][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    a.erase(0, 0);

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 2UL);

    a.erase(0, 0);

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(1, 1), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 2UL);

    a.erase(1, 1);
    a.erase(2, 0);

    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(0, 1), 0.0);
    REQUIRE_EQUALS(a.get(0, 0), 0.0);
    REQUIRE_EQUALS(a.get(2, 0), 0.0);
    REQUIRE_EQUALS(a.non_zeros(), 0UL);

    a.set(2, 0, 3);

    REQUIRE_EQUALS(a.get(2, 0), 3.0);
    REQUIRE_EQUALS(a.non_zeros(), 1UL);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/sequential/1", "[mat][erase][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));

    REQUIRE_EQUALS(a[0], 1.0);
    REQUIRE_EQUALS(a[1], 0.0);
    REQUIRE_EQUALS(a[2], 0.0);
    REQUIRE_EQUALS(a[3], 2.0);
    REQUIRE_EQUALS(a[4], 3.0);
    REQUIRE_EQUALS(a[5], 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/add/1", "[mat][add][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a + b;

    REQUIRE_EQUALS(c.get(0, 0), 3.0);
    REQUIRE_EQUALS(c.get(0, 1), 1.0);
    REQUIRE_EQUALS(c.get(1, 0), 0.0);
    REQUIRE_EQUALS(c.get(1, 1), 5.0);
    REQUIRE_EQUALS(c.get(2, 0), 3.0);
    REQUIRE_EQUALS(c.get(2, 1), 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/sub/1", "[mat][sub][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a - b;

    REQUIRE_EQUALS(c.get(0, 0), -1.0);
    REQUIRE_EQUALS(c.get(0, 1), -1.0);
    REQUIRE_EQUALS(c.get(1, 0), 0.0);
    REQUIRE_EQUALS(c.get(1, 1), -1.0);
    REQUIRE_EQUALS(c.get(2, 0), 3.0);
    REQUIRE_EQUALS(c.get(2, 1), 1.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/mul/1", "[mat][mul][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a >> b;

    REQUIRE_EQUALS(c.get(0, 0), 2.0);
    REQUIRE_EQUALS(c.get(0, 1), 0.0);
    REQUIRE_EQUALS(c.get(1, 0), 0.0);
    REQUIRE_EQUALS(c.get(1, 1), 6.0);
    REQUIRE_EQUALS(c.get(2, 0), 0.0);
    REQUIRE_EQUALS(c.get(2, 1), 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/div/1", "[mat][div][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.1, 2.0, 3.0, 1.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.1, 3.0, 1.0, 3.0}));
    etl::sparse_matrix<Z> c(3, 2);

    c = a / b;

    REQUIRE_EQUALS_APPROX(c.get(0, 0), Z(0.5));
    REQUIRE_EQUALS_APPROX(c.get(0, 1), Z(0.0));
    REQUIRE_EQUALS_APPROX(c.get(1, 0), Z(1.0));
    REQUIRE_EQUALS_APPROX(c.get(1, 1), Z(0.666666));
    REQUIRE_EQUALS_APPROX(c.get(2, 0), Z(3.0));
    REQUIRE_EQUALS_APPROX(c.get(2, 1), Z(0.333333));
}

TEMPLATE_TEST_CASE_2("sparse_matrix/inherit/0", "[mat][add][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 2, std::initializer_list<Z>({1.0, 0.0, 0.0, 2.0, 3.0, 0.0}));
    etl::sparse_matrix<Z> b(3, 2, std::initializer_list<Z>({2.0, 1.0, 0.0, 3.0, 0.0, 0.0}));
    etl::sparse_matrix<Z> c;

    c = a + b;

    REQUIRE_EQUALS(c.size(), a.size());
    REQUIRE_EQUALS(etl::dim<0>(c), etl::dim<0>(a));
    REQUIRE_EQUALS(etl::dim<1>(c), etl::dim<1>(a));

    REQUIRE_EQUALS(c.get(0, 0), 3.0);
    REQUIRE_EQUALS(c.get(0, 1), 1.0);
    REQUIRE_EQUALS(c.get(1, 0), 0.0);
    REQUIRE_EQUALS(c.get(1, 1), 5.0);
    REQUIRE_EQUALS(c.get(2, 0), 3.0);
    REQUIRE_EQUALS(c.get(2, 1), 0.0);
}

TEMPLATE_TEST_CASE_2("sparse_matrix/copy/1", "[mat][reference][sparse]", Z, double, float) {
    etl::sparse_matrix<Z> a(3, 3);
    etl::sparse_matrix<Z> b(3, 3);

    a(1, 1) = 42;
    a(0, 0) = 1.0;
    a(2, 2) = 2.0;

    REQUIRE_EQUALS(a.get(0, 0), 1.0);
    REQUIRE_EQUALS(a.get(1, 1), 42.0);
    REQUIRE_EQUALS(a.get(2, 2), 2.0);
    REQUIRE_EQUALS(a.non_zeros(), 3UL);

    b = a;

    REQUIRE_EQUALS(b.get(0, 0), 1.0);
    REQUIRE_EQUALS(b.get(1, 1), 42.0);
    REQUIRE_EQUALS(b.get(2, 2), 2.0);
    REQUIRE_EQUALS(b.non_zeros(), 3UL);
}
