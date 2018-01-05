//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

ETL_TEST_CASE("globals/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a;

    REQUIRE_DIRECT(a.is_square());
    REQUIRE_DIRECT(is_square(a));

    REQUIRE_DIRECT(!a.is_rectangular());
    REQUIRE_DIRECT(!is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_square());
    REQUIRE_DIRECT(is_square(a + a));

    REQUIRE_DIRECT(!expr.is_rectangular());
    REQUIRE_DIRECT(!is_rectangular(a + a));
}

ETL_TEST_CASE("globals/2", "[globals]") {
    etl::fast_matrix<double, 3, 2> a;

    REQUIRE_DIRECT(!a.is_square());
    REQUIRE_DIRECT(!is_square(a));

    REQUIRE_DIRECT(a.is_rectangular());
    REQUIRE_DIRECT(is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_square());
    REQUIRE_DIRECT(!is_square(a + a));

    REQUIRE_DIRECT(expr.is_rectangular());
    REQUIRE_DIRECT(is_rectangular(a + a));
}

ETL_TEST_CASE("globals/3", "[globals]") {
    etl::fast_matrix<double, 3, 2, 2> a;

    REQUIRE_DIRECT(a.is_sub_square());
    REQUIRE_DIRECT(is_sub_square(a));

    REQUIRE_DIRECT(is_square(a(1)));

    REQUIRE_DIRECT(!a.is_sub_rectangular());
    REQUIRE_DIRECT(!is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_sub_square());
    REQUIRE_DIRECT(is_sub_square(a + a));

    REQUIRE_DIRECT(!expr.is_sub_rectangular());
    REQUIRE_DIRECT(!is_sub_rectangular(a + a));
}

ETL_TEST_CASE("globals/4", "[globals]") {
    etl::fast_matrix<double, 3, 2, 3> a;

    REQUIRE_DIRECT(!a.is_sub_square());
    REQUIRE_DIRECT(!is_sub_square(a));

    REQUIRE_DIRECT(is_rectangular(a(1)));

    REQUIRE_DIRECT(a.is_sub_rectangular());
    REQUIRE_DIRECT(is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_sub_square());
    REQUIRE_DIRECT(!is_sub_square(a + a));

    REQUIRE_DIRECT(expr.is_sub_rectangular());
    REQUIRE_DIRECT(is_sub_rectangular(a + a));
}

ETL_TEST_CASE("globals/is_symmetric/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 2.0, 1.0};

    REQUIRE_DIRECT(a.is_symmetric());
    REQUIRE_DIRECT(is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_symmetric());
    REQUIRE_DIRECT(is_symmetric(expr));
}

ETL_TEST_CASE("globals/is_symmetric/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(a.is_symmetric());
    REQUIRE_DIRECT(is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_symmetric());
    REQUIRE_DIRECT(is_symmetric(expr));
}

ETL_TEST_CASE("globals/is_symmetric/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};
    etl::fast_matrix<double, 9, 1> b{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_symmetric());
    REQUIRE_DIRECT(!is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_symmetric());
    REQUIRE_DIRECT(!is_symmetric(expr));

    REQUIRE_DIRECT(!b.is_symmetric());
    REQUIRE_DIRECT(!is_symmetric(b));
}

ETL_TEST_CASE("globals/is_lower_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 2.0, 1.0};

    REQUIRE_DIRECT(a.is_lower_triangular());
    REQUIRE_DIRECT(is_lower_triangular(a));
    REQUIRE_DIRECT(a.is_triangular());
    REQUIRE_DIRECT(is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_lower_triangular());
    REQUIRE_DIRECT(is_lower_triangular(expr));
    REQUIRE_DIRECT(expr.is_triangular());
    REQUIRE_DIRECT(is_triangular(expr));
}

ETL_TEST_CASE("globals/is_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(a.is_lower_triangular());
    REQUIRE_DIRECT(is_lower_triangular(a));
    REQUIRE_DIRECT(a.is_triangular());
    REQUIRE_DIRECT(is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_lower_triangular());
    REQUIRE_DIRECT(is_lower_triangular(expr));
    REQUIRE_DIRECT(expr.is_triangular());
    REQUIRE_DIRECT(is_triangular(expr));
}

ETL_TEST_CASE("globals/is_lower_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};
    etl::fast_matrix<double, 1, 9> b{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_lower_triangular());
    REQUIRE_DIRECT(!is_lower_triangular(a));
    REQUIRE_DIRECT(!a.is_triangular());
    REQUIRE_DIRECT(!is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_lower_triangular());
    REQUIRE_DIRECT(!is_lower_triangular(expr));
    REQUIRE_DIRECT(!expr.is_triangular());
    REQUIRE_DIRECT(!is_triangular(expr));

    REQUIRE_DIRECT(!b.is_lower_triangular());
    REQUIRE_DIRECT(!is_lower_triangular(b));
    REQUIRE_DIRECT(!b.is_triangular());
    REQUIRE_DIRECT(!is_triangular(b));
}

ETL_TEST_CASE("globals/is_strictly_lower_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{0.0, 0.0, 2.0, 0.0};

    REQUIRE_DIRECT(a.is_strictly_lower_triangular());
    REQUIRE_DIRECT(is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_strictly_lower_triangular());
    REQUIRE_DIRECT(is_strictly_lower_triangular(expr));
}

ETL_TEST_CASE("globals/is_strictly_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0};

    REQUIRE_DIRECT(a.is_strictly_lower_triangular());
    REQUIRE_DIRECT(is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_strictly_lower_triangular());
    REQUIRE_DIRECT(is_strictly_lower_triangular(expr));
}

ETL_TEST_CASE("globals/is_strictly_lower_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};
    etl::fast_matrix<double, 9, 1> b{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_strictly_lower_triangular());
    REQUIRE_DIRECT(!is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_strictly_lower_triangular());
    REQUIRE_DIRECT(!is_strictly_lower_triangular(expr));

    REQUIRE_DIRECT(!b.is_strictly_lower_triangular());
    REQUIRE_DIRECT(!is_strictly_lower_triangular(b));
}

ETL_TEST_CASE("globals/is_upper_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 1.0, 0.0, 1.0};

    REQUIRE_DIRECT(a.is_upper_triangular());
    REQUIRE_DIRECT(is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_upper_triangular());
    REQUIRE_DIRECT(is_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_upper_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};

    REQUIRE_DIRECT(a.is_upper_triangular());
    REQUIRE_DIRECT(is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_upper_triangular());
    REQUIRE_DIRECT(is_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_upper_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};
    etl::fast_matrix<double, 9, 1> b{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_upper_triangular());
    REQUIRE_DIRECT(!is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_upper_triangular());
    REQUIRE_DIRECT(!is_upper_triangular(expr));

    REQUIRE_DIRECT(!b.is_upper_triangular());
    REQUIRE_DIRECT(!is_upper_triangular(b));
}

ETL_TEST_CASE("globals/is_strictly_upper_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{0.0, 1.0, 0.0, 0.0};

    REQUIRE_DIRECT(a.is_strictly_upper_triangular());
    REQUIRE_DIRECT(is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_strictly_upper_triangular());
    REQUIRE_DIRECT(is_strictly_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_strictly_upper_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

    REQUIRE_DIRECT(a.is_strictly_upper_triangular());
    REQUIRE_DIRECT(is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_strictly_upper_triangular());
    REQUIRE_DIRECT(is_strictly_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_strictly_upper_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};
    etl::fast_matrix<double, 9, 1> b{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_strictly_upper_triangular());
    REQUIRE_DIRECT(!is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_strictly_upper_triangular());
    REQUIRE_DIRECT(!is_strictly_upper_triangular(expr));

    REQUIRE_DIRECT(!b.is_strictly_upper_triangular());
    REQUIRE_DIRECT(!is_strictly_upper_triangular(b));
}

ETL_TEST_CASE("globals/is_uni_lower_triangular/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 11.0, 12.0, 1.0};
    etl::fast_matrix<double, 9, 1> b{1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 11.0, 12.0, 1.0};

    REQUIRE_DIRECT(a.is_uni_lower_triangular());
    REQUIRE_DIRECT(is_uni_lower_triangular(a));

    REQUIRE_DIRECT(!b.is_strictly_lower_triangular());
    REQUIRE_DIRECT(!is_strictly_lower_triangular(b));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_uni_lower_triangular());
    REQUIRE_DIRECT(!is_uni_lower_triangular(expr));
}

ETL_TEST_CASE("globals/is_uni_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.1, 0.0, 0.0, 2.0, 1.0, 0.0, 11.0, 12.0, 1.0};
    etl::fast_matrix<double, 9, 1> b{1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 11.0, 12.0, 1.0};

    REQUIRE_DIRECT(!a.is_uni_lower_triangular());
    REQUIRE_DIRECT(!is_uni_lower_triangular(a));

    REQUIRE_DIRECT(!b.is_strictly_lower_triangular());
    REQUIRE_DIRECT(!is_strictly_lower_triangular(b));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_uni_lower_triangular());
    REQUIRE_DIRECT(!is_uni_lower_triangular(expr));
}

ETL_TEST_CASE("globals/is_uni_upper_triangular/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 9, 1> b{1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0};

    REQUIRE_DIRECT(a.is_uni_upper_triangular());
    REQUIRE_DIRECT(is_uni_upper_triangular(a));

    REQUIRE_DIRECT(!b.is_strictly_upper_triangular());
    REQUIRE_DIRECT(!is_strictly_upper_triangular(b));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_uni_upper_triangular());
    REQUIRE_DIRECT(!is_uni_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_uni_upper_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 1.1, 4.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 9, 1> b{1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 0.0, 0.0, 1.0};

    REQUIRE_DIRECT(!a.is_uni_upper_triangular());
    REQUIRE_DIRECT(!is_uni_upper_triangular(a));

    REQUIRE_DIRECT(!b.is_strictly_upper_triangular());
    REQUIRE_DIRECT(!is_strictly_upper_triangular(b));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_uni_upper_triangular());
    REQUIRE_DIRECT(!is_uni_upper_triangular(expr));
}

ETL_TEST_CASE("globals/is_uniform/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a(1.0);

    REQUIRE_DIRECT(a.is_uniform());
    REQUIRE_DIRECT(is_uniform(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(expr.is_uniform());
    REQUIRE_DIRECT(is_uniform(expr));
}

ETL_TEST_CASE("globals/is_uniform/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE_DIRECT(!a.is_uniform());
    REQUIRE_DIRECT(!is_uniform(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_uniform());
    REQUIRE_DIRECT(!is_uniform(expr));
}

ETL_TEST_CASE("globals/is_diagonal/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0};
    etl::fast_matrix<double, 2, 4> b;

    REQUIRE_DIRECT(a.is_diagonal());
    REQUIRE_DIRECT(is_diagonal(a));

    decltype(auto) c = a + a;

    REQUIRE_DIRECT(c.is_diagonal());
    REQUIRE_DIRECT(is_diagonal(c));

    REQUIRE_DIRECT(!b.is_diagonal());
    REQUIRE_DIRECT(!is_diagonal(b));

    decltype(auto) d = b + b;

    REQUIRE_DIRECT(!d.is_diagonal());
    REQUIRE_DIRECT(!is_diagonal(d));
}

ETL_TEST_CASE("globals/is_diagonal/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, -1.0};

    REQUIRE_DIRECT(!a.is_diagonal());
    REQUIRE_DIRECT(!is_diagonal(a));

    decltype(auto) expr = a + a;

    REQUIRE_DIRECT(!expr.is_diagonal());
    REQUIRE_DIRECT(!is_diagonal(expr));
}

ETL_TEST_CASE("globals/trace/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 3.0, 4.0};

    REQUIRE_EQUALS_APPROX(trace(a), 5.0);

    decltype(auto) expr = a + a;

    REQUIRE_EQUALS_APPROX(trace(expr), 10.0);
}

ETL_TEST_CASE("globals/trace/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    REQUIRE_EQUALS_APPROX(trace(a), 15.0);

    decltype(auto) expr = a + a;

    REQUIRE_EQUALS_APPROX(trace(expr), 30.0);
}

ETL_TEST_CASE("globals/is_real_complex/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a;
    etl::fast_matrix<float, 3, 3> b;
    etl::fast_matrix<int, 3, 3> c;
    etl::fast_matrix<long, 3, 3> d;
    etl::fast_matrix<std::complex<float>, 3, 3> e;
    etl::fast_matrix<std::complex<double>, 3, 3> f;
    etl::fast_matrix<etl::complex<float>, 3, 3> g;
    etl::fast_matrix<etl::complex<double>, 3, 3> h;

    REQUIRE_DIRECT(etl::is_real_matrix(a));
    REQUIRE_DIRECT(etl::is_real_matrix(b));
    REQUIRE_DIRECT(etl::is_real_matrix(c));
    REQUIRE_DIRECT(etl::is_real_matrix(d));
    REQUIRE_DIRECT(!etl::is_real_matrix(e));
    REQUIRE_DIRECT(!etl::is_real_matrix(f));
    REQUIRE_DIRECT(!etl::is_real_matrix(g));
    REQUIRE_DIRECT(!etl::is_real_matrix(h));

    REQUIRE_DIRECT(!etl::is_complex_matrix(a));
    REQUIRE_DIRECT(!etl::is_complex_matrix(b));
    REQUIRE_DIRECT(!etl::is_complex_matrix(c));
    REQUIRE_DIRECT(!etl::is_complex_matrix(d));
    REQUIRE_DIRECT(etl::is_complex_matrix(e));
    REQUIRE_DIRECT(etl::is_complex_matrix(f));
    REQUIRE_DIRECT(etl::is_complex_matrix(g));
    REQUIRE_DIRECT(etl::is_complex_matrix(h));
}

ETL_TEST_CASE("globals/is_hermitian/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a;
    etl::fast_matrix<float, 3, 3> b;
    etl::fast_matrix<int, 3, 3> c;
    etl::fast_matrix<long, 3, 3> d;
    etl::fast_matrix<long, 4, 3> e;

    REQUIRE_DIRECT(!is_hermitian(a));
    REQUIRE_DIRECT(!is_hermitian(b));
    REQUIRE_DIRECT(!is_hermitian(c));
    REQUIRE_DIRECT(!is_hermitian(d));
    REQUIRE_DIRECT(!is_hermitian(e));
}

ETL_TEST_CASE("globals/is_hermitian/2", "[globals]") {
    etl::fast_matrix<std::complex<double>, 3, 3> a;
    etl::fast_matrix<std::complex<float>, 3, 3> b;
    etl::fast_matrix<etl::complex<double>, 3, 3> c;
    etl::fast_matrix<etl::complex<float>, 3, 3> d;
    etl::fast_matrix<etl::complex<float>, 4, 3> e;

    a(0, 1) = std::complex<double>(1.0, 2.0);
    a(0, 2) = std::complex<double>(2.0, -2.0);
    a(1, 0) = std::complex<double>(3.0, 4.0);
    a(1, 2) = std::complex<double>(0.0, 1.0);
    a(2, 0) = std::complex<double>(3.0, 1.0);
    a(2, 1) = std::complex<double>(4.0, 5.0);

    b(0, 1) = std::complex<float>(2.0, 2.0);
    b(0, 2) = std::complex<float>(2.0, -3.0);
    b(1, 0) = std::complex<float>(3.0, 3.0);
    b(1, 2) = std::complex<float>(0.0, 3.0);
    b(2, 0) = std::complex<float>(-3.0, -1.0);
    b(2, 1) = std::complex<float>(4.0, 5.0);

    c(0, 1) = etl::complex<double>(2.0, 2.0);
    c(0, 2) = etl::complex<double>(2.0, -3.0);
    c(1, 0) = etl::complex<double>(3.0, 3.0);
    c(1, 2) = etl::complex<double>(0.0, 3.0);
    c(2, 0) = etl::complex<double>(-3.0, -1.0);
    c(2, 1) = etl::complex<double>(4.0, 5.0);

    d(0, 1) = etl::complex<float>(2.0, 2.0);
    d(0, 2) = etl::complex<float>(2.0, -3.0);
    d(1, 0) = etl::complex<float>(2.0, -2.0);
    d(1, 2) = etl::complex<float>(0.0, 3.0);
    d(2, 0) = etl::complex<float>(2.0, 3.0);
    d(2, 1) = etl::complex<float>(0.0, 3.0);

    REQUIRE_DIRECT(!is_hermitian(a));
    REQUIRE_DIRECT(!is_hermitian(b));
    REQUIRE_DIRECT(!is_hermitian(c));
    REQUIRE_DIRECT(!is_hermitian(d));
    REQUIRE_DIRECT(!is_hermitian(e));
}

ETL_TEST_CASE("globals/is_hermitian/3", "[globals]") {
    etl::fast_matrix<std::complex<double>, 3, 3> a;
    etl::fast_matrix<std::complex<float>, 3, 3> b;
    etl::fast_matrix<etl::complex<double>, 3, 3> c;
    etl::fast_matrix<etl::complex<float>, 3, 3> d;

    a(0, 1) = std::complex<double>(1.0, 2.0);
    a(0, 2) = std::complex<double>(2.0, -2.0);
    a(1, 0) = std::complex<double>(1.0, -2.0);
    a(1, 2) = std::complex<double>(0.0, 1.0);
    a(2, 0) = std::complex<double>(2.0, 2.0);
    a(2, 1) = std::complex<double>(0.0, -1.0);

    b(0, 1) = std::complex<float>(2.0, 2.0);
    b(0, 2) = std::complex<float>(2.0, -3.0);
    b(1, 0) = std::complex<float>(2.0, -2.0);
    b(1, 2) = std::complex<float>(0.0, 3.0);
    b(2, 0) = std::complex<float>(2.0, 3.0);
    b(2, 1) = std::complex<float>(0.0, -3.0);

    c(0, 1) = etl::complex<double>(2.0, 2.0);
    c(0, 2) = etl::complex<double>(2.0, -3.0);
    c(1, 0) = etl::complex<double>(2.0, -2.0);
    c(1, 2) = etl::complex<double>(0.0, 1.5);
    c(2, 0) = etl::complex<double>(2.0, 3.0);
    c(2, 1) = etl::complex<double>(0.0, -1.5);

    d(0, 1) = etl::complex<float>(2.0, 2.0);
    d(0, 2) = etl::complex<float>(1.0, 0.0);
    d(1, 0) = etl::complex<float>(2.0, -2.0);
    d(1, 2) = etl::complex<float>(0.0, 3.0);
    d(2, 0) = etl::complex<float>(1.0, 0.0);
    d(2, 1) = etl::complex<float>(0.0, -3.0);

    REQUIRE_DIRECT(is_hermitian(a));
    REQUIRE_DIRECT(is_hermitian(b));
    REQUIRE_DIRECT(is_hermitian(c));
    REQUIRE_DIRECT(is_hermitian(d));
}

ETL_TEST_CASE("globals/is_permutation/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 1.0, 0.0};
    etl::fast_matrix<double, 2, 2> b{1.0, 1.0, 1.0, 0.0};
    etl::fast_matrix<double, 2, 2> c{1.0, 0.0, 0.0, 0.0};
    etl::fast_matrix<double, 2, 2> d{1.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 2, 2> e{0.0, 1.0, 1.0, 0.0};
    etl::fast_matrix<double, 3, 2> f;

    REQUIRE_DIRECT(!is_permutation_matrix(a));
    REQUIRE_DIRECT(!is_permutation_matrix(b));
    REQUIRE_DIRECT(!is_permutation_matrix(c));
    REQUIRE_DIRECT(is_permutation_matrix(d));
    REQUIRE_DIRECT(is_permutation_matrix(e));
    REQUIRE_DIRECT(!is_permutation_matrix(f));
}

ETL_TEST_CASE("globals/is_permutation/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> b{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> c{1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> d{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};

    REQUIRE_DIRECT(!is_permutation_matrix(a));
    REQUIRE_DIRECT(!is_permutation_matrix(b));
    REQUIRE_DIRECT(!is_permutation_matrix(c));
}

ETL_TEST_CASE("globals/determinant/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 2, 2> b{0.0, 1.0, 1.0, 0.0};

    REQUIRE_EQUALS(determinant(a), 1.0);
    REQUIRE_EQUALS(determinant(b), -1.0);
}

ETL_TEST_CASE("globals/determinant/2", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 3.0, 4.0};
    etl::fast_matrix<double, 2, 2> b{-2.0, 3.0, 5.0, -8.0};

    REQUIRE_EQUALS_APPROX(determinant(a), -2.0);
    REQUIRE_EQUALS_APPROX(determinant(b), 1.0);
}

ETL_TEST_CASE("globals/determinant/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{-2, 3, 4, 5, -8, 1, 0, 2, 3};
    etl::fast_matrix<double, 3, 3> b{-2, 0.5, 4, 5, -1.5, 1, 0, 2, 3};

    REQUIRE_EQUALS_APPROX(determinant(a), 47.0);
    REQUIRE_EQUALS_APPROX(determinant(b), 45.5);
}

ETL_TEST_CASE("globals/shuffle/1", "[globals]") {
    etl::fast_matrix<double, 5> a{0, 1, 2, 3, 4};

    shuffle(a);

    REQUIRE_DIRECT(a[0] >= 0 && a[0] <= 5);
    REQUIRE_DIRECT(a[1] >= 0 && a[1] <= 5);
    REQUIRE_DIRECT(a[2] >= 0 && a[2] <= 5);
    REQUIRE_DIRECT(a[3] >= 0 && a[3] <= 5);
    REQUIRE_DIRECT(a[4] >= 0 && a[4] <= 5);

    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 0), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 1), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 2), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 3), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 4), 1);
}

ETL_TEST_CASE("globals/shuffle/2", "[globals]") {
    etl::fast_matrix<double, 5, 2> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    shuffle(a);

    REQUIRE_DIRECT(a[0] >= 0 && a[0] <= 9);
    REQUIRE_DIRECT(a[1] >= 0 && a[1] <= 9);
    REQUIRE_DIRECT(a[2] >= 0 && a[2] <= 9);
    REQUIRE_DIRECT(a[3] >= 0 && a[3] <= 9);
    REQUIRE_DIRECT(a[4] >= 0 && a[4] <= 9);
    REQUIRE_DIRECT(a[5] >= 0 && a[5] <= 9);
    REQUIRE_DIRECT(a[6] >= 0 && a[6] <= 9);
    REQUIRE_DIRECT(a[7] >= 0 && a[7] <= 9);
    REQUIRE_DIRECT(a[8] >= 0 && a[8] <= 9);
    REQUIRE_DIRECT(a[9] >= 0 && a[9] <= 9);

    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 0), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 1), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 2), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 3), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 4), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 5), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 6), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 7), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 8), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 9), 1);
}

ETL_TEST_CASE("globals/shuffle/3", "[globals]") {
    etl::fast_matrix<double, 1, 10> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<double, 1, 10> b{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    shuffle(b);

    REQUIRE_DIRECT(a == b);
}

ETL_TEST_CASE("globals/parallel_shuffle/1", "[globals]") {
    etl::fast_matrix<double, 5> a{0, 1, 2, 3, 4};
    etl::fast_matrix<double, 5> b{1, 2, 3, 4, 5};

    parallel_shuffle(a, b);

    REQUIRE_DIRECT(a[0] >= 0 && a[0] <= 5);
    REQUIRE_DIRECT(a[1] >= 0 && a[1] <= 5);
    REQUIRE_DIRECT(a[2] >= 0 && a[2] <= 5);
    REQUIRE_DIRECT(a[3] >= 0 && a[3] <= 5);
    REQUIRE_DIRECT(a[4] >= 0 && a[4] <= 5);

    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 0), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 1), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 2), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 3), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 4), 1);

    REQUIRE_EQUALS(a[0], b[0] - 1);
    REQUIRE_EQUALS(a[1], b[1] - 1);
    REQUIRE_EQUALS(a[2], b[2] - 1);
    REQUIRE_EQUALS(a[3], b[3] - 1);
    REQUIRE_EQUALS(a[4], b[4] - 1);
}

ETL_TEST_CASE("globals/parallel_shuffle/2", "[globals]") {
    etl::fast_matrix<double, 5, 2> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<double, 5, 2> b;
    b = a + 1;

    parallel_shuffle(a, b);

    REQUIRE_DIRECT(a[0] >= 0 && a[0] <= 9);
    REQUIRE_DIRECT(a[1] >= 0 && a[1] <= 9);
    REQUIRE_DIRECT(a[2] >= 0 && a[2] <= 9);
    REQUIRE_DIRECT(a[3] >= 0 && a[3] <= 9);
    REQUIRE_DIRECT(a[4] >= 0 && a[4] <= 9);
    REQUIRE_DIRECT(a[5] >= 0 && a[5] <= 9);
    REQUIRE_DIRECT(a[6] >= 0 && a[6] <= 9);
    REQUIRE_DIRECT(a[7] >= 0 && a[7] <= 9);
    REQUIRE_DIRECT(a[8] >= 0 && a[8] <= 9);
    REQUIRE_DIRECT(a[9] >= 0 && a[9] <= 9);

    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 0), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 1), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 2), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 3), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 4), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 5), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 6), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 7), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 8), 1);
    REQUIRE_EQUALS(std::count(a.begin(), a.end(), 9), 1);

    REQUIRE_EQUALS(a[0], b[0] - 1);
    REQUIRE_EQUALS(a[1], b[1] - 1);
    REQUIRE_EQUALS(a[2], b[2] - 1);
    REQUIRE_EQUALS(a[3], b[3] - 1);
    REQUIRE_EQUALS(a[4], b[4] - 1);
    REQUIRE_EQUALS(a[5], b[5] - 1);
    REQUIRE_EQUALS(a[6], b[6] - 1);
    REQUIRE_EQUALS(a[7], b[7] - 1);
    REQUIRE_EQUALS(a[8], b[8] - 1);
    REQUIRE_EQUALS(a[9], b[9] - 1);
}

ETL_TEST_CASE("globals/parallel_shuffle/3", "[globals]") {
    etl::fast_matrix<double, 1, 10> a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<double, 1, 10> b{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    etl::fast_matrix<double, 1, 10> c{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    parallel_shuffle(a, b);

    REQUIRE_DIRECT(a == c);
    REQUIRE_DIRECT(b == c);
}

TEMPLATE_TEST_CASE_2("globals/binarize/0", "[globals]", Z, double, float) {
    etl::dyn_vector<Z> x  = {1.0, 200.0, 30.0, 50.0};

    binarize(x, Z(25));

    REQUIRE(x[0] == 0.0);
    REQUIRE(x[1] == 1.0);
    REQUIRE(x[2] == 1.0);
    REQUIRE(x[3] == 1.0);
}

TEMPLATE_TEST_CASE_2("globals/binarize/1", "[globals]", Z, double, float) {
    etl::dyn_vector<Z> x  = {1.0, 200.0, 30.0, 50.0};

    binarize(x, Z(180));

    REQUIRE(x[0] == 0.0);
    REQUIRE(x[1] == 1.0);
    REQUIRE(x[2] == 0.0);
    REQUIRE(x[3] == 0.0);
}

TEMPLATE_TEST_CASE_2("globals/binarize/2", "[globals]", Z, double, float) {
    etl::dyn_vector<Z> x  = {1.0, 200.0, 30.0, 50.0};

    binarize(x, Z(225));

    REQUIRE(x[0] == 0.0);
    REQUIRE(x[1] == 0.0);
    REQUIRE(x[2] == 0.0);
    REQUIRE(x[3] == 0.0);
}

TEMPLATE_TEST_CASE_2("globals/normalize/0", "[globals]", Z, double, float) {
    etl::dyn_vector<Z> x  = {1.0, 2.0, 3.0, 4.0, 5.0};

    normalize_flat(x);

    REQUIRE_EQUALS_APPROX(x[0], -1.4142);
    REQUIRE_EQUALS_APPROX(x[1], -0.7071);
    REQUIRE_EQUALS_APPROX(x[2], 0.0);
    REQUIRE_EQUALS_APPROX(x[3], 0.7071);
    REQUIRE_EQUALS_APPROX(x[4], 1.4142);
}

TEMPLATE_TEST_CASE_2("globals/normalize/1", "[globals]", Z, double, float) {
    etl::fast_dyn_matrix<Z, 2, 5> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    normalize_sub(x);

    REQUIRE_EQUALS_APPROX(x[0], -1.4142);
    REQUIRE_EQUALS_APPROX(x[1], -0.7071);
    REQUIRE_EQUALS_APPROX(x[2], 0.0);
    REQUIRE_EQUALS_APPROX(x[3], 0.7071);
    REQUIRE_EQUALS_APPROX(x[4], 1.4142);

    REQUIRE_EQUALS_APPROX(x[5], -1.4142);
    REQUIRE_EQUALS_APPROX(x[6], -0.7071);
    REQUIRE_EQUALS_APPROX(x[7], 0.0);
    REQUIRE_EQUALS_APPROX(x[8], 0.7071);
    REQUIRE_EQUALS_APPROX(x[9], 1.4142);
}

TEMPLATE_TEST_CASE_2("globals/normalize/2", "[globals]", Z, double, float) {
    etl::fast_dyn_matrix<Z, 2, 5> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    normalize_flat(x);

    REQUIRE_EQUALS_APPROX(x[0], -1.5667);
    REQUIRE_EQUALS_APPROX(x[1], -1.218543);
    REQUIRE_EQUALS_APPROX(x[2], -0.87039);
    REQUIRE_EQUALS_APPROX(x[3], -0.52223);
    REQUIRE_EQUALS_APPROX(x[4], -0.17407);

    REQUIRE_EQUALS_APPROX(x[5], 0.174077);
    REQUIRE_EQUALS_APPROX(x[6], 0.52222);
    REQUIRE_EQUALS_APPROX(x[7], 0.87039);
    REQUIRE_EQUALS_APPROX(x[8], 1.218543);
    REQUIRE_EQUALS_APPROX(x[9], 1.5667);
}
