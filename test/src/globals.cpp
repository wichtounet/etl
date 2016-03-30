//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

TEST_CASE("globals/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a;

    REQUIRE(a.is_square());
    REQUIRE(is_square(a));

    REQUIRE(!a.is_rectangular());
    REQUIRE(!is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_square());
    REQUIRE(is_square(a + a));

    REQUIRE(!expr.is_rectangular());
    REQUIRE(!is_rectangular(a + a));
}

TEST_CASE("globals/2", "[globals]") {
    etl::fast_matrix<double, 3, 2> a;

    REQUIRE(!a.is_square());
    REQUIRE(!is_square(a));

    REQUIRE(a.is_rectangular());
    REQUIRE(is_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_square());
    REQUIRE(!is_square(a + a));

    REQUIRE(expr.is_rectangular());
    REQUIRE(is_rectangular(a + a));
}

TEST_CASE("globals/3", "[globals]") {
    etl::fast_matrix<double, 3, 2, 2> a;

    REQUIRE(a.is_sub_square());
    REQUIRE(is_sub_square(a));

    REQUIRE(is_square(a(1)));

    REQUIRE(!a.is_sub_rectangular());
    REQUIRE(!is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_sub_square());
    REQUIRE(is_sub_square(a + a));

    REQUIRE(!expr.is_sub_rectangular());
    REQUIRE(!is_sub_rectangular(a + a));
}

TEST_CASE("globals/4", "[globals]") {
    etl::fast_matrix<double, 3, 2, 3> a;

    REQUIRE(!a.is_sub_square());
    REQUIRE(!is_sub_square(a));

    REQUIRE(is_rectangular(a(1)));

    REQUIRE(a.is_sub_rectangular());
    REQUIRE(is_sub_rectangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_sub_square());
    REQUIRE(!is_sub_square(a + a));

    REQUIRE(expr.is_sub_rectangular());
    REQUIRE(is_sub_rectangular(a + a));
}

TEST_CASE("globals/is_symmetric/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 2.0, 1.0};

    REQUIRE(a.is_symmetric());
    REQUIRE(is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_symmetric());
    REQUIRE(is_symmetric(expr));
}

TEST_CASE("globals/is_symmetric/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 2.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(a.is_symmetric());
    REQUIRE(is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_symmetric());
    REQUIRE(is_symmetric(expr));
}

TEST_CASE("globals/is_symmetric/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_symmetric());
    REQUIRE(!is_symmetric(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_symmetric());
    REQUIRE(!is_symmetric(expr));
}

TEST_CASE("globals/is_lower_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 2.0, 1.0};

    REQUIRE(a.is_lower_triangular());
    REQUIRE(is_lower_triangular(a));
    REQUIRE(a.is_triangular());
    REQUIRE(is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_lower_triangular());
    REQUIRE(is_lower_triangular(expr));
    REQUIRE(expr.is_triangular());
    REQUIRE(is_triangular(expr));
}

TEST_CASE("globals/is_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0, 4.0, 3.0};

    REQUIRE(a.is_lower_triangular());
    REQUIRE(is_lower_triangular(a));
    REQUIRE(a.is_triangular());
    REQUIRE(is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_lower_triangular());
    REQUIRE(is_lower_triangular(expr));
    REQUIRE(expr.is_triangular());
    REQUIRE(is_triangular(expr));
}

TEST_CASE("globals/is_lower_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_lower_triangular());
    REQUIRE(!is_lower_triangular(a));
    REQUIRE(!a.is_triangular());
    REQUIRE(!is_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_lower_triangular());
    REQUIRE(!is_lower_triangular(expr));
    REQUIRE(!expr.is_triangular());
    REQUIRE(!is_triangular(expr));
}

TEST_CASE("globals/is_strictly_lower_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{0.0, 0.0, 2.0, 0.0};

    REQUIRE(a.is_strictly_lower_triangular());
    REQUIRE(is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_strictly_lower_triangular());
    REQUIRE(is_strictly_lower_triangular(expr));
}

TEST_CASE("globals/is_strictly_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0};

    REQUIRE(a.is_strictly_lower_triangular());
    REQUIRE(is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_strictly_lower_triangular());
    REQUIRE(is_strictly_lower_triangular(expr));
}

TEST_CASE("globals/is_strictly_lower_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_strictly_lower_triangular());
    REQUIRE(!is_strictly_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_strictly_lower_triangular());
    REQUIRE(!is_strictly_lower_triangular(expr));
}

TEST_CASE("globals/is_upper_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 1.0, 0.0, 1.0};

    REQUIRE(a.is_upper_triangular());
    REQUIRE(is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_upper_triangular());
    REQUIRE(is_upper_triangular(expr));
}

TEST_CASE("globals/is_upper_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};

    REQUIRE(a.is_upper_triangular());
    REQUIRE(is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_upper_triangular());
    REQUIRE(is_upper_triangular(expr));
}

TEST_CASE("globals/is_upper_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_upper_triangular());
    REQUIRE(!is_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_upper_triangular());
    REQUIRE(!is_upper_triangular(expr));
}

TEST_CASE("globals/is_strictly_upper_triangular/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{0.0, 1.0, 0.0, 0.0};

    REQUIRE(a.is_strictly_upper_triangular());
    REQUIRE(is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_strictly_upper_triangular());
    REQUIRE(is_strictly_upper_triangular(expr));
}

TEST_CASE("globals/is_strictly_upper_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{0.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

    REQUIRE(a.is_strictly_upper_triangular());
    REQUIRE(is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_strictly_upper_triangular());
    REQUIRE(is_strictly_upper_triangular(expr));
}

TEST_CASE("globals/is_strictly_upper_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_strictly_upper_triangular());
    REQUIRE(!is_strictly_upper_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_strictly_upper_triangular());
    REQUIRE(!is_strictly_upper_triangular(expr));
}

TEST_CASE("globals/is_uniform/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a(1.0);

    REQUIRE(a.is_uniform());
    REQUIRE(is_uniform(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_uniform());
    REQUIRE(is_uniform(expr));
}

TEST_CASE("globals/is_uniform/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_uniform());
    REQUIRE(!is_uniform(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_uniform());
    REQUIRE(!is_uniform(expr));
}

TEST_CASE("globals/is_diagonal/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -1.0};

    REQUIRE(a.is_diagonal());
    REQUIRE(is_diagonal(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_diagonal());
    REQUIRE(is_diagonal(expr));
}

TEST_CASE("globals/is_diagonal/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, -1.0};

    REQUIRE(!a.is_diagonal());
    REQUIRE(!is_diagonal(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_diagonal());
    REQUIRE(!is_diagonal(expr));
}

TEST_CASE("globals/trace/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 2.0, 3.0, 4.0};

    REQUIRE(trace(a) == Approx(5.0));

    decltype(auto) expr = a + a;

    REQUIRE(trace(expr) == Approx(10.0));
}

TEST_CASE("globals/trace/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    REQUIRE(trace(a) == Approx(15.0));

    decltype(auto) expr = a + a;

    REQUIRE(trace(expr) == Approx(30.0));
}

TEST_CASE("globals/is_real_complex/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a;
    etl::fast_matrix<float, 3, 3> b;
    etl::fast_matrix<int, 3, 3> c;
    etl::fast_matrix<long, 3, 3> d;
    etl::fast_matrix<std::complex<float>, 3, 3> e;
    etl::fast_matrix<std::complex<double>, 3, 3> f;
    etl::fast_matrix<etl::complex<float>, 3, 3> g;
    etl::fast_matrix<etl::complex<double>, 3, 3> h;

    REQUIRE(etl::is_real_matrix(a));
    REQUIRE(etl::is_real_matrix(b));
    REQUIRE(etl::is_real_matrix(c));
    REQUIRE(etl::is_real_matrix(d));
    REQUIRE(!etl::is_real_matrix(e));
    REQUIRE(!etl::is_real_matrix(f));
    REQUIRE(!etl::is_real_matrix(g));
    REQUIRE(!etl::is_real_matrix(h));

    REQUIRE(!etl::is_complex_matrix(a));
    REQUIRE(!etl::is_complex_matrix(b));
    REQUIRE(!etl::is_complex_matrix(c));
    REQUIRE(!etl::is_complex_matrix(d));
    REQUIRE(etl::is_complex_matrix(e));
    REQUIRE(etl::is_complex_matrix(f));
    REQUIRE(etl::is_complex_matrix(g));
    REQUIRE(etl::is_complex_matrix(h));
}

TEST_CASE("globals/is_hermitian/1", "[globals]") {
    etl::fast_matrix<double, 3, 3> a;
    etl::fast_matrix<float, 3, 3> b;
    etl::fast_matrix<int, 3, 3> c;
    etl::fast_matrix<long, 3, 3> d;

    REQUIRE(!is_hermitian(a));
    REQUIRE(!is_hermitian(b));
    REQUIRE(!is_hermitian(c));
    REQUIRE(!is_hermitian(d));
}

TEST_CASE("globals/is_hermitian/2", "[globals]") {
    etl::fast_matrix<std::complex<double>, 3, 3> a;
    etl::fast_matrix<std::complex<float>, 3, 3> b;
    etl::fast_matrix<etl::complex<double>, 3, 3> c;
    etl::fast_matrix<etl::complex<float>, 3, 3> d;

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

    REQUIRE(!is_hermitian(a));
    REQUIRE(!is_hermitian(b));
    REQUIRE(!is_hermitian(c));
    REQUIRE(!is_hermitian(d));
}

TEST_CASE("globals/is_hermitian/3", "[globals]") {
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

    REQUIRE(is_hermitian(a));
    REQUIRE(is_hermitian(b));
    REQUIRE(is_hermitian(c));
    REQUIRE(is_hermitian(d));
}

TEST_CASE("globals/is_permutation/1", "[globals]") {
    etl::fast_matrix<double, 2, 2> a{1.0, 0.0, 1.0, 0.0};
    etl::fast_matrix<double, 2, 2> b{1.0, 1.0, 1.0, 0.0};
    etl::fast_matrix<double, 2, 2> c{1.0, 0.0, 0.0, 0.0};
    etl::fast_matrix<double, 2, 2> d{1.0, 0.0, 0.0, 1.0};
    etl::fast_matrix<double, 2, 2> e{0.0, 1.0, 1.0, 0.0};

    REQUIRE(!is_permutation_matrix(a));
    REQUIRE(!is_permutation_matrix(b));
    REQUIRE(!is_permutation_matrix(c));
    REQUIRE(is_permutation_matrix(d));
    REQUIRE(is_permutation_matrix(e));
}

TEST_CASE("globals/is_permutation/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> b{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> c{1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    etl::fast_matrix<double, 3, 3> d{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};

    REQUIRE(!is_permutation_matrix(a));
    REQUIRE(!is_permutation_matrix(b));
    REQUIRE(!is_permutation_matrix(c));
}
