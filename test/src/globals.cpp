//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_lower_triangular());
    REQUIRE(is_lower_triangular(expr));
}

TEST_CASE("globals/is_lower_triangular/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 3.0, 4.0, 3.0};

    REQUIRE(a.is_lower_triangular());
    REQUIRE(is_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(expr.is_lower_triangular());
    REQUIRE(is_lower_triangular(expr));
}

TEST_CASE("globals/is_lower_triangular/3", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{2.0, 2.0, 3.0, 5.0, 2.0, 4.0, 3.0, 4.0, 3.0};

    REQUIRE(!a.is_lower_triangular());
    REQUIRE(!is_lower_triangular(a));

    decltype(auto) expr = a + a;

    REQUIRE(!expr.is_lower_triangular());
    REQUIRE(!is_lower_triangular(expr));
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

    REQUIRE(trace(a) == 5.0);

    decltype(auto) expr = a + a;

    REQUIRE(trace(expr) == Approx(10.0));
}

TEST_CASE("globals/trace/2", "[globals]") {
    etl::fast_matrix<double, 3, 3> a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    REQUIRE(trace(a) == Approx(15.0));

    decltype(auto) expr = a + a;

    REQUIRE(trace(expr) == Approx(30.0));
}
