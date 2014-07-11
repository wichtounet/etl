#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/traits.hpp"

TEST_CASE( "etl_traits/fast_vector_1", "etl_traits<fast_vector>" ) {
    using type = etl::fast_vector<double, 4>;
    type test_vector(3.3);

    REQUIRE(etl_traits<type>::size(test_vector) == 4);
    REQUIRE(size(test_vector) == 4);
    REQUIRE(etl_traits<type>::is_vector);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_matrix);

    constexpr const auto size_1 = etl_traits<type>::size(test_vector);
    REQUIRE(size_1 == 4);

    constexpr const auto size_2 = size(test_vector);
    REQUIRE(size_2 == 4);
}

TEST_CASE( "etl_traits/fast_matrix_1", "etl_traits<fast_matrix>" ) {
    using type = etl::fast_matrix<double, 3, 2>;
    type test_matrix(3.3);

    REQUIRE(etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(size(test_matrix) == 6);
    REQUIRE(etl_traits<type>::rows(test_matrix) == 3);
    REQUIRE(rows(test_matrix) == 3);
    REQUIRE(etl_traits<type>::columns(test_matrix) == 2);
    REQUIRE(columns(test_matrix) == 2);
    REQUIRE(etl_traits<type>::is_matrix);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_vector);

    constexpr const auto size_1 = etl_traits<type>::size(test_matrix);
    constexpr const auto rows_1 = etl_traits<type>::rows(test_matrix);
    constexpr const auto columns_1 = etl_traits<type>::columns(test_matrix);

    REQUIRE(size_1 == 6);
    REQUIRE(rows_1 == 3);
    REQUIRE(columns_1 == 2);

    constexpr const auto size_2 = size(test_matrix);
    constexpr const auto rows_2 = rows(test_matrix);
    constexpr const auto columns_2 = columns(test_matrix);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);
}

TEST_CASE( "etl_traits/dyn_vector_1", "etl_traits<dyn_vector>" ) {
    using type = etl::dyn_vector<double>;
    type test_vector(4, 3.3);

    REQUIRE(etl_traits<type>::size(test_vector) == 4);
    REQUIRE(size(test_vector) == 4);
    REQUIRE(etl_traits<type>::is_vector);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(!etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_matrix);
}

TEST_CASE( "etl_traits/dyn_matrix_1", "etl_traits<dyn_matrix>" ) {
    using type = etl::dyn_matrix<double>;
    type test_matrix(3, 2, 3.3);

    REQUIRE(etl_traits<type>::size(test_matrix) == 6);
    REQUIRE(size(test_matrix) == 6);
    REQUIRE(etl_traits<type>::rows(test_matrix) == 3);
    REQUIRE(rows(test_matrix) == 3);
    REQUIRE(etl_traits<type>::columns(test_matrix) == 2);
    REQUIRE(columns(test_matrix) == 2);
    REQUIRE(etl_traits<type>::is_matrix);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(!etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_vector);
}