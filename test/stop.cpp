#include "catch.hpp"

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/dyn_vector.hpp"
#include "etl/dyn_matrix.hpp"
#include "etl/stop.hpp"

TEST_CASE( "stop/fast_vector_1", "stop<unary<fast_vec>>" ) {
    using mat_type = etl::fast_vector<double, 4>;
    mat_type test_vector(3.3);

    auto r = s(log(test_vector));

    using type = remove_reference_t<remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 4);
    REQUIRE(etl_traits<type>::size(r) == 4);
    REQUIRE(size(r) == 4);
    REQUIRE(etl_traits<type>::is_vector);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_matrix);

    constexpr const auto size_1 = etl_traits<type>::size();
    REQUIRE(size_1 == 4);

    constexpr const auto size_2 = size(r);
    REQUIRE(size_2 == 4);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == log(3.3));
    }
}

TEST_CASE( "stop/fast_matrix_1", "stop<unary<fast_mat>>" ) {
    using mat_type = etl::fast_matrix<double, 3, 2>;
    mat_type test_matrix(3.3);

    auto r = s(log(test_matrix));

    using type = remove_reference_t<remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl_traits<type>::size(r) == 6);
    REQUIRE(size(r) == 6);
    REQUIRE(etl_traits<type>::rows(r) == 3);
    REQUIRE(rows(r) == 3);
    REQUIRE(etl_traits<type>::columns(r) == 2);
    REQUIRE(columns(r) == 2);
    REQUIRE(etl_traits<type>::is_matrix);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_vector);

    constexpr const auto size_1 = etl_traits<type>::size();
    constexpr const auto rows_1 = etl_traits<type>::rows();
    constexpr const auto columns_1 = etl_traits<type>::columns();

    REQUIRE(size_1 == 6);
    REQUIRE(rows_1 == 3);
    REQUIRE(columns_1 == 2);

    constexpr const auto size_2 = size(r);
    constexpr const auto rows_2 = rows(r);
    constexpr const auto columns_2 = columns(r);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == log(3.3));
    }
}

TEST_CASE( "stop/fast_matrix_2", "stop<binary<fast_mat>>" ) {
    using mat_type = etl::fast_matrix<double, 3, 2>;
    mat_type test_matrix(3.3);

    auto r = s(test_matrix + test_matrix);

    using type = remove_reference_t<remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl_traits<type>::size(r) == 6);
    REQUIRE(size(r) == 6);
    REQUIRE(etl_traits<type>::rows(r) == 3);
    REQUIRE(rows(r) == 3);
    REQUIRE(etl_traits<type>::columns(r) == 2);
    REQUIRE(columns(r) == 2);
    REQUIRE(etl_traits<type>::is_matrix);
    REQUIRE(etl_traits<type>::is_value);
    REQUIRE(etl_traits<type>::is_fast);
    REQUIRE(!etl_traits<type>::is_vector);

    constexpr const auto size_1 = etl_traits<type>::size();
    constexpr const auto rows_1 = etl_traits<type>::rows();
    constexpr const auto columns_1 = etl_traits<type>::columns();

    REQUIRE(size_1 == 6);
    REQUIRE(rows_1 == 3);
    REQUIRE(columns_1 == 2);

    constexpr const auto size_2 = size(r);
    constexpr const auto rows_2 = rows(r);
    constexpr const auto columns_2 = columns(r);

    REQUIRE(size_2 == 6);
    REQUIRE(rows_2 == 3);
    REQUIRE(columns_2 == 2);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == 6.6);
    }
}