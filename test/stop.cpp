//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "catch.hpp"

#include "etl/etl.hpp"
#include "etl/stop.hpp"

TEST_CASE( "stop/fast_vector_1", "stop<unary<fast_vec>>" ) {
    using mat_type = etl::fast_vector<double, 4>;
    mat_type test_vector(3.3);

    auto r = s(log(test_vector));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 4);
    REQUIRE(etl::etl_traits<type>::size(r) == 4);
    REQUIRE(etl::size(r) == 4);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    REQUIRE(size_1 == 4);

    constexpr const auto size_2 = etl::size(r);
    REQUIRE(size_2 == 4);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == log(3.3));
        REQUIRE(r(i) == log(3.3));
    }
}

TEST_CASE( "stop/fast_matrix_1", "stop<unary<fast_mat>>" ) {
    using mat_type = etl::fast_matrix<double, 3, 2>;
    mat_type test_matrix(3.3);

    auto r = s(log(test_matrix));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl::etl_traits<type>::size(r) == 6);
    REQUIRE(etl::size(r) == 6);
    REQUIRE(etl::rows(r) == 3);
    REQUIRE(etl::columns(r) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();

    REQUIRE(size_1 == 6);

    constexpr const auto size_2 = etl::size(r);
    constexpr const auto rows_2 = etl::rows(r);
    constexpr const auto columns_2 = etl::columns(r);

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

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl::etl_traits<type>::size(r) == 6);
    REQUIRE(etl::size(r) == 6);
    REQUIRE(etl::rows(r) == 3);
    REQUIRE(etl::columns(r) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();

    REQUIRE(size_1 == 6);

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

TEST_CASE( "stop/dyn_vector_1", "stop<unary<dyn_vec>>" ) {
    using mat_type = etl::dyn_vector<double>;
    mat_type test_vector(4, 3.3);

    auto r = s(log(test_vector));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 4);
    REQUIRE(etl::etl_traits<type>::size(r) == 4);
    REQUIRE(etl::size(r) == 4);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == log(3.3));
        REQUIRE(r(i) == log(3.3));
    }
}

TEST_CASE( "stop/dyn_matrix_1", "stop<unary<dyn_mat>>" ) {
    using mat_type = etl::dyn_matrix<double>;
    mat_type test_matrix(3, 2, 3.3);

    auto r = s(log(test_matrix));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl::etl_traits<type>::size(r) == 6);
    REQUIRE(etl::size(r) == 6);
    REQUIRE(etl::rows(r) == 3);
    REQUIRE(etl::columns(r) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == log(3.3));
    }
}

TEST_CASE( "stop/dyn_matrix_2", "stop<binary<dyn_mat>>" ) {
    using mat_type = etl::dyn_matrix<double>;
    mat_type test_matrix(3, 2, 3.3);

    auto r = s(test_matrix + test_matrix);

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE(r.size() == 6);
    REQUIRE(etl::etl_traits<type>::size(r) == 6);
    REQUIRE(etl::size(r) == 6);
    REQUIRE(etl::rows(r) == 3);
    REQUIRE(etl::columns(r) == 2);
    REQUIRE(etl::etl_traits<type>::is_value);
    REQUIRE(!etl::etl_traits<type>::is_fast);

    for(std::size_t i = 0; i < r.size(); ++i){
        REQUIRE(r[i] == 6.6);
    }
}