//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "etl/stop.hpp"

TEMPLATE_TEST_CASE_2("stop/fast_vector_1", "stop<unary<fast_vec>>", Z, float, double) {
    using mat_type = etl::fast_vector<Z, 4>;
    mat_type test_vector(3.3);

    auto r = s(log(test_vector));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 4UL);
    REQUIRE_EQUALS(etl::size(r), 4UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();
    REQUIRE_EQUALS(size_1, 4UL);

    constexpr const auto size_2 = etl::size(r);
    REQUIRE_EQUALS(size_2, 4UL);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS_APPROX(r[i], std::log(Z(3.3)));
        REQUIRE_EQUALS_APPROX(r(i), std::log(Z(3.3)));
    }
}

TEMPLATE_TEST_CASE_2("stop/fast_matrix_1", "stop<unary<fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    auto r = s(log(test_matrix));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 6UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 6UL);
    REQUIRE_EQUALS(etl::size(r), 6UL);
    REQUIRE_EQUALS(etl::rows(r), 3UL);
    REQUIRE_EQUALS(etl::columns(r), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();

    REQUIRE_EQUALS(size_1, 6UL);

    constexpr const auto size_2    = etl::size(r);
    constexpr const auto rows_2    = etl::rows(r);
    constexpr const auto columns_2 = etl::columns(r);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS_APPROX(r[i], std::log(Z(3.3)));
    }
}

TEMPLATE_TEST_CASE_2("stop/fast_matrix_2", "stop<binary<fast_mat>>", Z, float, double) {
    using mat_type = etl::fast_matrix<Z, 3, 2>;
    mat_type test_matrix(3.3);

    auto r = s(test_matrix + test_matrix);

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 6UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 6UL);
    REQUIRE_EQUALS(etl::size(r), 6UL);
    REQUIRE_EQUALS(etl::rows(r), 3UL);
    REQUIRE_EQUALS(etl::columns(r), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_fast);

    constexpr const auto size_1 = etl::etl_traits<type>::size();

    REQUIRE_EQUALS(size_1, 6UL);

    constexpr const auto size_2    = etl::size(r);
    constexpr const auto rows_2    = etl::rows(r);
    constexpr const auto columns_2 = etl::columns(r);

    REQUIRE_EQUALS(size_2, 6UL);
    REQUIRE_EQUALS(rows_2, 3UL);
    REQUIRE_EQUALS(columns_2, 2UL);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS(r[i], Z(6.6));
    }
}

TEMPLATE_TEST_CASE_2("stop/dyn_vector_1", "stop<unary<dyn_vec>>", Z, float, double) {
    using mat_type = etl::dyn_vector<Z>;
    mat_type test_vector(4, 3.3);

    auto r = s(log(test_vector));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 4UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 4UL);
    REQUIRE_EQUALS(etl::size(r), 4UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<type>::is_fast);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS_APPROX(r[i], std::log(Z(3.3)));
        REQUIRE_EQUALS_APPROX(r(i), std::log(Z(3.3)));
    }
}

TEMPLATE_TEST_CASE_2("stop/dyn_matrix_1", "stop<unary<dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
    mat_type test_matrix(3, 2, 3.3);

    auto r = s(log(test_matrix));

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 6UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 6UL);
    REQUIRE_EQUALS(etl::size(r), 6UL);
    REQUIRE_EQUALS(etl::rows(r), 3UL);
    REQUIRE_EQUALS(etl::columns(r), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<type>::is_fast);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS_APPROX(r[i], std::log(Z(3.3)));
    }
}

TEMPLATE_TEST_CASE_2("stop/dyn_matrix_2", "stop<binary<dyn_mat>>", Z, float, double) {
    using mat_type = etl::dyn_matrix<Z>;
    mat_type test_matrix(3, 2, 3.3);

    auto r = s(test_matrix + test_matrix);

    using type = std::remove_reference_t<std::remove_cv_t<decltype(r)>>;

    REQUIRE_EQUALS(r.size(), 6UL);
    REQUIRE_EQUALS(etl::etl_traits<type>::size(r), 6UL);
    REQUIRE_EQUALS(etl::size(r), 6UL);
    REQUIRE_EQUALS(etl::rows(r), 3UL);
    REQUIRE_EQUALS(etl::columns(r), 2UL);
    REQUIRE_DIRECT(etl::etl_traits<type>::is_value);
    REQUIRE_DIRECT(!etl::etl_traits<type>::is_fast);

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_EQUALS(r[i], Z(6.6));
    }
}
