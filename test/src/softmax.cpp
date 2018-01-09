//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"

TEMPLATE_TEST_CASE_2("softmax/1", "fast_vector::softmax", Z, float, double) {
    etl::fast_vector<Z, 3> a = {1.0, 2.0, 3.0};

    etl::fast_vector<Z, 3> d;
    d = etl::softmax(a);

    auto sum = std::exp(Z(1.0)) + std::exp(Z(2.0)) + std::exp(Z(3.0));

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(1.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(2.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(3.0)) / sum);

    REQUIRE_EQUALS_APPROX(etl::mean(d), 1.0 / 3.0);
}

TEMPLATE_TEST_CASE_2("softmax/2", "fast_vector::softmax", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 4.0, 5.0};

    etl::fast_vector<Z, 3> d;
    d = etl::softmax(a);

    auto sum = std::exp(Z(-1.0)) + std::exp(Z(4.0)) + std::exp(Z(5.0));

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(4.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(5.0)) / sum);

    REQUIRE_EQUALS_APPROX(etl::mean(d), 1.0 / 3.0);
}

TEMPLATE_TEST_CASE_2("softmax/3", "fast_vector::softmax", Z, float, double) {
    etl::fast_matrix<Z, 9> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    etl::fast_matrix<Z, 9> d;
    d = etl::softmax(a);

    REQUIRE_EQUALS_APPROX(etl::mean(d), 1.0 / 9.0);
}

TEMPLATE_TEST_CASE_2("softmax/4", "fast_vector::softmax", Z, float, double) {
    etl::fast_matrix<Z, 9> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<Z, 9> b(1.4);

    etl::fast_matrix<Z, 9> d;
    d = etl::softmax(a + (a * 2) + b + (b >> a));

    REQUIRE_EQUALS_APPROX(etl::mean(d), 1.0 / 9.0);
}

TEMPLATE_TEST_CASE_2("softmax/5", "fast_vector::softmax", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<Z, 3, 3> b(1.4);

    etl::fast_matrix<Z, 3, 3> c;
    etl::fast_matrix<Z, 3, 3> c_ref;

    c_ref(0) = etl::softmax(a(0));
    c_ref(1) = etl::softmax(a(1));
    c_ref(2) = etl::softmax(a(2));

    c = etl::softmax(a);

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}

TEMPLATE_TEST_CASE_2("stable_softmax/1", "stable_softmax", Z, float, double) {
    etl::fast_vector<Z, 3> a = {-1.0, 4.0, 5.0};

    etl::fast_vector<Z, 3> d;
    d = etl::stable_softmax(a);

    auto sum = std::exp(Z(-1.0)) + std::exp(Z(4.0)) + std::exp(Z(5.0));

    REQUIRE_EQUALS_APPROX(d[0], std::exp(Z(-1.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[1], std::exp(Z(4.0)) / sum);
    REQUIRE_EQUALS_APPROX(d[2], std::exp(Z(5.0)) / sum);

    REQUIRE_EQUALS_APPROX(etl::mean(d), 1.0 / 3.0);
}

TEMPLATE_TEST_CASE_2("stable_softmax/2", "fast_vector::softmax", Z, float, double) {
    etl::fast_matrix<Z, 3, 3> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    etl::fast_matrix<Z, 3, 3> b(1.4);

    etl::fast_matrix<Z, 3, 3> c;
    etl::fast_matrix<Z, 3, 3> c_ref;

    c_ref(0) = etl::stable_softmax(a(0));
    c_ref(1) = etl::stable_softmax(a(1));
    c_ref(2) = etl::stable_softmax(a(2));

    c = etl::stable_softmax(a);

    for(size_t i = 0; i < etl::size(c); ++i){
        REQUIRE_EQUALS_APPROX(c[i], c_ref[i]);
    }
}
