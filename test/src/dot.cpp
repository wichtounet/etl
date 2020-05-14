//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "dot_test.hpp"

DOT_TEST_CASE("dot/1", "[dot]") {
    etl::fast_vector<T, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<T, 3> b = {2.0, 3.0, 2.0};

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, 21.0);
}

DOT_TEST_CASE("dot/2", "[dot]") {
    etl::fast_vector<T, 3> a = {-1.0, 2.0, 8.5};
    etl::fast_vector<T, 3> b = {-2.0, -3.0, -2.0};

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, -21.0);
}

DOT_TEST_CASE("dot/3", "[dot]") {
    etl::dyn_vector<T> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<T> b({2.0, 3.0, 2.0});

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, 21.0);
}

DOT_TEST_CASE("dot/4", "[dot]") {
    etl::dyn_vector<T> a({-1.0, 2.0, 8.5});
    etl::dyn_vector<T> b({-2.0, -3.0, -2.0});

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, -21.0);
}

DOT_TEST_CASE("dot/5", "[dot]") {
    etl::dyn_vector<T> a(15);
    etl::dyn_vector<T> b(15);

    a = etl::sequence_generator<T>(1);
    b = etl::sequence_generator<T>(2);

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, 1360.0);
}

DOT_TEST_CASE("dot/6", "[dot]") {
    etl::dyn_vector<T> a(33);
    etl::dyn_vector<T> b(33);

    a = etl::sequence_generator<T>(1);
    b = etl::sequence_generator<T>(2);

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, 13090.0);
}

DOT_TEST_CASE("dot/7", "[dot]") {
    etl::dyn_vector<T> a(57);
    etl::dyn_vector<T> b(57);

    a = etl::sequence_generator<T>(1);
    b = etl::sequence_generator<T>(2);

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS(value, 65018.0);
}

DOT_TEST_CASE("dot/8", "[dot]") {
    etl::dyn_vector<T> a(1024 - 7);
    etl::dyn_vector<T> b(1024 - 7);

    a = T(0.01) * etl::sequence_generator<T>(1);
    b = T(0.02) * etl::sequence_generator<T>(2);

    T value = 0;
    Impl::apply(a, b, value);

    REQUIRE_EQUALS_APPROX(value, 70331.7876);
}
