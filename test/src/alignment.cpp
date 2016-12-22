//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

namespace {

template<typename T>
struct outer {
    int five;
    T inner;
    int special;

    outer() : inner() {}
    outer(std::size_t n) : inner(n){}
};

} // end of anonymous namespace

TEMPLATE_TEST_CASE_4("alignment/fast/1", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::fast_vector<ZZZ, 5>;

    type a;
    auto b = a;
    auto* c = new type();

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/fast/2", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::fast_vector<ZZZ, 5>;

    outer<type> a;
    auto b = a;
    auto* c = new outer<type>();

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/fast_dyn/1", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::fast_dyn_vector<ZZZ, 5>;

    type a;
    auto b = a;
    auto* c = new type();

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/fast_dyn/2", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::fast_dyn_vector<ZZZ, 5>;

    outer<type> a;
    auto b = a;
    auto* c = new outer<type>();

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/dyn/1", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::dyn_vector<ZZZ>;

    type a(5);
    auto b = a;
    auto* c = new type(5);

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/dyn/2", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    using type = etl::dyn_vector<ZZZ>;

    outer<type> a(5);
    auto b = a;
    auto* c = new outer<type>(5);

    REQUIRE_DIRECT(reinterpret_cast<size_t>(a.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(b.inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
    REQUIRE_DIRECT(reinterpret_cast<size_t>(c->inner.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/temporary/1", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<ZZZ> a(3, 3);
    etl::dyn_matrix<ZZZ> b(3, 3);

    auto c = a * b;
    *c;

    REQUIRE_DIRECT(reinterpret_cast<size_t>(c.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/temporary/2", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    etl::dyn_vector<ZZZ> a(3);
    etl::dyn_matrix<ZZZ> b(3, 3);

    auto c = a * b;
    *c;

    REQUIRE_DIRECT(reinterpret_cast<size_t>(c.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}

TEMPLATE_TEST_CASE_4("alignment/temporary/3", "etl_traits<fast_vector>", ZZZ, double, float, std::complex<float>, std::complex<double>) {
    etl::dyn_matrix<ZZZ> a(3, 3);
    etl::dyn_vector<ZZZ> b(3);

    auto c = a * b;
    *c;

    REQUIRE_DIRECT(reinterpret_cast<size_t>(c.memory_start()) % etl::default_intrinsic_traits<ZZZ>::alignment == 0);
}
