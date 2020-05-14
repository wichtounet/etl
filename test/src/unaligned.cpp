//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test_light.hpp"
#include "cpp_utils/algorithm.hpp"

namespace {

template <typename T>
struct unaligned_ptr {
    T* base;
    T* frakked_up;

    unaligned_ptr(T* base, T* frakked_up)
            : base(base), frakked_up(frakked_up) {}

    unaligned_ptr(const unaligned_ptr& rhs) = delete;
    unaligned_ptr& operator=(const unaligned_ptr& rhs) = delete;

    unaligned_ptr(unaligned_ptr&& rhs) {
        base       = rhs.base;
        frakked_up = rhs.frakked_up;

        rhs.base       = nullptr;
        rhs.frakked_up = nullptr;
    }

    unaligned_ptr& operator=(unaligned_ptr&& rhs) {
        if (this != &rhs) {
            if (base) {
                free(base);
            }

            base       = rhs.base;
            frakked_up = rhs.frakked_up;

            rhs.base       = nullptr;
            rhs.frakked_up = nullptr;
        }

        return *this;
    }

    ~unaligned_ptr() {
        if (base) {
            free(base);
        }
    }

    T* get() {
        return frakked_up;
    }
};

template <typename T>
unaligned_ptr<T> get_unaligned_memory(size_t n) {
    auto required_bytes = (1 + sizeof(T)) * n;
    auto orig           = static_cast<T*>(malloc(required_bytes));

    if (!orig) {
        return {nullptr, nullptr};
    }

    if (reinterpret_cast<uintptr_t>(orig) % 16 == 0) {
        return {orig, orig + 1};
    } else {
        return {orig, orig};
    }
}

} // end of anonymous namespace

// By default all ETL memory is aligned, but it should support
// unaligned as well

TEMPLATE_TEST_CASE_2("unaligned/assign", "[unaligned][assign]", Z, double, float) {
    auto mem_a = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_b = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_c = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);

    etl::custom_dyn_matrix<Z> a(mem_a.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> b(mem_b.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> c(mem_c.get(), etl::parallel_threshold + 100UL, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);

    c = a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(a[i] + b[i]));
    }
}

TEMPLATE_TEST_CASE_2("unaligned/add", "[unaligned][add]", Z, double, float) {
    auto mem_a = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_b = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_c = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);

    etl::custom_dyn_matrix<Z> a(mem_a.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> b(mem_b.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> c(mem_c.get(), etl::parallel_threshold + 100UL, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 1200.0;

    c += a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0) + (a[i] + b[i]));
    }
}

TEMPLATE_TEST_CASE_2("unaligned/sub", "[unaligned][add]", Z, double, float) {
    auto mem_a = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_b = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_c = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);

    etl::custom_dyn_matrix<Z> a(mem_a.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> b(mem_b.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> c(mem_c.get(), etl::parallel_threshold + 100UL, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 1200.0;

    c -= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0) - (a[i] + b[i]));
    }
}

TEMPLATE_TEST_CASE_2("unaligned/mul", "[unaligned][add]", Z, double, float) {
    auto mem_a = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_b = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_c = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);

    etl::custom_dyn_matrix<Z> a(mem_a.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> b(mem_b.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> c(mem_c.get(), etl::parallel_threshold + 100UL, 1UL);

    a = etl::uniform_generator(-1000.0, 5000.0);
    b = etl::uniform_generator(-1000.0, 5000.0);
    c = 1200.0;

    c *= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0) * (a[i] + b[i]));
    }
}

TEMPLATE_TEST_CASE_2("unaligned/div", "[unaligned][add]", Z, double, float) {
    auto mem_a = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_b = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);
    auto mem_c = get_unaligned_memory<Z>(etl::parallel_threshold + 100UL);

    etl::custom_dyn_matrix<Z> a(mem_a.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> b(mem_b.get(), etl::parallel_threshold + 100UL, 1UL);
    etl::custom_dyn_matrix<Z> c(mem_c.get(), etl::parallel_threshold + 100UL, 1UL);

    a = etl::uniform_generator(1000.0, 5000.0);
    b = etl::uniform_generator(1000.0, 5000.0);
    c = 1200.0;

    c /= a + b;

    for (size_t i = 0; i < c.size(); ++i) {
        REQUIRE_EQUALS_APPROX(c[i], Z(1200.0) / (a[i] + b[i]));
    }
}
