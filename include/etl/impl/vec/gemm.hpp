//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//======================================================================= #pragma once

// The idea of the kernel is largely inspired by the kernels in Blaze
// by Klaus Igleberg

namespace etl {

namespace impl {

namespace vec {

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv_small_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(a);
    const auto n = columns(a);

    //TODO Add FMA (fmadd) support to vectorization

    size_t i = 0;

    // 8-Unrolled outer loop
    for (; i + 7 < m; i += 8) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto b1 = b.template loadu<vec_type>(k);

            auto a1 = a.template loadu<vec_type>((i + 0) * n + k);
            auto a2 = a.template loadu<vec_type>((i + 1) * n + k);
            auto a3 = a.template loadu<vec_type>((i + 2) * n + k);
            auto a4 = a.template loadu<vec_type>((i + 3) * n + k);
            auto a5 = a.template loadu<vec_type>((i + 4) * n + k);
            auto a6 = a.template loadu<vec_type>((i + 5) * n + k);
            auto a7 = a.template loadu<vec_type>((i + 6) * n + k);
            auto a8 = a.template loadu<vec_type>((i + 7) * n + k);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a2, b1));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a3, b1));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a4, b1));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a5, b1));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a6, b1));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a7, b1));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a8, b1));
        }

        c[i + 0] = vec_type::template hadd<T>(r1);
        c[i + 1] = vec_type::template hadd<T>(r2);
        c[i + 2] = vec_type::template hadd<T>(r3);
        c[i + 3] = vec_type::template hadd<T>(r4);
        c[i + 4] = vec_type::template hadd<T>(r5);
        c[i + 5] = vec_type::template hadd<T>(r6);
        c[i + 6] = vec_type::template hadd<T>(r7);
        c[i + 7] = vec_type::template hadd<T>(r8);

        // Remainder inner loop
        for (; k < n; ++k) {
            c[i + 0] += a(i + 0, k) * b(k);
            c[i + 1] += a(i + 1, k) * b(k);
            c[i + 2] += a(i + 2, k) * b(k);
            c[i + 3] += a(i + 3, k) * b(k);
            c[i + 4] += a(i + 4, k) * b(k);
            c[i + 5] += a(i + 5, k) * b(k);
            c[i + 6] += a(i + 6, k) * b(k);
            c[i + 7] += a(i + 7, k) * b(k);
        }
    }

    // 2-Unrolled outer loop
    for (; i + 1 < m; i += 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto b1 = b.template loadu<vec_type>(k);

            auto a1 = a.template loadu<vec_type>((i + 0) * n + k);
            auto a2 = a.template loadu<vec_type>((i + 1) * n + k);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a2, b1));
        }

        c[i + 0] = vec_type::template hadd<T>(r1);
        c[i + 1] = vec_type::template hadd<T>(r2);

        // Remainder inner loop
        for (; k < n; ++k) {
            c[i + 0] += a(i + 0, k) * b(k);
            c[i + 1] += a(i + 1, k) * b(k);
        }
    }

    // Remainder loop
    for (; i < m; ++i) {
        auto r1 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto a1 = a.template loadu<vec_type>(i * n + k);
            auto b1 = b.template loadu<vec_type>(k);

            auto t1 = vec_type::template mul<Cx>(a1, b1);
            r1 = vec_type::add(r1, t1);
        }

        auto result = vec_type::template hadd<T>(r1);

        // Remainder inner loop
        for (; k < n; ++k) {
            result += a(i, k) * b(k);
        }

        c[i] = result;
    }
}

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv_large_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(a);
    const auto n = columns(a);

    //TODO Add FMA (fmadd) support to vectorization

    size_t i = 0;

    // 8-Unrolled outer loop
    for (; i + 7 < m; i += 8) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        size_t k = 0;

        // 4-Unrolled Vectorized inner loop
        for (; k + (vec_size * 4) - 1 < n; k += vec_size * 4) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;
            const auto k3 = k + 2 * vec_size;
            const auto k4 = k + 3 * vec_size;

            auto b1 = b.template loadu<vec_type>(k1);
            auto b2 = b.template loadu<vec_type>(k2);
            auto b3 = b.template loadu<vec_type>(k3);
            auto b4 = b.template loadu<vec_type>(k4);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k1), b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k1), b1));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k1), b1));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k1), b1));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k1), b1));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k1), b1));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k1), b1));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k1), b1));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k2), b2));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k2), b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k2), b2));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k2), b2));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k2), b2));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k2), b2));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k2), b2));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k2), b2));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k3), b3));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k3), b3));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k3), b3));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k3), b3));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k3), b3));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k3), b3));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k3), b3));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k3), b3));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k4), b4));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k4), b4));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k4), b4));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k4), b4));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k4), b4));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k4), b4));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k4), b4));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k4), b4));
        }

        // 2-Unrolled Vectorized inner loop
        for (; k + (vec_size * 2) - 1 < n; k += vec_size * 2) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;

            auto b1 = b.template loadu<vec_type>(k1);
            auto b2 = b.template loadu<vec_type>(k2);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k1), b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k1), b1));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k1), b1));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k1), b1));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k1), b1));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k1), b1));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k1), b1));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k1), b1));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k2), b2));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k2), b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 2) * n + k2), b2));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 3) * n + k2), b2));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 4) * n + k2), b2));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 5) * n + k2), b2));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 6) * n + k2), b2));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 7) * n + k2), b2));
        }

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto b1 = b.template loadu<vec_type>(k);

            auto a1 = a.template loadu<vec_type>((i + 0) * n + k);
            auto a2 = a.template loadu<vec_type>((i + 1) * n + k);
            auto a3 = a.template loadu<vec_type>((i + 2) * n + k);
            auto a4 = a.template loadu<vec_type>((i + 3) * n + k);
            auto a5 = a.template loadu<vec_type>((i + 4) * n + k);
            auto a6 = a.template loadu<vec_type>((i + 5) * n + k);
            auto a7 = a.template loadu<vec_type>((i + 6) * n + k);
            auto a8 = a.template loadu<vec_type>((i + 7) * n + k);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a2, b1));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a3, b1));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a4, b1));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a5, b1));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a6, b1));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a7, b1));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a8, b1));
        }

        c[i + 0] = vec_type::template hadd<T>(r1);
        c[i + 1] = vec_type::template hadd<T>(r2);
        c[i + 2] = vec_type::template hadd<T>(r3);
        c[i + 3] = vec_type::template hadd<T>(r4);
        c[i + 4] = vec_type::template hadd<T>(r5);
        c[i + 5] = vec_type::template hadd<T>(r6);
        c[i + 6] = vec_type::template hadd<T>(r7);
        c[i + 7] = vec_type::template hadd<T>(r8);

        // Remainder inner loop
        for (; k < n; ++k) {
            c[i + 0] += a(i + 0, k) * b(k);
            c[i + 1] += a(i + 1, k) * b(k);
            c[i + 2] += a(i + 2, k) * b(k);
            c[i + 3] += a(i + 3, k) * b(k);
            c[i + 4] += a(i + 4, k) * b(k);
            c[i + 5] += a(i + 5, k) * b(k);
            c[i + 6] += a(i + 6, k) * b(k);
            c[i + 7] += a(i + 7, k) * b(k);
        }
    }

    // 2-Unrolled outer loop
    for (; i + 1 < m; i += 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        size_t k = 0;

        // 4-Unrolled Vectorized inner loop
        for (; k + (vec_size * 4) - 1 < n; k += vec_size * 4) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;
            const auto k3 = k + 2 * vec_size;
            const auto k4 = k + 3 * vec_size;

            auto b1 = b.template loadu<vec_type>(k1);
            auto b2 = b.template loadu<vec_type>(k2);
            auto b3 = b.template loadu<vec_type>(k3);
            auto b4 = b.template loadu<vec_type>(k4);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k1), b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k1), b1));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k2), b2));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k2), b2));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k3), b3));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k3), b3));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k4), b4));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k4), b4));
        }

        // 2-Unrolled Vectorized inner loop
        for (; k + (vec_size * 2) - 1 < n; k += vec_size * 2) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;

            auto b1 = b.template loadu<vec_type>(k1);
            auto b2 = b.template loadu<vec_type>(k2);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k1), b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k1), b1));

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 0) * n + k2), b2));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a.template loadu<vec_type>((i + 1) * n + k2), b2));
        }

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto b1 = b.template loadu<vec_type>(k);

            auto a1 = a.template loadu<vec_type>((i + 0) * n + k);
            auto a2 = a.template loadu<vec_type>((i + 1) * n + k);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a2, b1));
        }

        c[i + 0] = vec_type::template hadd<T>(r1);
        c[i + 1] = vec_type::template hadd<T>(r2);

        // Remainder inner loop
        for (; k < n; ++k) {
            c[i + 0] += a(i + 0, k) * b(k);
            c[i + 1] += a(i + 1, k) * b(k);
        }
    }

    // Remainder loop
    for (; i < m; ++i) {
        auto r1 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k + vec_size - 1 < n; k += vec_size) {
            auto a1 = a.template loadu<vec_type>(i * n + k);
            auto b1 = b.template loadu<vec_type>(k);

            auto t1 = vec_type::template mul<Cx>(a1, b1);
            r1 = vec_type::add(r1, t1);
        }

        auto result = vec_type::template hadd<T>(r1);

        // Remainder inner loop
        for (; k < n; ++k) {
            result += a(i, k) * b(k);
        }

        c[i] = result;
    }
}

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if (etl::size(a) < gemv_small_threshold) {
        gemv_small_kernel<default_vec>(a, b, c);
    } else {
        gemv_large_kernel<default_vec>(a, b, c);
    }
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    const auto m = rows(a);
    const auto n = columns(a);

    c = 0;

    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            c(i) += a(i, k) * b(k);
        }
    }
}

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm_small_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(b);
    const auto n = columns(b);

    size_t j = 0;

    for (; j + vec_size * 8 - 1 < n; j += vec_size * 8) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
            auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
            auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
            auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);
            auto b5 = b.template loadu<vec_type>(k * n + j + 4 * vec_size);
            auto b6 = b.template loadu<vec_type>(k * n + j + 5 * vec_size);
            auto b7 = b.template loadu<vec_type>(k * n + j + 6 * vec_size);
            auto b8 = b.template loadu<vec_type>(k * n + j + 7 * vec_size);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a1, b5));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a1, b6));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a1, b7));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a1, b8));
        }

        c.template storeu<vec_type>(r1, j + 0 * vec_size);
        c.template storeu<vec_type>(r2, j + 1 * vec_size);
        c.template storeu<vec_type>(r3, j + 2 * vec_size);
        c.template storeu<vec_type>(r4, j + 3 * vec_size);
        c.template storeu<vec_type>(r5, j + 4 * vec_size);
        c.template storeu<vec_type>(r6, j + 5 * vec_size);
        c.template storeu<vec_type>(r7, j + 6 * vec_size);
        c.template storeu<vec_type>(r8, j + 7 * vec_size);
    }

    for (; j + vec_size * 4 - 1 < n; j += vec_size * 4) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
            auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
            auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
            auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
        }

        c.template storeu<vec_type>(r1, j + 0 * vec_size);
        c.template storeu<vec_type>(r2, j + 1 * vec_size);
        c.template storeu<vec_type>(r3, j + 2 * vec_size);
        c.template storeu<vec_type>(r4, j + 3 * vec_size);
    }

    for (; j + vec_size - 1 < n; j += vec_size) {
        auto r1 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
        }

        c.template storeu<vec_type>(r1, j);
    }

    for (; j < n; j++) {
        auto value = T();

        for (size_t k = 0; k < m; k++) {
            value += a(k) * b(k, j);
        }

        c[j] = 0;
    }
}

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm_large_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(b);
    const auto n = columns(b);

    const size_t n_block = (32 * 1024) / sizeof(T);
    const size_t m_block = n < n_block ? 8 : 4;

    c = 0;

    for (size_t block_j = 0; block_j < n; block_j += n_block) {
        for (size_t block_k = 0; block_k < m; block_k += m_block) {
            const size_t m_end = std::min(block_k + m_block, m);
            const size_t n_end = std::min(block_j + n_block, n) & size_t(-vec_size);

            size_t j = block_j;

            // 8-Unrolled Vectorized loop
            for (; j + vec_size * 8 - 1 < n_end; j += vec_size * 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(a[k]);

                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
                    auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
                    auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);
                    auto b5 = b.template loadu<vec_type>(k * n + j + 4 * vec_size);
                    auto b6 = b.template loadu<vec_type>(k * n + j + 5 * vec_size);
                    auto b7 = b.template loadu<vec_type>(k * n + j + 6 * vec_size);
                    auto b8 = b.template loadu<vec_type>(k * n + j + 7 * vec_size);

                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                    r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                    r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                    r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                    r5 = vec_type::add(r5, vec_type::template mul<Cx>(a1, b5));
                    r6 = vec_type::add(r6, vec_type::template mul<Cx>(a1, b6));
                    r7 = vec_type::add(r7, vec_type::template mul<Cx>(a1, b7));
                    r8 = vec_type::add(r8, vec_type::template mul<Cx>(a1, b8));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 1 * vec_size), r2), j + 1 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 2 * vec_size), r3), j + 2 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 3 * vec_size), r4), j + 3 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 4 * vec_size), r5), j + 4 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 5 * vec_size), r6), j + 5 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 6 * vec_size), r7), j + 6 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 7 * vec_size), r8), j + 7 * vec_size);
            }

            // 4-Unrolled vectorized loop
            for (; j + vec_size * 4 - 1 < n_end; j += vec_size * 4) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(a[k]);

                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
                    auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
                    auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);

                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                    r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                    r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                    r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 1 * vec_size), r2), j + 1 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 2 * vec_size), r3), j + 2 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 3 * vec_size), r4), j + 3 * vec_size);
            }

            // Base vectorized loop
            for (; j + vec_size - 1 < n_end; j += vec_size) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(a[k]);
                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
            }

            // Remainder non-vectorized loop
            for (; j < n_end; ++j) {
                auto r1 = T();

                for (size_t k = block_k; k < m_end; ++k) {
                    r1 += a[k] * b(k, j);
                }

                c[j] += r1;
            }
        }
    }
}

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if(etl::size(b) < gevm_small_threshold){
        gevm_small_kernel<default_vec>(a, b, c);
    } else {
        gevm_large_kernel<default_vec>(a, b, c);
    }
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gevm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    c = 0;

    for (std::size_t k = 0; k < etl::dim<0>(a); k++) {
        for (std::size_t j = 0; j < columns(b); j++) {
            c(j) += a(k) * b(k, j);
        }
    }
}

template <typename V, typename A, typename B, typename C>
void gemm_small_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const size_t M = etl::rows(a);
    const size_t N = etl::columns(b);
    const size_t K = etl::columns(a);

    size_t j = 0;

    for (; j + vec_size * 8 - 1 < N; j += vec_size * 8) {
        for (size_t i = 0; i < M; ++i) {
            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();
            auto r3 = vec_type::template zero<T>();
            auto r4 = vec_type::template zero<T>();
            auto r5 = vec_type::template zero<T>();
            auto r6 = vec_type::template zero<T>();
            auto r7 = vec_type::template zero<T>();
            auto r8 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a(i, k));

                auto b1 = b.template loadu(k * N + j + vec_size * 0);
                auto b2 = b.template loadu(k * N + j + vec_size * 1);
                auto b3 = b.template loadu(k * N + j + vec_size * 2);
                auto b4 = b.template loadu(k * N + j + vec_size * 3);
                auto b5 = b.template loadu(k * N + j + vec_size * 4);
                auto b6 = b.template loadu(k * N + j + vec_size * 5);
                auto b7 = b.template loadu(k * N + j + vec_size * 6);
                auto b8 = b.template loadu(k * N + j + vec_size * 7);

                r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                r5 = vec_type::add(r5, vec_type::template mul<Cx>(a1, b5));
                r6 = vec_type::add(r6, vec_type::template mul<Cx>(a1, b6));
                r7 = vec_type::add(r7, vec_type::template mul<Cx>(a1, b7));
                r8 = vec_type::add(r8, vec_type::template mul<Cx>(a1, b8));
            }

            c.template storeu<vec_type>(r1, i * N + j + 0 * vec_size);
            c.template storeu<vec_type>(r2, i * N + j + 1 * vec_size);
            c.template storeu<vec_type>(r3, i * N + j + 2 * vec_size);
            c.template storeu<vec_type>(r4, i * N + j + 3 * vec_size);
            c.template storeu<vec_type>(r5, i * N + j + 4 * vec_size);
            c.template storeu<vec_type>(r6, i * N + j + 5 * vec_size);
            c.template storeu<vec_type>(r7, i * N + j + 6 * vec_size);
            c.template storeu<vec_type>(r8, i * N + j + 7 * vec_size);
        }
    }

    for (; j + (4 * vec_size) - 1 < N; j += 4 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2){
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            auto r31 = vec_type::template zero<T>();
            auto r32 = vec_type::template zero<T>();

            auto r41 = vec_type::template zero<T>();
            auto r42 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j + 0 * vec_size);
                auto b2 = b.template loadu<vec_type>(k * N + j + 1 * vec_size);
                auto b3 = b.template loadu<vec_type>(k * N + j + 2 * vec_size);
                auto b4 = b.template loadu<vec_type>(k * N + j + 3 * vec_size);

                auto a1 = vec_type::set(a(i, k));
                auto a2 = vec_type::set(a(i + 1, k));

                r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                r12 = vec_type::add(r12, vec_type::template mul<Cx>(a2, b1));

                r21 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b2));
                r22 = vec_type::add(r22, vec_type::template mul<Cx>(a2, b2));

                r31 = vec_type::add(r31, vec_type::template mul<Cx>(a1, b3));
                r32 = vec_type::add(r32, vec_type::template mul<Cx>(a2, b3));

                r41 = vec_type::add(r41, vec_type::template mul<Cx>(a1, b4));
                r42 = vec_type::add(r42, vec_type::template mul<Cx>(a2, b4));
            }

            c.template storeu<vec_type>(r11, (i+0) * N + j + 0 * vec_size);
            c.template storeu<vec_type>(r12, (i+1) * N + j + 0 * vec_size);

            c.template storeu<vec_type>(r21, (i+0) * N + j + 1 * vec_size);
            c.template storeu<vec_type>(r22, (i+1) * N + j + 1 * vec_size);

            c.template storeu<vec_type>(r31, (i+0) * N + j + 2 * vec_size);
            c.template storeu<vec_type>(r32, (i+1) * N + j + 2 * vec_size);

            c.template storeu<vec_type>(r41, (i+0) * N + j + 3 * vec_size);
            c.template storeu<vec_type>(r42, (i+1) * N + j + 3 * vec_size);
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j + 0 * vec_size);
                auto b2 = b.template loadu<vec_type>(k * N + j + 1 * vec_size);
                auto b3 = b.template loadu<vec_type>(k * N + j + 2 * vec_size);
                auto b4 = b.template loadu<vec_type>(k * N + j + 3 * vec_size);

                auto a1 = vec_type::set(a(i, k));

                r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                r21 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b2));
                r31 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b3));
                r41 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b4));
            }

            c.template storeu<vec_type>(r11, i * N + j + 0 * vec_size);
            c.template storeu<vec_type>(r21, i * N + j + 1 * vec_size);
            c.template storeu<vec_type>(r31, i * N + j + 2 * vec_size);
            c.template storeu<vec_type>(r41, i * N + j + 3 * vec_size);
        }
    }

    for (; j + (2 * vec_size) - 1 < N; j += 2 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j + 0 * vec_size);
                auto b2 = b.template loadu<vec_type>(k * N + j + 1 * vec_size);

                auto a1 = vec_type::set(a(i, k));
                auto a2 = vec_type::set(a(i + 1, k));

                r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                r12 = vec_type::add(r12, vec_type::template mul<Cx>(a2, b1));

                r21 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b2));
                r22 = vec_type::add(r22, vec_type::template mul<Cx>(a2, b2));
            }

            c.template storeu<vec_type>(r11, (i+0) * N + j + 0 * vec_size);
            c.template storeu<vec_type>(r12, (i+1) * N + j + 0 * vec_size);

            c.template storeu<vec_type>(r21, (i+0) * N + j + 1 * vec_size);
            c.template storeu<vec_type>(r22, (i+1) * N + j + 1 * vec_size);
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j + 0 * vec_size);
                auto b2 = b.template loadu<vec_type>(k * N + j + 1 * vec_size);

                auto a1 = vec_type::set(a(i, k));

                r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                r21 = vec_type::add(r21, vec_type::template mul<Cx>(a1, b2));
            }

            c.template storeu<vec_type>(r11, i * N + j + 0 * vec_size);
            c.template storeu<vec_type>(r21, i * N + j + 1 * vec_size);
        }
    }

    for (; j + vec_size - 1 < N; j += vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j);

                auto a1 = vec_type::set(a(i, k));
                auto a2 = vec_type::set(a(i + 1, k));

                r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                r2 = vec_type::add(r2, vec_type::template mul<Cx>(a2, b1));
            }

            c.template storeu<vec_type>(r1, (i+0) * N + j);
            c.template storeu<vec_type>(r2, (i+1) * N + j);
        }

        if (i < M) {
            auto r1 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = b.template loadu<vec_type>(k * N + j);

                auto a1 = vec_type::set(a(i, k));

                r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            }

            c.template storeu<vec_type>(r1, i * N + j);
        }
    }

    for (; j < N; ++j) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r1 = T();
            auto r2 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a(i, k) * b(k, j);
                r2 += a(i + 1, k) * b(k, j);
            }

            c(i + 0, j) = r1;
            c(i + 1, j) = r2;
        }

        if (i < M) {
            auto r1 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a(i, k) * b(k, j);
            }

            c(i, j) = r1;
        }
    }
}

template <typename V, typename A, typename B, typename C>
void gemm_large_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const size_t M = etl::rows(a);
    const size_t N = etl::columns(b);
    const size_t K = etl::columns(a);

    const size_t n_block_size = 128;
    const size_t m_block_size = 64;
    const size_t k_block_size = 128;

    for (size_t block_j = 0; block_j < N; block_j += n_block_size) {
        const size_t j_end = std::min(block_j + n_block_size, N);

        for (size_t block_i = 0; block_i < M; block_i += m_block_size) {
            const size_t i_end = std::min(block_i + m_block_size, M);

            for (size_t i = block_i; i < i_end; ++i) {
                for (size_t j = block_j; j < j_end; ++j) {
                    c(i, j) = 0;
                }
            }

            for (size_t block_k = 0; block_k < K; block_k += k_block_size) {
                const size_t k_end = std::min(block_k + k_block_size, K);

                size_t j = block_j;

                for (; j + vec_size * 4 - 1 < j_end; j += vec_size * 4) {
                    const size_t j1 = j + vec_size * 1;
                    const size_t j2 = j + vec_size * 2;
                    const size_t j3 = j + vec_size * 3;

                    size_t i = block_i;

                    for (; i + 1 < i_end; i += 2) {
                        auto r11 = c.template loadu<vec_type>((i + 0) * N + j);
                        auto r12 = c.template loadu<vec_type>((i + 0) * N + j1);
                        auto r13 = c.template loadu<vec_type>((i + 0) * N + j2);
                        auto r14 = c.template loadu<vec_type>((i + 0) * N + j3);

                        auto r21 = c.template loadu<vec_type>((i + 1) * N + j);
                        auto r22 = c.template loadu<vec_type>((i + 1) * N + j1);
                        auto r23 = c.template loadu<vec_type>((i + 1) * N + j2);
                        auto r24 = c.template loadu<vec_type>((i + 1) * N + j3);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i + 0, k));
                            auto a2 = vec_type::set(a(i + 1, k));

                            auto b1 = b.template loadu(k * N + j);
                            auto b2 = b.template loadu(k * N + j1);
                            auto b3 = b.template loadu(k * N + j2);
                            auto b4 = b.template loadu(k * N + j3);

                            r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                            r12 = vec_type::add(r12, vec_type::template mul<Cx>(a1, b2));
                            r13 = vec_type::add(r13, vec_type::template mul<Cx>(a1, b3));
                            r14 = vec_type::add(r14, vec_type::template mul<Cx>(a1, b4));

                            r21 = vec_type::add(r21, vec_type::template mul<Cx>(a2, b1));
                            r22 = vec_type::add(r22, vec_type::template mul<Cx>(a2, b2));
                            r23 = vec_type::add(r23, vec_type::template mul<Cx>(a2, b3));
                            r24 = vec_type::add(r24, vec_type::template mul<Cx>(a2, b4));
                        }

                        c.template storeu<vec_type>(r11, (i + 0) * N + j);
                        c.template storeu<vec_type>(r12, (i + 0) * N + j1);
                        c.template storeu<vec_type>(r13, (i + 0) * N + j2);
                        c.template storeu<vec_type>(r14, (i + 0) * N + j3);
                        c.template storeu<vec_type>(r21, (i + 1) * N + j);
                        c.template storeu<vec_type>(r22, (i + 1) * N + j1);
                        c.template storeu<vec_type>(r23, (i + 1) * N + j2);
                        c.template storeu<vec_type>(r24, (i + 1) * N + j3);
                    }

                    if (i < i_end) {
                        auto r1 = c.template loadu<vec_type>(i * N + j);
                        auto r2 = c.template loadu<vec_type>(i * N + j1);
                        auto r3 = c.template loadu<vec_type>(i * N + j2);
                        auto r4 = c.template loadu<vec_type>(i * N + j3);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i, k));

                            auto b1 = b.template loadu(k * N + j);
                            auto b2 = b.template loadu(k * N + j1);
                            auto b3 = b.template loadu(k * N + j2);
                            auto b4 = b.template loadu(k * N + j3);

                            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                        }

                        c.template storeu<vec_type>(r1, i * N + j);
                        c.template storeu<vec_type>(r2, i * N + j1);
                        c.template storeu<vec_type>(r3, i * N + j2);
                        c.template storeu<vec_type>(r4, i * N + j3);
                    }
                }

                for (; j + vec_size * 2 - 1 < j_end; j += vec_size * 2) {
                    const size_t j1(j + vec_size);

                    size_t i = block_i;

                    for (; i + 3 < i_end; i += 4) {
                        auto r11 = c.template loadu<vec_type>((i + 0) * N + j);
                        auto r12 = c.template loadu<vec_type>((i + 0) * N + j1);

                        auto r21 = c.template loadu<vec_type>((i + 1) * N + j);
                        auto r22 = c.template loadu<vec_type>((i + 1) * N + j1);

                        auto r31 = c.template loadu<vec_type>((i + 2) * N + j);
                        auto r32 = c.template loadu<vec_type>((i + 2) * N + j1);

                        auto r41 = c.template loadu<vec_type>((i + 3) * N + j);
                        auto r42 = c.template loadu<vec_type>((i + 3) * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i + 0, k));
                            auto a2 = vec_type::set(a(i + 1, k));
                            auto a3 = vec_type::set(a(i + 2, k));
                            auto a4 = vec_type::set(a(i + 3, k));

                            auto b1 = b.template loadu(k * N + j);
                            auto b2 = b.template loadu(k * N + j1);

                            r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                            r12 = vec_type::add(r12, vec_type::template mul<Cx>(a1, b2));

                            r21 = vec_type::add(r21, vec_type::template mul<Cx>(a2, b1));
                            r22 = vec_type::add(r22, vec_type::template mul<Cx>(a2, b2));

                            r31 = vec_type::add(r31, vec_type::template mul<Cx>(a3, b1));
                            r32 = vec_type::add(r32, vec_type::template mul<Cx>(a3, b2));

                            r41 = vec_type::add(r41, vec_type::template mul<Cx>(a4, b1));
                            r42 = vec_type::add(r42, vec_type::template mul<Cx>(a4, b2));
                        }

                        c.template storeu<vec_type>(r11, (i + 0) * N + j);
                        c.template storeu<vec_type>(r12, (i + 0) * N + j1);
                        c.template storeu<vec_type>(r21, (i + 1) * N + j);
                        c.template storeu<vec_type>(r22, (i + 1) * N + j1);
                        c.template storeu<vec_type>(r31, (i + 2) * N + j);
                        c.template storeu<vec_type>(r32, (i + 2) * N + j1);
                        c.template storeu<vec_type>(r41, (i + 3) * N + j);
                        c.template storeu<vec_type>(r42, (i + 3) * N + j1);
                    }

                    for (; i + 2 - 1 < i_end; i += 2) {
                        auto r11 = c.template loadu<vec_type>((i + 0) * N + j);
                        auto r12 = c.template loadu<vec_type>((i + 0) * N + j1);

                        auto r21 = c.template loadu<vec_type>((i + 1) * N + j);
                        auto r22 = c.template loadu<vec_type>((i + 1) * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i + 0, k));
                            auto a2 = vec_type::set(a(i + 1, k));

                            auto b1 = b.template loadu(k * N + j);
                            auto b2 = b.template loadu(k * N + j1);

                            r11 = vec_type::add(r11, vec_type::template mul<Cx>(a1, b1));
                            r12 = vec_type::add(r12, vec_type::template mul<Cx>(a1, b2));

                            r21 = vec_type::add(r21, vec_type::template mul<Cx>(a2, b1));
                            r22 = vec_type::add(r22, vec_type::template mul<Cx>(a2, b2));
                        }

                        c.template storeu<vec_type>(r11, (i + 0) * N + j);
                        c.template storeu<vec_type>(r12, (i + 0) * N + j1);
                        c.template storeu<vec_type>(r21, (i + 1) * N + j);
                        c.template storeu<vec_type>(r22, (i + 1) * N + j1);
                    }

                    if (i < i_end) {
                        auto r1 = c.template loadu<vec_type>(i * N + j);
                        auto r2 = c.template loadu<vec_type>(i * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i, k));

                            auto b1 = b.template loadu(k * N + j);
                            auto b2 = b.template loadu(k * N + j1);

                            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                        }

                        c.template storeu<vec_type>(r1, i * N + j);
                        c.template storeu<vec_type>(r2, i * N + j1);
                    }
                }

                for (; j + vec_size - 1 < j_end; j += vec_size) {
                    for (size_t i = block_i; i < i_end; ++i) {
                        auto r1 = c.template loadu<vec_type>(i * N + j);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a(i, k));
                            auto b1 = b.template loadu(k * N + j);
                            r1      = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                        }

                        c.template storeu<vec_type>(r1, i * N + j);
                    }
                }

                for (; j < j_end; ++j) {
                    for (size_t i = block_i; i < i_end; ++i) {
                        auto value = c(i, j);

                        for (size_t k = block_k; k < k_end; ++k) {
                            value += a(i, k) * b(k, j);
                        }

                        c(i, j) = value;
                    }
                }
            }
        }
    }
}

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if(etl::size(b) < 10000){
        gemm_small_kernel<default_vec>(a, b, c);
    } else {
        gemm_large_kernel<default_vec>(a, b, c);
    }
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    c = 0;

    for (std::size_t i = 0; i < rows(a); i++) {
        for (std::size_t k = 0; k < columns(a); k++) {
            for (std::size_t j = 0; j < columns(b); j++) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
}

} //end of namespace vec

} //end of namespace impl

} //end of namespace etl
