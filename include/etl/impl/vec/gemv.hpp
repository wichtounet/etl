//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// The idea of the GEMM kernels is largely inspired by the kernels in Blaze by
// Klaus Igleberg

namespace etl::impl::vec {

/*!
 * \brief Optimized version of small GEMV for row major version
 * \param aa The lhs matrix
 * \param bb The rhs vector
 * \param cc The result vector
 */
template <typename V, bool Padded, typename T>
void gemv_small_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    cpp_assert(aa, "Invalid memory in entry to gemv_small_kernel_rr");
    cpp_assert(bb, "Invalid memory in entry to gemv_small_kernel_rr");
    cpp_assert(cc, "Invalid memory in entry to gemv_small_kernel_rr");

    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    static constexpr bool remainder = !advanced_padding || !Padded;
    const size_t last               = remainder ? (n & size_t(-vec_size)) : n;

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
        for (; k < last; k += vec_size) {
            auto b1 = vec_type::loadu(bb + k);

            auto a1 = vec_type::loadu(aa + (i + 0) * n + k);
            auto a2 = vec_type::loadu(aa + (i + 1) * n + k);
            auto a3 = vec_type::loadu(aa + (i + 2) * n + k);
            auto a4 = vec_type::loadu(aa + (i + 3) * n + k);
            auto a5 = vec_type::loadu(aa + (i + 4) * n + k);
            auto a6 = vec_type::loadu(aa + (i + 5) * n + k);
            auto a7 = vec_type::loadu(aa + (i + 6) * n + k);
            auto a8 = vec_type::loadu(aa + (i + 7) * n + k);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a2, b1, r2);
            r3 = vec_type::fmadd(a3, b1, r3);
            r4 = vec_type::fmadd(a4, b1, r4);
            r5 = vec_type::fmadd(a5, b1, r5);
            r6 = vec_type::fmadd(a6, b1, r6);
            r7 = vec_type::fmadd(a7, b1, r7);
            r8 = vec_type::fmadd(a8, b1, r8);
        }

        cc[i + 0] = vec_type::hadd(r1);
        cc[i + 1] = vec_type::hadd(r2);
        cc[i + 2] = vec_type::hadd(r3);
        cc[i + 3] = vec_type::hadd(r4);
        cc[i + 4] = vec_type::hadd(r5);
        cc[i + 5] = vec_type::hadd(r6);
        cc[i + 6] = vec_type::hadd(r7);
        cc[i + 7] = vec_type::hadd(r8);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            cc[i + 0] += aa[(i + 0) * n + k] * bb[k];
            cc[i + 1] += aa[(i + 1) * n + k] * bb[k];
            cc[i + 2] += aa[(i + 2) * n + k] * bb[k];
            cc[i + 3] += aa[(i + 3) * n + k] * bb[k];
            cc[i + 4] += aa[(i + 4) * n + k] * bb[k];
            cc[i + 5] += aa[(i + 5) * n + k] * bb[k];
            cc[i + 6] += aa[(i + 6) * n + k] * bb[k];
            cc[i + 7] += aa[(i + 7) * n + k] * bb[k];
        }
    }

    // 2-Unrolled outer loop
    for (; i + 1 < m; i += 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k < last; k += vec_size) {
            auto b1 = vec_type::loadu(bb + k);

            auto a1 = vec_type::loadu(aa + (i + 0) * n + k);
            auto a2 = vec_type::loadu(aa + (i + 1) * n + k);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a2, b1, r2);
        }

        cc[i + 0] = vec_type::hadd(r1);
        cc[i + 1] = vec_type::hadd(r2);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            cc[i + 0] += aa[(i + 0) * n + k] * bb[k];
            cc[i + 1] += aa[(i + 1) * n + k] * bb[k];
        }
    }

    // Remainder loop
    if (i < m) {
        auto r1 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k < last; k += vec_size) {
            auto b1 = vec_type::loadu(bb + k);
            auto a1 = vec_type::loadu(aa + (i + 0) * n + k);
            r1      = vec_type::fmadd(a1, b1, r1);
        }

        auto result = vec_type::hadd(r1);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            result += aa[i * n + k] * bb[k];
        }

        cc[i] = result;
    }
}

/*!
 * \brief Optimized version of large GEMV for row major version
 * \param aa The lhs matrix
 * \param bb The rhs vector
 * \param cc The result vector
 */
template <typename V, bool Padded, typename T>
void gemv_large_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    static constexpr bool remainder = !advanced_padding || !Padded;
    const size_t last               = remainder ? (n & size_t(-vec_size)) : n;

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
        for (; k + (vec_size * 3) < last; k += vec_size * 4) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;
            const auto k3 = k + 2 * vec_size;
            const auto k4 = k + 3 * vec_size;

            auto b1 = vec_type::loadu(bb + k1);
            auto b2 = vec_type::loadu(bb + k2);
            auto b3 = vec_type::loadu(bb + k3);
            auto b4 = vec_type::loadu(bb + k4);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k1), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k1), b1, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k1), b1, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k1), b1, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k1), b1, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k1), b1, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k1), b1, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k1), b1, r8);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k2), b2, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k2), b2, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k2), b2, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k2), b2, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k2), b2, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k2), b2, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k2), b2, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k2), b2, r8);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k3), b3, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k3), b3, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k3), b3, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k3), b3, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k3), b3, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k3), b3, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k3), b3, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k3), b3, r8);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k4), b4, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k4), b4, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k4), b4, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k4), b4, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k4), b4, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k4), b4, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k4), b4, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k4), b4, r8);
        }

        // 2-Unrolled Vectorized inner loop
        for (; k + (vec_size) < last; k += vec_size * 2) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;

            auto b1 = vec_type::loadu(bb + k1);
            auto b2 = vec_type::loadu(bb + k2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k1), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k1), b1, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k1), b1, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k1), b1, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k1), b1, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k1), b1, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k1), b1, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k1), b1, r8);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k2), b2, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k2), b2, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k2), b2, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k2), b2, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k2), b2, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k2), b2, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k2), b2, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k2), b2, r8);
        }

        // Vectorized inner loop
        if (k < last) {
            auto b1 = vec_type::loadu(bb + k);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k), b1, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2) * n + k), b1, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3) * n + k), b1, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4) * n + k), b1, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5) * n + k), b1, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6) * n + k), b1, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7) * n + k), b1, r8);

            k += vec_size;
        }

        cc[i + 0] = vec_type::hadd(r1);
        cc[i + 1] = vec_type::hadd(r2);
        cc[i + 2] = vec_type::hadd(r3);
        cc[i + 3] = vec_type::hadd(r4);
        cc[i + 4] = vec_type::hadd(r5);
        cc[i + 5] = vec_type::hadd(r6);
        cc[i + 6] = vec_type::hadd(r7);
        cc[i + 7] = vec_type::hadd(r8);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            cc[i + 0] += aa[(i + 0) * n + k] * bb[k];
            cc[i + 1] += aa[(i + 1) * n + k] * bb[k];
            cc[i + 2] += aa[(i + 2) * n + k] * bb[k];
            cc[i + 3] += aa[(i + 3) * n + k] * bb[k];
            cc[i + 4] += aa[(i + 4) * n + k] * bb[k];
            cc[i + 5] += aa[(i + 5) * n + k] * bb[k];
            cc[i + 6] += aa[(i + 6) * n + k] * bb[k];
            cc[i + 7] += aa[(i + 7) * n + k] * bb[k];
        }
    }

    // 2-Unrolled outer loop
    for (; i + 1 < m; i += 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        size_t k = 0;

        // 4-Unrolled Vectorized inner loop
        for (; k + (vec_size * 3) < last; k += vec_size * 4) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;
            const auto k3 = k + 2 * vec_size;
            const auto k4 = k + 3 * vec_size;

            auto b1 = vec_type::loadu(bb + k1);
            auto b2 = vec_type::loadu(bb + k2);
            auto b3 = vec_type::loadu(bb + k3);
            auto b4 = vec_type::loadu(bb + k4);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k1), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k1), b1, r2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k2), b2, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k2), b2, r2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k3), b3, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k3), b3, r2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k4), b4, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k4), b4, r2);
        }

        // 2-Unrolled Vectorized inner loop
        for (; k + vec_size < last; k += vec_size * 2) {
            const auto k1 = k + 0 * vec_size;
            const auto k2 = k + 1 * vec_size;

            auto b1 = vec_type::loadu(bb + k1);
            auto b2 = vec_type::loadu(bb + k2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k1), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k1), b1, r2);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k2), b2, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k2), b2, r2);
        }

        // Vectorized inner loop
        if (k < last) {
            auto b1 = vec_type::loadu(bb + k);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1) * n + k), b1, r2);

            k += vec_size;
        }

        cc[i + 0] = vec_type::hadd(r1);
        cc[i + 1] = vec_type::hadd(r2);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            cc[i + 0] += aa[i + 0 * n + k] * bb[k];
            cc[i + 1] += aa[i + 1 * n + k] * bb[k];
        }
    }

    // Remainder loop
    if (i < m) {
        auto r1 = vec_type::template zero<T>();

        size_t k = 0;

        // Vectorized inner loop
        for (; k < last; k += vec_size) {
            auto b1 = vec_type::loadu(bb + k);
            r1      = vec_type::fmadd(vec_type::loadu(aa + (i + 0) * n + k), b1, r1);
        }

        auto result = vec_type::hadd(r1);

        // Remainder inner loop
        for (; remainder && k < n; ++k) {
            result += aa[i * n + k] * bb[k];
        }

        cc[i] = result;
    }
}

/*!
 * \brief Optimized version of small GEMV for column major version
 * \param aa The lhs matrix
 * \param bb The rhs vector
 * \param cc The result vector
 */
template <typename V, bool Padded, typename T>
void gemv_small_kernel_cc(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    static constexpr bool remainder = !advanced_padding || !Padded;
    const size_t last               = remainder ? (m & size_t(-vec_size)) : m;

    size_t i = 0;

#ifdef ETL_GEMV_SMALL_CC_8
    // Vectorized loop, unrolled 8x
    for (; i + 8 * vec_size - 1 < last; i += 8 * vec_size) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        for (size_t j = 0; j < n; ++j) {
            auto b1 = vec_type::set(bb[j]);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0 * vec_size) + j * m), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1 * vec_size) + j * m), b1, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2 * vec_size) + j * m), b1, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3 * vec_size) + j * m), b1, r4);
            r5 = vec_type::fmadd(vec_type::loadu(aa + (i + 4 * vec_size) + j * m), b1, r5);
            r6 = vec_type::fmadd(vec_type::loadu(aa + (i + 5 * vec_size) + j * m), b1, r6);
            r7 = vec_type::fmadd(vec_type::loadu(aa + (i + 6 * vec_size) + j * m), b1, r7);
            r8 = vec_type::fmadd(vec_type::loadu(aa + (i + 7 * vec_size) + j * m), b1, r8);
        }

        vec_type::storeu(cc + i + 0 * vec_size, r1);
        vec_type::storeu(cc + i + 1 * vec_size, r2);
        vec_type::storeu(cc + i + 2 * vec_size, r3);
        vec_type::storeu(cc + i + 3 * vec_size, r4);
        vec_type::storeu(cc + i + 4 * vec_size, r5);
        vec_type::storeu(cc + i + 5 * vec_size, r6);
        vec_type::storeu(cc + i + 6 * vec_size, r7);
        vec_type::storeu(cc + i + 7 * vec_size, r8);
    }
#endif

    // Vectorized loop, unrolled 4x
    for (; i + 4 * vec_size - 1 < last; i += 4 * vec_size) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();

        for (size_t j = 0; j < n; ++j) {
            auto b1 = vec_type::set(bb[j]);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0 * vec_size) + j * m), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1 * vec_size) + j * m), b1, r2);
            r3 = vec_type::fmadd(vec_type::loadu(aa + (i + 2 * vec_size) + j * m), b1, r3);
            r4 = vec_type::fmadd(vec_type::loadu(aa + (i + 3 * vec_size) + j * m), b1, r4);
        }

        vec_type::storeu(cc + i + 0 * vec_size, r1);
        vec_type::storeu(cc + i + 1 * vec_size, r2);
        vec_type::storeu(cc + i + 2 * vec_size, r3);
        vec_type::storeu(cc + i + 3 * vec_size, r4);
    }

    // Vectorized loop, unrolled 2x
    for (; i + 2 * vec_size - 1 < last; i += 2 * vec_size) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        for (size_t j = 0; j < n; ++j) {
            auto b1 = vec_type::set(bb[j]);

            r1 = vec_type::fmadd(vec_type::loadu(aa + (i + 0 * vec_size) + j * m), b1, r1);
            r2 = vec_type::fmadd(vec_type::loadu(aa + (i + 1 * vec_size) + j * m), b1, r2);
        }

        vec_type::storeu(cc + i + 0 * vec_size, r1);
        vec_type::storeu(cc + i + 1 * vec_size, r2);
    }

    // Vectorized loop, not unrolled
    for (; i + vec_size - 1 < last; i += vec_size) {
        auto r1 = vec_type::template zero<T>();

        for (size_t j = 0; j < n; ++j) {
            auto b1 = vec_type::set(bb[j]);
            r1      = vec_type::fmadd(vec_type::loadu(aa + i + j * m), b1, r1);
        }

        vec_type::storeu(cc + i, r1);
    }

    // Normal loop
    for (; remainder && i < m; ++i) {
        T value(0);

        for (size_t j = 0; j < n; ++j) {
            value += aa[i + j * m] * bb[j];
        }

        cc[i] = value;
    }
}

/*!
 * \brief Optimized version of large GEMV for column major version
 * \param aa The lhs matrix
 * \param bb The rhs vector
 * \param cc The result vector
 */
template <typename V, bool Padded, typename T>
void gemv_large_kernel_cc(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t m_block = (32 * 1024) / sizeof(T);
    const size_t n_block = (n < m_block) ? 8 : 4;

    for (size_t block_i = 0U; block_i < m; block_i += m_block) {
        for (size_t block_j = 0UL; block_j < n; block_j += n_block) {
            const size_t m_end = std::min(block_i + m_block, m);
            const size_t n_end = std::min(block_j + n_block, n);

            size_t i = block_i;

#ifdef ETL_GEMV_LARGE_CC_4
            // Vectorized loop, unrolled 4x
            for (; i + 4 * vec_size - 1 < m_end; i += 4 * vec_size) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                for (size_t j = block_j; j < n_end; ++j) {
                    auto b1 = vec_type::set(bb[j]);

                    r1 = vec_type::fmadd(vec_type::loadu(aa + i + 0 * vec_size + j * m), b1, r1);
                    r2 = vec_type::fmadd(vec_type::loadu(aa + i + 1 * vec_size + j * m), b1, r2);
                    r3 = vec_type::fmadd(vec_type::loadu(aa + i + 2 * vec_size + j * m), b1, r3);
                    r4 = vec_type::fmadd(vec_type::loadu(aa + i + 3 * vec_size + j * m), b1, r4);
                }

                vec_type::storeu(cc + i + 0 * vec_size, vec_type::add(vec_type::loadu(cc + i + 0 * vec_size), r1));
                vec_type::storeu(cc + i + 1 * vec_size, vec_type::add(vec_type::loadu(cc + i + 1 * vec_size), r2));
                vec_type::storeu(cc + i + 2 * vec_size, vec_type::add(vec_type::loadu(cc + i + 2 * vec_size), r3));
                vec_type::storeu(cc + i + 3 * vec_size, vec_type::add(vec_type::loadu(cc + i + 3 * vec_size), r4));
            }
#endif

            // Vectorized loop, unrolled 2x
            for (; i + 2 * vec_size - 1 < m_end; i += 2 * vec_size) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t j = block_j; j < n_end; ++j) {
                    auto b1 = vec_type::set(bb[j]);

                    r1 = vec_type::fmadd(vec_type::loadu(aa + i + 0 * vec_size + j * m), b1, r1);
                    r2 = vec_type::fmadd(vec_type::loadu(aa + i + 1 * vec_size + j * m), b1, r2);
                }

                vec_type::storeu(cc + i + 0 * vec_size, vec_type::add(vec_type::loadu(cc + i + 0 * vec_size), r1));
                vec_type::storeu(cc + i + 1 * vec_size, vec_type::add(vec_type::loadu(cc + i + 1 * vec_size), r2));
            }

            // Vectorized loop
            for (; i + vec_size - 1 < m_end; i += vec_size) {
                auto r1 = vec_type::template zero<T>();

                for (size_t j = block_j; j < n_end; ++j) {
                    auto b1 = vec_type::set(bb[j]);
                    r1      = vec_type::fmadd(vec_type::loadu(aa + i + j * m), b1, r1);
                }

                vec_type::storeu(cc + i, vec_type::add(vec_type::loadu(cc + i), r1));
            }

            // Remainder loop
            for (; i < m_end; ++i) {
                T value(0);

                for (size_t j = block_j; j < n_end; ++j) {
                    value += aa[i + j * m] * bb[j];
                }

                cc[i] += value;
            }
        }
    }
}

/*!
 * \brief Optimized version of GEMV for column major version
 * \param a The lhs matrix
 * \param b The rhs vector
 * \param c The result vector
 */
template <typename A, typename B, typename C>
void gemv(A&& a, B&& b, C&& c) {
    if constexpr (vec_enabled && vectorize_impl && all_homogeneous<A, B, C> && all_vectorizable<vector_mode, A, B, C>) {
        cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        const auto m = rows(a);
        const auto n = columns(a);

        if constexpr (is_row_major<A>) {
            if (etl::size(a) < gemv_rm_small_threshold) {
                gemv_small_kernel_rr<default_vec, all_padded<A, B, C>>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            } else {
                gemv_large_kernel_rr<default_vec, all_padded<A, B, C>>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            }
        } else {
            if (etl::size(a) < gemv_cm_small_threshold) {
                gemv_small_kernel_cc<default_vec, all_padded<A, B, C>>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            } else {
                c = 0;
                gemv_large_kernel_cc<default_vec, all_padded<A, B, C>>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            }
        }

        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid operation called vec::gemv with heterogeneous types");
    }
}

// Versions with transpose

/*!
 * \brief Optimized version of GEMV for column major version
 * \param a The lhs matrix
 * \param b The rhs vector
 * \param c The result vector
 */
template <typename A, typename B, typename C>
void gemv_t(A&& a, B&& b, C&& c) {
    if constexpr (vec_enabled && vectorize_impl && all_homogeneous<A, B, C> && all_vectorizable<vector_mode, A, B, C>) {
        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        const auto m = rows(a);
        const auto n = columns(a);

        if constexpr (is_row_major<A>) {
            if (etl::size(a) < gemv_rm_small_threshold) {
                gemv_small_kernel_cc<default_vec, all_padded<A, B, C>>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            } else {
                gemv_large_kernel_cc<default_vec, all_padded<A, B, C>>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            }
        } else {
            if (etl::size(a) < gemv_cm_small_threshold) {
                gemv_small_kernel_rr<default_vec, all_padded<A, B, C>>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            } else {
                c = 0;
                gemv_large_kernel_rr<default_vec, all_padded<A, B, C>>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            }
        }
        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid operation called vec::gemv with heterogeneous types");
    }
}

} //end of namespace etl::impl::vec
