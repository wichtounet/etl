//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * This is a tentative implementation of a BLIS Like kernel
 * However, this is current slower than the other kernels
 * and it is also only "optimized" for float
 * so, it is current disabled.
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief BLIS-like GEMM config
 */
template <typename T>
struct gemm_config;

/*!
 * \brief BLIS-like GEMM config for single-precision
 */
template <>
struct gemm_config<float> {
    static constexpr size_t MC = 768;  ///< The first dimension buffer
    static constexpr size_t KC = 384;  ///< The second dimension buffer
    static constexpr size_t NC = 4096; ///< The third dimension buffer

    static constexpr size_t MR = 8; ///< The first dimension of micro-kernel
    static constexpr size_t NR = 4; ///< The second dimension of micro-kernel
};

/*!
 * \brief BLIS-like GEMM config for double-precision
 */
template <>
struct gemm_config<double> {
    static constexpr size_t MC = 384;  ///< The first dimension buffer
    static constexpr size_t KC = 384;  ///< The second dimension buffer
    static constexpr size_t NC = 4096; ///< The third dimension buffer

    static constexpr size_t MR = 4; ///< The first dimension of micro-kernel
    static constexpr size_t NR = 4; ///< The second dimension of micro-kernel
};

/*!
 * \brief Packing panels of A, with padding if required.
 */
template <typename T>
void pack_a(size_t mc, size_t kc, const T* A, size_t incRowA, size_t incColA, T* _A) {
    static constexpr const size_t MR = gemm_config<T>::MR;

    const size_t mp  = mc / MR;
    const size_t _mr = mc % MR;

    for (size_t k = 0; k < mp; ++k) {
        for (size_t j = 0; j < kc; ++j) {
            for (size_t i = 0; i < MR; ++i) {
                _A[k * kc * MR + j * MR + i] = A[k * MR * incRowA + j * incColA + i * incRowA];
            }
        }
    }

    if (_mr > 0) {
        const size_t k = mp;

        for (size_t j = 0; j < kc; ++j) {
            for (size_t i = 0; i < _mr; ++i) {
                _A[k * kc * MR + j * MR + i] = A[k * MR * incRowA + j * incColA + i * incRowA];
            }

            for (size_t i = _mr; i < MR; ++i) {
                _A[k * kc * MR + j * MR + i] = 0.0;
            }
        }
    }
}

/*!
 * \brief Packing panels of B, with padding if required.
 */
template <typename T>
void pack_b(size_t kc, size_t nc, const T* B, size_t incRowB, size_t incColB, T* _B) {
    static constexpr const size_t NR = gemm_config<T>::NR;

    const size_t np  = nc / NR;
    const size_t _nr = nc % NR;

    for (size_t k = 0; k < np; ++k) {
        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < NR; ++j) {
                _B[k * kc * NR + i * NR + j] = B[k * NR * incColB + i * incRowB + j * incColB];
            }
        }
    }

    if (_nr > 0) {
        const size_t k = np;

        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < _nr; ++j) {
                _B[k * kc * NR + i * NR + j] = B[k * NR * incColB + i * incRowB + j * incColB];
            }

            for (size_t j = _nr; j < NR; ++j) {
                _B[k * kc * NR + i * NR + j] = 0.0;
            }
        }
    }
}

/*!
 * \brief Compute Y += alpha*X
 */
template <typename T>
void dgeaxpy(size_t m, size_t n, T alpha, const T* X, size_t incRowX, size_t incColX, T* Y, size_t incRowY, size_t incColY) {
    if (alpha != 1.0) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                Y[i * incRowY + j * incColY] += alpha * X[i * incRowX + j * incColX];
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                Y[i * incRowY + j * incColY] += X[i * incRowX + j * incColX];
            }
        }
    }
}

/*!
 * \brief Scale X by alpha
 */
template <typename T>
void dgescal(size_t m, size_t n, T alpha, T* X, size_t incRowX, size_t incColX) {
    if (alpha != 0.0) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                X[i * incRowX + j * incColX] *= alpha;
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                X[i * incRowX + j * incColX] = 0.0;
            }
        }
    }
}

/*!
 * \brief Optimized pico kernel for BLIS, for avx and float
 */
template <typename V, typename T>
void gemm_pico_kernel(size_t kc, const T* ETL_RESTRICT A, const T* ETL_RESTRICT B, T* ETL_RESTRICT AB) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    if constexpr (std::is_same_v<float, T> && vector_mode == vector_mode_t::AVX) {
        using vec_type = V;

        static constexpr size_t vec_size = vec_type::template traits<T>::size;

        static_assert(NR == 4, "Invalid algorithm selection");
        static_assert(vec_size == MR, "Invalid algorith selection");

        auto AB1 = vec_type::template zero<T>();
        auto AB2 = vec_type::template zero<T>();
        auto AB3 = vec_type::template zero<T>();
        auto AB4 = vec_type::template zero<T>();

        size_t l = 0;

        for (; l + 3 < kc; l += 4) {
            auto A1 = vec_type::loadu(A + (l + 0) * MR);

            AB1 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 3]), AB4);

            auto A2 = vec_type::loadu(A + (l + 1) * MR);

            AB1 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 3]), AB4);

            auto A3 = vec_type::loadu(A + (l + 2) * MR);

            AB1 = vec_type::fmadd(A3, vec_type::set(B[(l + 2) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A3, vec_type::set(B[(l + 2) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A3, vec_type::set(B[(l + 2) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A3, vec_type::set(B[(l + 2) * NR + 3]), AB4);

            auto A4 = vec_type::loadu(A + (l + 3) * MR);

            AB1 = vec_type::fmadd(A4, vec_type::set(B[(l + 3) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A4, vec_type::set(B[(l + 3) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A4, vec_type::set(B[(l + 3) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A4, vec_type::set(B[(l + 3) * NR + 3]), AB4);
        }

        for (; l + 1 < kc; l += 2) {
            auto A1 = vec_type::loadu(A + (l + 0) * MR);

            AB1 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 3]), AB4);

            auto A2 = vec_type::loadu(A + (l + 1) * MR);

            AB1 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A2, vec_type::set(B[(l + 1) * NR + 3]), AB4);
        }

        if (l < kc) {
            auto A1 = vec_type::loadu(A + (l + 0) * MR);

            AB1 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 0]), AB1);
            AB2 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 1]), AB2);
            AB3 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 2]), AB3);
            AB4 = vec_type::fmadd(A1, vec_type::set(B[(l + 0) * NR + 3]), AB4);
        }

        vec_type::storeu(AB + 0 * MR, AB1);
        vec_type::storeu(AB + 1 * MR, AB2);
        vec_type::storeu(AB + 2 * MR, AB3);
        vec_type::storeu(AB + 3 * MR, AB4);
    } else {
        for (size_t l = 0; l < MR * NR; ++l) {
            AB[l] = 0;
        }

        for (size_t l = 0; l < kc; ++l) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    AB[j * MR + i] += A[l * MR + i] * B[l * NR + j];
                }
            }
        }
    }
}

/*!
 * \brief Micro kernel for BLIS
 */
template <typename V, typename T>
void gemm_micro_kernel(size_t kc, T alpha, const T* A, const T* B, T beta, T* C, size_t incRowC, size_t incColC) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    #ifdef __ARM_ARCH
    alignas(16) T AB[MR * NR];
    #else
    alignas(64) T AB[MR * NR];
    #endif


    // Bottleneck kernel

    gemm_pico_kernel<V>(kc, A, B, AB);

    if (alpha == T(1.0)) {
        if (beta == T(0.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = AB[i + j * MR];
                }
            }
        } else if (beta != T(1.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = beta * C[i * incRowC + j * incColC] + AB[i + j * MR];
                }
            }
        }
    } else {
        if (beta == T(0.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = alpha * AB[i + j * MR];
                }
            }
        } else if (beta != T(1.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = beta * C[i * incRowC + j * incColC] + alpha * AB[i + j * MR];
                }
            }
        }
    }
}

/*!
 * \brief Macro kernel for the BLIS version of the kernels. Assuming that they
 * are already packed in _A and _B
 */
template <typename V, typename T>
void gemm_macro_kernel(size_t mc,
                       size_t nc,
                       size_t kc,
                       T alpha,
                       T beta,
                       T* C,
                       size_t incRowC,
                       size_t incColC,
                       etl::dyn_matrix<T, 2>& _A,
                       etl::dyn_matrix<T, 2>& _B,
                       etl::dyn_matrix<T, 2>& _C) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    const size_t mp = (mc + MR - 1) / MR;
    const size_t np = (nc + NR - 1) / NR;

    const size_t _mr = mc % MR;
    const size_t _nr = nc % NR;

    for (size_t j = 0; j < np; ++j) {
        size_t nr = (j != np - 1 || _nr == 0) ? NR : _nr;

        for (size_t i = 0; i < mp; ++i) {
            size_t mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

            if (mr == MR && nr == NR) {
                gemm_micro_kernel<V>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR], beta, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
            } else {
                gemm_micro_kernel<V>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR], T(0.0), _C.memory_start(), 1, MR);
                dgescal(mr, nr, beta, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
                dgeaxpy(mr, nr, T(1.0), _C.memory_start(), 1, MR, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
            }
        }
    }
}

/*!
 * \brief Optimized version of large GEMM for row major version with workspace
 * on the form of the BLIS kernels.
 *
 * On very large matrices, this is somewhat faster but seems to depend on the
 * processor. This will need more work and complete kernels.
 *
 * From: http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
 *
 * \param A The lhs matrix
 * \param B The rhs matrix
 * \param C The result matrix
 * \param beta The multipliying of the previous value
 */
template <typename V, typename T>
void gemm_large_kernel_workspace_rr(const T* A, const T* B, T* C, size_t m, size_t n, size_t k, T beta) {
    if constexpr (is_floating_t<T>) {
        static constexpr const size_t MC = gemm_config<T>::MC;
        static constexpr const size_t KC = gemm_config<T>::KC;
        static constexpr const size_t NC = gemm_config<T>::NC;

        static constexpr const size_t MR = gemm_config<T>::MR;
        static constexpr const size_t NR = gemm_config<T>::NR;

        etl::dyn_matrix<T, 2> _A(MC, KC);
        etl::dyn_matrix<T, 2> _B(KC, NC);
        etl::dyn_matrix<T, 2> _C(MR, NR);

        const size_t mb = (m + MC - 1) / MC;
        const size_t nb = (n + NC - 1) / NC;
        const size_t kb = (k + KC - 1) / KC;

        const size_t _mc = m % MC;
        const size_t _nc = n % NC;
        const size_t _kc = k % KC;

        const size_t incRowA = k;
        const size_t incColA = 1;

        const size_t incRowB = n;
        const size_t incColB = 1;

        const size_t incRowC = n;
        const size_t incColC = 1;

        const T alpha = 1.0;

        for (size_t j = 0; j < nb; ++j) {
            const size_t nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

            for (size_t l = 0; l < kb; ++l) {
                const size_t kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
                T _beta         = (l == 0) ? beta : 1.0;

                pack_b(kc, nc, &B[l * KC * incRowB + j * NC * incColB], incRowB, incColB, _B.memory_start());

                for (size_t i = 0; i < mb; ++i) {
                    const size_t mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                    pack_a(mc, kc, &A[i * MC * incRowA + l * KC * incColA], incRowA, incColA, _A.memory_start());

                    gemm_macro_kernel<V>(mc, nc, kc, alpha, _beta, &C[i * MC * incRowC + j * NC * incColC], incRowC, incColC, _A, _B, _C);
                }
            }
        }
    } else {
        cpp_unreachable("Should probably not get called");
    }
}

} //end of namespace etl::impl::vec
