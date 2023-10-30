//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Kernel for probabilistic max pooling (for hidden units)
 * for a kernel of 2x2
 *
 * This is especially optimized because this is the most common
 * kernel used in machine learning.
 *
 * \param exp_sub The exponentials
 * \param base The output matrix
 */
template <typename T>
inline void pmp_h_kernel_2x2(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m >> 1) << 1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n >> 1) << 1;

            base(m, n) = exp_sub(start_mm + 0, start_nn + 0) + exp_sub(start_mm + 0, start_nn + 1) + exp_sub(start_mm + 1, start_nn + 0)
                         + exp_sub(start_mm + 1, start_nn + 1);
        }
    }
}

/*!
 * \brief Kernel for probabilistic max pooling (for hidden units)
 * \param exp_sub The exponentials
 * \param base The output matrix
 * \tparam C1 The first dimension pooling ratio
 * \tparam C2 The second dimension pooling ratio
 */
template <size_t C1, size_t C2, typename T>
inline void pmp_h_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m / C1) * C1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n / C2) * C2;

            auto p = T(0);

            for (size_t mm = start_mm; mm < start_mm + C1; ++mm) {
                for (size_t nn = start_nn; nn < start_nn + C2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

/*!
 * \brief Kernel for probabilistic max pooling (for hidden units)
 * \param exp_sub The exponentials
 * \param base The output matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename T>
inline void pmp_h_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base, size_t c1, size_t c2) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m / c1) * c1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n / c2) * c2;

            auto p = T(0);

            for (size_t mm = start_mm; mm < start_mm + c1; ++mm) {
                for (size_t nn = start_nn; nn < start_nn + c2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

/*!
 * \brief 2D Implemenetation of Probabilistic Max Pooling for hidden units
 */
struct pmp_h_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_2d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_h does not support strides");
        static_assert(S2 == C2, "pmp_h does not support strides");
        static_assert(P1 == 0, "pmp_h does not support padding");
        static_assert(P2 == 0, "pmp_h does not support padding");

        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            exp_sub = exp(a);

            if (C1 == 2 && C2 == 2) {
                pmp_h_kernel_2x2(exp_sub, base);
            } else {
                pmp_h_kernel<C1, C2>(exp_sub, base);
            }

            c = exp_sub / (1.0 + base);
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_3d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_h does not support strides");
        static_assert(S2 == C2, "pmp_h does not support strides");
        static_assert(P1 == 0, "pmp_h does not support padding");
        static_assert(P2 == 0, "pmp_h does not support padding");

        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            if (C1 == 2 && C2 == 2) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(l));

                    pmp_h_kernel_2x2(exp_sub, base);

                    c(l) = exp_sub / (1.0 + base);
                }
            } else {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(l));

                    pmp_h_kernel<C1, C2>(exp_sub, base);

                    c(l) = exp_sub / (1.0 + base);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_4d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_h does not support strides");
        static_assert(S2 == C2, "pmp_h does not support strides");
        static_assert(P1 == 0, "pmp_h does not support padding");
        static_assert(P2 == 0, "pmp_h does not support padding");

        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            if (C1 == 2 && C2 == 2) {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t l = 0; l < L; ++l) {
                        exp_sub = exp(a(k)(l));

                        pmp_h_kernel_2x2(exp_sub, base);

                        c(k)(l) = exp_sub / (1.0 + base);
                    }
                }
            } else {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t l = 0; l < L; ++l) {
                        exp_sub = exp(a(k)(l));

                        pmp_h_kernel<C1, C2>(exp_sub, base);

                        c(k)(l) = exp_sub / (1.0 + base);
                    }
                }
            }
        }
    }
};

/*!
 * \brief Dynamic Implemenetation of Probabilistic Max Pooling for hidden units
 */
struct dyn_pmp_h_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_2d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            exp_sub = exp(a);

            if (c1 == 2 && c2 == 2) {
                pmp_h_kernel_2x2(exp_sub, base);
            } else {
                pmp_h_kernel(exp_sub, base, c1, c2);
            }

            c = exp_sub / (1.0 + base);
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_3d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            if (c1 == 2 && c2 == 2) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(l));

                    pmp_h_kernel_2x2(exp_sub, base);

                    c(l) = exp_sub / (1.0 + base);
                }
            } else {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(l));

                    pmp_h_kernel(exp_sub, base, c1, c2);

                    c(l) = exp_sub / (1.0 + base);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_4d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        CPU_SECTION {
            if (c1 == 2 && c2 == 2) {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t l = 0; l < L; ++l) {
                        exp_sub = exp(a(k)(l));

                        pmp_h_kernel_2x2(exp_sub, base);

                        c(k)(l) = exp_sub / (1.0 + base);
                    }
                }
            } else {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t l = 0; l < L; ++l) {
                        exp_sub = exp(a(k)(l));

                        pmp_h_kernel(exp_sub, base, c1, c2);

                        c(k)(l) = exp_sub / (1.0 + base);
                    }
                }
            }
        }
    }
};

/*!
 * \brief Kernel for probabilistic max pooling (for pooling units)
 * with a 2x2 kernel
 *
 * This is especially optimized because this is the most common
 * kernel used in machine learning.
 *
 * \param exp_sub The exponentials
 * \param base The output matrix
 */
template <typename T>
inline void pmp_p_kernel_2x2(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / 2; ++m) {
        const auto start_mm = m * 2;

        for (size_t n = 0; n < N / 2; ++n) {
            const auto start_nn = n * 2;

            base(m, n) = exp_sub(start_mm + 0, start_nn + 0) + exp_sub(start_mm + 0, start_nn + 1) + exp_sub(start_mm + 1, start_nn + 0)
                         + exp_sub(start_mm + 1, start_nn + 1);
        }
    }
}

/*!
 * \brief Kernel for probabilistic max pooling (for pooling units)
 * \param exp_sub The exponentials
 * \param base The output matrix
 * \tparam C1 The first dimension pooling ratio
 * \tparam C2 The second dimension pooling ratio
 */
template <size_t C1, size_t C2, typename T>
inline void pmp_p_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / C1; ++m) {
        const auto start_mm = m * C1;

        for (size_t n = 0; n < N / C2; ++n) {
            const auto start_nn = n * C2;

            auto p = T(0);

            for (size_t mm = start_mm; mm < start_mm + C1; ++mm) {
                for (size_t nn = start_nn; nn < start_nn + C2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

/*!
 * \brief Kernel for probabilistic max pooling (for pooling units)
 * \param exp_sub The exponentials
 * \param base The output matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename T>
inline void pmp_p_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base, size_t c1, size_t c2) {
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / c1; ++m) {
        const auto start_mm = m * c1;

        for (size_t n = 0; n < N / c2; ++n) {
            const auto start_nn = n * c2;

            auto p = T(0);

            for (size_t mm = start_mm; mm < start_mm + c1; ++mm) {
                for (size_t nn = start_nn; nn < start_nn + c2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

/*!
 * \brief Implemenetation of Probabilistic Max Pooling for pooling units
 */
struct pmp_p_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_2d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_p does not support strides");
        static_assert(S2 == C2, "pmp_p does not support strides");
        static_assert(P1 == 0, "pmp_p does not support padding");
        static_assert(P2 == 0, "pmp_p does not support padding");

        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / C1, N / C2);

        exp_sub = exp(a);

        if (C1 == 2 && C2 == 2) {
            pmp_p_kernel_2x2(exp_sub, base);
        } else {
            pmp_p_kernel<C1, C2>(exp_sub, base);
        }

        c = 1.0 / (1.0 + base);
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_3d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_p does not support strides");
        static_assert(S2 == C2, "pmp_p does not support strides");
        static_assert(P1 == 0, "pmp_p does not support padding");
        static_assert(P2 == 0, "pmp_p does not support padding");

        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / C1, N / C2);

        if (C1 == 2 && C2 == 2) {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

                pmp_p_kernel_2x2(exp_sub, base);

                c(l) = 1.0 / (1.0 + base);
            }
        } else {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

                pmp_p_kernel<C1, C2>(exp_sub, base);

                c(l) = 1.0 / (1.0 + base);
            }
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_4d A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(S1 == C1, "pmp_p does not support strides");
        static_assert(S2 == C2, "pmp_p does not support strides");
        static_assert(P1 == 0, "pmp_p does not support padding");
        static_assert(P2 == 0, "pmp_p does not support padding");

        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / C1, N / C2);

        if (C1 == 2 && C2 == 2) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

                    pmp_p_kernel_2x2(exp_sub, base);

                    c(k)(l) = 1.0 / (1.0 + base);
                }
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

                    pmp_p_kernel<C1, C2>(exp_sub, base);

                    c(k)(l) = 1.0 / (1.0 + base);
                }
            }
        }
    }
};

/*!
 * \brief Dynamic 4D Implemenetation of Probabilistic Max Pooling for pooling units
 */
struct dyn_pmp_p_impl {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_2d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / c1, N / c2);

        exp_sub = exp(a);

        if (c1 == 2 && c2 == 2) {
            pmp_p_kernel_2x2(exp_sub, base);
        } else {
            pmp_p_kernel(exp_sub, base, c1, c2);
        }

        c = 1.0 / (1.0 + base);
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_3d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / c1, N / c2);

        if (c1 == 2 && c2 == 2) {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

                pmp_p_kernel_2x2(exp_sub, base);

                c(l) = 1.0 / (1.0 + base);
            }
        } else {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

                pmp_p_kernel(exp_sub, base, c1, c2);

                c(l) = 1.0 / (1.0 + base);
            }
        }
    }

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <etl_4d A, typename C>
    static void apply(A&& a, C&& c, size_t c1, size_t c2, [[maybe_unused]] size_t s1, [[maybe_unused]] size_t s2, [[maybe_unused]] size_t p1, [[maybe_unused]] size_t p2) {
        cpp_assert(s1 == c1, "pmp_p does not support strides");
        cpp_assert(s2 == c2, "pmp_p does not support strides");
        cpp_assert(p1 == 0, "pmp_p does not support pooling");
        cpp_assert(p2 == 0, "pmp_p does not support pooling");

        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M / c1, N / c2);

        if (c1 == 2 && c2 == 2) {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

                    pmp_p_kernel_2x2(exp_sub, base);

                    c(k)(l) = 1.0 / (1.0 + base);
                }
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

                    pmp_p_kernel(exp_sub, base, c1, c2);

                    c(k)(l) = 1.0 / (1.0 + base);
                }
            }
        }
    }
};

} //end of namespace etl::impl::standard
