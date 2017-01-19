//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace detail {

template<typename T>
inline void pmp_h_kernel_2x2(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m >> 1) << 1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n >> 1) << 1;

            base(m, n) =
                exp_sub(start_mm + 0, start_nn + 0)
                + exp_sub(start_mm + 0, start_nn + 1)
                + exp_sub(start_mm + 1, start_nn + 0)
                + exp_sub(start_mm + 1, start_nn + 1);
        }
    }
}

template<size_t C1, size_t C2, typename T>
inline void pmp_h_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m / C1) * C1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n / C2) * C2;

            auto p = T(0);

            for (std::size_t mm = start_mm; mm < start_mm + C1; ++mm) {
                for (std::size_t nn = start_nn; nn < start_nn + C2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

template<typename T>
inline void pmp_h_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base, size_t c1, size_t c2){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M; ++m) {
        const auto start_mm = (m / c1) * c1;

        for (size_t n = 0; n < N; ++n) {
            const auto start_nn = (n / c2) * c2;

            auto p = T(0);

            for (std::size_t mm = start_mm; mm < start_mm + c1; ++mm) {
                for (std::size_t nn = start_nn; nn < start_nn + c2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

/*!
 * \brief Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<size_t D, size_t C1, size_t C2>
struct pmp_h_impl ;

/*!
 * \brief 2D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<size_t C1, size_t C2>
struct pmp_h_impl <2, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        exp_sub = exp(a);

        if (C1 == 2 && C2 == 2) {
            pmp_h_kernel_2x2(exp_sub, base);
        } else {
            pmp_h_kernel<C1, C2>(exp_sub, base);
        }

        c = exp_sub / (1.0 + base);
    }
};

/*!
 * \brief 3D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<size_t C1, size_t C2>
struct pmp_h_impl <3, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

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
};

/*!
 * \brief 4D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<size_t C1, size_t C2>
struct pmp_h_impl <4, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

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
};

/*!
 * \brief Dynamic Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<size_t D>
struct dyn_pmp_h_impl ;

/*!
 * \brief Dynamic 2D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<>
struct dyn_pmp_h_impl <2> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_h_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        using T = value_t<A>;

        const size_t M = etl::dim<0>(a);
        const size_t N = etl::dim<1>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

        exp_sub = exp(a);

        if(c1 == 2 && c2 == 2){
            pmp_h_kernel_2x2(exp_sub, base);
        } else {
            pmp_h_kernel(exp_sub, base, c1, c2);
        }

        c = exp_sub / (1.0 + base);
    }
};

/*!
 * \brief Dynamic 3D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<>
struct dyn_pmp_h_impl <3> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_h_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        using T = value_t<A>;

        const size_t L = etl::dim<0>(a);
        const size_t M = etl::dim<1>(a);
        const size_t N = etl::dim<2>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

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
};

/*!
 * \brief Dynamic 4D Implemenetation of Probabilistic Max Pooling for hidden units
 */
template<>
struct dyn_pmp_h_impl <4> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_h_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        using T = value_t<A>;

        const size_t K = etl::dim<0>(a);
        const size_t L = etl::dim<1>(a);
        const size_t M = etl::dim<2>(a);
        const size_t N = etl::dim<3>(a);

        etl::dyn_matrix<T, 2> exp_sub(M, N);
        etl::dyn_matrix<T, 2> base(M, N);

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
};

template<typename T>
inline void pmp_p_kernel_2x2(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / 2; ++m) {
        const auto start_mm = m * 2;

        for (size_t n = 0; n < N / 2; ++n) {
            const auto start_nn = n * 2;

            base(m, n) =
                    exp_sub(start_mm + 0, start_nn + 0)
                +   exp_sub(start_mm + 0, start_nn + 1)
                +   exp_sub(start_mm + 1, start_nn + 0)
                +   exp_sub(start_mm + 1, start_nn + 1);
        }
    }
}

template<size_t C1, size_t C2, typename T>
inline void pmp_p_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / C1; ++m) {
        const auto start_mm = m * C1;

        for (size_t n = 0; n < N / C2; ++n) {
            const auto start_nn = n * C2;

            auto p = T(0);

            for (std::size_t mm = start_mm; mm < start_mm + C1; ++mm) {
                for (std::size_t nn = start_nn; nn < start_nn + C2; ++nn) {
                    p += exp_sub(mm, nn);
                }
            }

            base(m, n) = p;
        }
    }
}

template<typename T>
inline void pmp_p_kernel(etl::dyn_matrix<T, 2>& exp_sub, etl::dyn_matrix<T, 2>& base, size_t c1, size_t c2){
    const size_t M = etl::dim<0>(exp_sub);
    const size_t N = etl::dim<1>(exp_sub);

    for (size_t m = 0; m < M / c1; ++m) {
        const auto start_mm = m * c1;

        for (size_t n = 0; n < N / c2; ++n) {
            const auto start_nn = n * c2;

            auto p = T(0);

            for (std::size_t mm = start_mm; mm < start_mm + c1; ++mm) {
                for (std::size_t nn = start_nn; nn < start_nn + c2; ++nn) {
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
template<size_t D, size_t C1, size_t C2>
struct pmp_p_impl ;

/*!
 * \brief 2D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<size_t C1, size_t C2>
struct pmp_p_impl <2, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
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
};

/*!
 * \brief 3D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<size_t C1, size_t C2>
struct pmp_p_impl <3, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
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
};

/*!
 * \brief 4D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<size_t C1, size_t C2>
struct pmp_p_impl <4, C1, C2> {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
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
 * \brief Dynamic Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<size_t D>
struct dyn_pmp_p_impl ;

/*!
 * \brief Dynamic 2D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<>
struct dyn_pmp_p_impl <2> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_p_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
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
};

/*!
 * \brief Dynamic 3D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<>
struct dyn_pmp_p_impl <3> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_p_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
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
};

/*!
 * \brief Dynamic 4D Implemenetation of Probabilistic Max Pooling for pooling units
 */
template<>
struct dyn_pmp_p_impl <4> {
    const size_t c1; ///< Pooling factor for the first dimension
    const size_t c2; ///< Pooling factor for the second dimension

    /*!
     * \brief Construct a new functor with the given pooling ratios
     */
    dyn_pmp_p_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
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

} //end of namespace detail

} //end of namespace etl
