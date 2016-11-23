//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace detail {

template<size_t D, size_t C1, size_t C2>
struct pmp_2d_impl ;

template<size_t C1, size_t C2>
struct pmp_2d_impl <2, C1, C2> {
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

            c = exp_sub / (1.0 + base);
        } else {
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

            c = exp_sub / (1.0 + base);
        }
    }
};

template<size_t C1, size_t C2>
struct pmp_2d_impl <3, C1, C2> {
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

                for (size_t m = 0; m < M; ++m) {
                    const auto start_mm = (m / 2) * 2;

                    for (size_t n = 0; n < N; ++n) {
                        const auto start_nn = (n / 2) * 2;

                        base(m, n) = exp_sub(start_mm + 0, start_nn + 0)
                            + exp_sub(start_mm + 0, start_nn + 1)
                            + exp_sub(start_mm + 1, start_nn + 0)
                            + exp_sub(start_mm + 1, start_nn + 1);
                    }
                }

                c(l) = exp_sub / (1.0 + base);
            }
        } else {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        auto start_mm = (m / C1) * C1;
                        auto start_nn = (n / C2) * C2;

                        auto p = T(0);

                        for (std::size_t mm = start_mm; mm < start_mm + C1; ++mm) {
                            for (std::size_t nn = start_nn; nn < start_nn + C2; ++nn) {
                                p += exp_sub(mm, nn);
                            }
                        }

                        base(m, n) = p;
                    }
                }

                c(l) = exp_sub / (1.0 + base);
            }
        }
    }
};

template<size_t C1, size_t C2>
struct pmp_2d_impl <4, C1, C2> {
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

                    for (size_t m = 0; m < M; ++m) {
                        const auto start_mm = (m / 2) * 2;

                        for (size_t n = 0; n < N; ++n) {
                            const auto start_nn = (n / 2) * 2;

                            base(m, n) = exp_sub(start_mm + 0, start_nn + 0)
                                + exp_sub(start_mm + 0, start_nn + 1)
                                + exp_sub(start_mm + 1, start_nn + 0)
                                + exp_sub(start_mm + 1, start_nn + 1);
                        }
                    }

                    c(k)(l) = exp_sub / (1.0 + base);
                }
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

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

                    c(k)(l) = exp_sub / (1.0 + base);
                }
            }
        }
    }
};

template<size_t D>
struct dyn_pmp_2d_impl ;

template<>
struct dyn_pmp_2d_impl <2> {
    const size_t c1;
    const size_t c2;

    dyn_pmp_2d_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

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
            for (size_t m = 0; m < M; ++m) {
                const auto start_mm = (m >> 1) << 1;

                for (size_t n = 0; n < N; ++n) {
                    const auto start_nn = (n >> 1) << 1;

                    base(m, n) = exp_sub(start_mm + 0, start_nn + 0)
                        + exp_sub(start_mm + 0, start_nn + 1)
                        + exp_sub(start_mm + 1, start_nn + 0)
                        + exp_sub(start_mm + 1, start_nn + 1);
                }
            }

            c = exp_sub / (1.0 + base);
        } else {
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

            c = exp_sub / (1.0 + base);
        }
    }
};

template<>
struct dyn_pmp_2d_impl <3> {
    const size_t c1;
    const size_t c2;

    dyn_pmp_2d_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

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

                for (size_t m = 0; m < M; ++m) {
                    const auto start_mm = (m / 2) * 2;

                    for (size_t n = 0; n < N; ++n) {
                        const auto start_nn = (n / 2) * 2;

                        base(m, n) = exp_sub(start_mm + 0, start_nn + 0)
                            + exp_sub(start_mm + 0, start_nn + 1)
                            + exp_sub(start_mm + 1, start_nn + 0)
                            + exp_sub(start_mm + 1, start_nn + 1);
                    }
                }

                c(l) = exp_sub / (1.0 + base);
            }
        } else {
            for (size_t l = 0; l < L; ++l) {
                exp_sub = exp(a(l));

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

                c(l) = exp_sub / (1.0 + base);
            }
        }
    }
};

template<>
struct dyn_pmp_2d_impl <4> {
    const size_t c1;
    const size_t c2;

    dyn_pmp_2d_impl(size_t c1, size_t c2) : c1(c1), c2(c2) {}

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

                    for (size_t m = 0; m < M; ++m) {
                        const auto start_mm = (m / 2) * 2;

                        for (size_t n = 0; n < N; ++n) {
                            const auto start_nn = (n / 2) * 2;

                            base(m, n) = exp_sub(start_mm + 0, start_nn + 0)
                                + exp_sub(start_mm + 0, start_nn + 1)
                                + exp_sub(start_mm + 1, start_nn + 0)
                                + exp_sub(start_mm + 1, start_nn + 1);
                        }
                    }

                    c(k)(l) = exp_sub / (1.0 + base);
                }
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                for (size_t l = 0; l < L; ++l) {
                    exp_sub = exp(a(k)(l));

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

                    c(k)(l) = exp_sub / (1.0 + base);
                }
            }
        }
    }
};

} //end of namespace detail

} //end of namespace etl
