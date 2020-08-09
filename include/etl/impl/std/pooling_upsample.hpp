//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Functor for the derivative of 2D Max Pooling
 */
struct max_pool_upsample_2d {
    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M>
    static void pool_block_2d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j) {
        auto max   = out(i, j);
        auto error = errors(i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                if (max == in(i * S1 + ii, j * S2 + jj)) {
                    m(i * S1 + ii, j * S2 + jj) = error;
                } else {
                    m(i * S1 + ii, j * S2 + jj) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M>
    static void pool_block_3d(const A& in, const B& out, const C& errors, M& m, size_t q, size_t i, size_t j) {
        auto max   = out(q, i, j);
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                if (max == in(q, i * S1 + ii, j * S2 + jj)) {
                    m(q, i * S1 + ii, j * S2 + jj) = error;
                } else {
                    m(q, i * S1 + ii, j * S2 + jj) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M>
    static void pool_block_4d(const A& in, const B& out, const C& errors, M& m, size_t p, size_t q, size_t i, size_t j) {
        auto max   = out(p, q, i, j);
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                if (max == in(p, q, i * S1 + ii, j * S2 + jj)) {
                    m(p, q, i * S1 + ii, j * S2 + jj) += error;
                } else {
                    m(p, q, i * S1 + ii, j * S2 + jj) += 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void pool_block_2d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto max   = out(i, j);
        auto error = errors(i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                if (max == in(i * s1 + ii, j * s2 + jj)) {
                    m(i * s1 + ii, j * s2 + jj) = error;
                } else {
                    m(i * s1 + ii, j * s2 + jj) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_2d_upsample(const C& errors, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(i * s1 + ii, j * s2 + jj) += error;
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_2d_select(const A& in, const B& out, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out((i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(i, j) *= count;
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void pool_block_3d(const A& in, const B& out, const C& errors, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto max   = out(q, i, j);
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                if (max == in(q, i * s1 + ii, j * s2 + jj)) {
                    m(q, i * s1 + ii, j * s2 + jj) = error;
                } else {
                    m(q, i * s1 + ii, j * s2 + jj) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_3d_upsample(const C& errors, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(q, i * s1 + ii, j * s2 + jj) += error;
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_3d_select(const A& in, const B& out, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(q, i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out(q, (i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(q, i, j) *= count;
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void pool_block_4d(const A& in, const B& out, const C& errors, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto max   = out(p, q, i, j);
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                if (max == in(p, q, i * s1 + ii, j * s2 + jj)) {
                    m(p, q, i * s1 + ii, j * s2 + jj) = error;
                } else {
                    m(p, q, i * s1 + ii, j * s2 + jj) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_4d_upsample(const C& errors, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(p, q, i * s1 + ii, j * s2 + jj) += error;
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_4d_select(const A& in, const B& out, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(p, q, i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out(p, q, (i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(p, q, i, j) *= count;
    }

    // 2D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m) {
        if constexpr (S1 == C1 && S2 == C2) {
            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d<C1, C2, S1, S2>(in, out, errors, m, i, j);
                }

            }
        } else {
            // Note: The stride versions are highly unoptimized

            m = 0;

            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d_upsample(errors, m, i, j, C1, C2, S1, S2);
                }
            }

            for (size_t i = 0; i < etl::dim<0>(in); ++i) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    pool_block_2d_select(in, out, m, i, j, C1, C2, S1, S2);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 == c1 && s2 == c2) {
            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d(in, out, errors, m, i, j, c1, c2, s1, s2);
                }
            }
        } else {
            // Note: The stride versions are highly unoptimized

            m = 0;

            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d_upsample(errors, m, i, j, c1, c2, s1, s2);
                }
            }

            for (size_t i = 0; i < etl::dim<0>(in); ++i) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    pool_block_2d_select(in, out, m, i, j, c1, c2, s1, s2);
                }
            }
        }
    }

    // 3D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m) {
        if (S1 != C1 || S2 != C2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                if constexpr (S1 == C1 && S2 == C2) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d<C1, C2, S1, S2>(in, out, errors, m, q, i, j);
                        }
                    }
                } else {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d_upsample(errors, m, q, i, j, C1, C2, S1, S2);
                        }
                    }

                    for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                            pool_block_3d_select(in, out, m, q, i, j, C1, C2, S1, S2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 != c1 || s2 != c2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            if (s1 == c1 && s2 == c2) {
                for (size_t q = first; q < last; ++q) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d(in, out, errors, m, q, i, j, c1, c2, s1, s2);
                        }
                    }
                }
            } else {
                for (size_t q = first; q < last; ++q) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d_upsample(errors, m, q, i, j, c1, c2, s1, s2);
                        }
                    }

                    for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                            pool_block_3d_select(in, out, m, q, i, j, c1, c2, s1, s2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // 4D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m) {
        if (S1 != C1 || S2 != C2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t p = first; p < last; ++p) {
                for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                    if constexpr (S1 == C1 && S2 == C2) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d<C1, C2, S1, S2>(in, out, errors, m, p, q, i, j);
                            }
                        }
                    } else {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_3d_upsample(errors, m, p, q, i, j, C1, C2, S1, S2);
                            }
                        }

                        for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(in); ++j) {
                                pool_block_3d_select(in, out, m, p, q, i, j, C1, C2, S1, S2);
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 != c1 || s2 != c2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            if(s1 == c1 && s2 == c2) {
                for (size_t p = first; p < last; ++p) {
                    for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d(in, out, errors, m, p, q, i, j, c1, c2, s1, s2);
                            }
                        }
                    }
                }
            } else {
                for (size_t p = first; p < last; ++p) {
                    for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_3d_upsample(errors, m, p, q, i, j, c1, c2, s1, s2);
                            }
                        }

                        for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(in); ++j) {
                                pool_block_3d_select(in, out, m, p, q, i, j, c1, c2, s1, s2);
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, S1, S2>(in(i), out(i), errors(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, s1, s2);
        }
    }
};

/*!
 * \brief Functor for the derivative of 3D Max Pooling
 */
struct max_pool_upsample_3d {
    /*!
     * \brief Pool a 3D block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M>
    static void pool_block_3d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j, size_t k) {
        auto max   = out(i, j, k);
        auto error = errors(i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    if (max == in(i * C1 + ii, j * C2 + jj, k * C3 + kk)) {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = error;
                    } else {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 0.0;
                    }
                }
            }
        }
    }

    /*!
     * \brief Pool a 4D block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M>
    static void pool_block_4d(const A& in, const B& out, const C& errors, M& m, size_t n, size_t i, size_t j, size_t k) {
        auto max   = out(n, i, j, k);
        auto error = errors(n, i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    if (max == in(n, i * C1 + ii, j * C2 + jj, k * C3 + kk)) {
                        m(n, i * C1 + ii, j * C2 + jj, k * C3 + kk) = error;
                    } else {
                        m(n, i * C1 + ii, j * C2 + jj, k * C3 + kk) = 0.0;
                    }
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_block_3d<C1, C2, C3>(in, out, errors, m, i, j, k);
                }
            }
        }
    }

    /*!
     * \brief Pool a 3D block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void pool_block_3d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto max   = out(i, j, k);
        auto error = errors(i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    if (max == in(i * c1 + ii, j * c2 + jj, k * c3 + kk)) {
                        m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = error;
                    } else {
                        m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = 0.0;
                    }
                }
            }
        }
    }

    /*!
     * \brief Pool a 4D block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void pool_block_4d(const A& in, const B& out, const C& errors, M& m, size_t n, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto max   = out(n, i, j, k);
        auto error = errors(n, i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    if (max == in(n, i * c1 + ii, j * c2 + jj, k * c3 + kk)) {
                        m(n, i * c1 + ii, j * c2 + jj, k * c3 + kk) = error;
                    } else {
                        m(n, i * c1 + ii, j * c2 + jj, k * c3 + kk) = 0.0;
                    }
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_block_3d(in, out, errors, m, i, j, k, c1, c2, c3);
                }
            }
        }
    }

    /*
     * 4D handling
     *
     * This is especially optimized because this is the most common
     * case in machine learning. Moreover, this is also easy to
     * parallelize and optimize
     */

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            for (size_t n = first; n < last; ++n) {
                for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                            max_pool_upsample_3d::pool_block_4d<C1, C2, C3>(in, out, errors, m, n, i, j, k);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            for (size_t n = first; n < last; ++n) {
                for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                            max_pool_upsample_3d::pool_block_4d(in, out, errors, m, n, i, j, k, c1, c2, c3);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(!is_3d<A> && !is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3>(in(i), out(i), errors(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(!is_3d<A> && !is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, c3);
        }
    }
};

/*!
 * \brief Functor for the derivative of 2D Max Pooling
 */
struct avg_pool_upsample_2d {
    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename C, typename M>
    static void pool_block_2d(const C& errors, M& m, size_t i, size_t j) {
        auto error = errors(i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                m(i * S1 + ii, j * S2 + jj) = error / (C1 * C2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename C, typename M>
    static void pool_block_3d(const C& errors, M& m, size_t q, size_t i, size_t j) {
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                m(q, i * S1 + ii, j * S2 + jj) = error / (C1 * C2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename C, typename M>
    static void pool_block_4d(const C& errors, M& m, size_t p, size_t q, size_t i, size_t j) {
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                m(p, q, i * S1 + ii, j * S2 + jj) = error / (C1 * C2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_2d(const C& errors, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(i * s1 + ii, j * s2 + jj) = error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_2d_upsample(const C& errors, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(i * s1 + ii, j * s2 + jj) += error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_2d_select(const A& in, const B& out, M& m, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out((i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(i, j) *= count;
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_3d(const C& errors, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(q, i * s1 + ii, j * s2 + jj) = error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_3d_upsample(const C& errors, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(q, i * s1 + ii, j * s2 + jj) += error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_3d_select(const A& in, const B& out, M& m, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(q, i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out(q, (i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(q, i, j) *= count;
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_4d(const C& errors, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(p, q, i * s1 + ii, j * s2 + jj) = error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_4d_upsample(const C& errors, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto error = errors(p, q, i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                m(p, q, i * s1 + ii, j * s2 + jj) += error / (c1 * c2);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_block_4d_select(const A& in, const B& out, M& m, size_t p, size_t q, size_t i, size_t j, size_t c1, size_t c2, size_t s1, size_t s2) {
        auto value = in(p, q, i, j);

        size_t count = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                size_t i3 = i + ii;
                size_t j3 = j + jj;

                if (i3 >= 1 && (i3 - 1) / s1 < etl::dim<0>(out)) {
                    if (j3 >= 1 && (j3 - 1) / s2 < etl::dim<1>(out)) {
                        if (value == out(p, q, (i3 - 1) / s1, (j3 - 1) / s2)) {
                            ++count;
                        }
                    }
                }
            }
        }

        m(p, q, i, j) *= count;
    }

    // 2D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m) {
        if constexpr (S1 == C1 && S2 == C2) {
            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d<C1, C2, S1, S2>(errors, m, i, j);
                }
            }
        } else {
            // Note: The stride versions are highly unoptimized

            m = 0;

            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d_upsample(errors, m, i, j, C1, C2, S1, S2);
                }
            }

            for (size_t i = 0; i < etl::dim<0>(in); ++i) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    pool_block_2d_select(in, out, m, i, j, C1, C2, S1, S2);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 == c1 && s2 == c2) {
            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d(errors, m, i, j, c1, c2, s1, s2);
                }
            }
        } else {
            // Note: The stride versions are highly unoptimized

            m = 0;

            for (size_t i = 0; i < etl::dim<0>(out); ++i) {
                for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                    pool_block_2d_upsample(errors, m, i, j, c1, c2, s1, s2);
                }
            }

            for (size_t i = 0; i < etl::dim<0>(in); ++i) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    pool_block_2d_select(in, out, m, i, j, c1, c2, s1, s2);
                }
            }
        }
    }

    // 3D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m) {
        if (S1 != C1 || S2 != C2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                if constexpr (S1 == C1 && S2 == C2) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d<C1, C2, S1, S2>(errors, m, q, i, j);
                        }
                    }
                } else {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d_upsample(errors, m, q, i, j, C1, C2, S1, S2);
                        }
                    }

                    for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                            pool_block_3d_select(in, out, m, q, i, j, C1, C2, S1, S2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 != c1 || s2 != c2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            if (s1 == c1 && s2 == c2) {
                for (size_t q = first; q < last; ++q) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d(errors, m, q, i, j, c1, c2, s1, s2);
                        }
                    }
                }
            } else {
                for (size_t q = first; q < last; ++q) {
                    for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                            pool_block_3d_upsample(errors, m, q, i, j, c1, c2, s1, s2);
                        }
                    }

                    for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                        for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                            pool_block_3d_select(in, out, m, q, i, j, c1, c2, s1, s2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // 4D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m) {
        if (S1 != C1 || S2 != C2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t p = first; p < last; ++p) {
                for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                    if constexpr (S1 == C1 && S2 == C2) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d<C1, C2, S1, S2>(errors, m, p, q, i, j);
                            }
                        }
                    } else {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d_upsample(errors, m, q, i, j, C1, C2, S1, S2);
                            }
                        }

                        for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(in); ++j) {
                                pool_block_4d_select(in, out, m, q, i, j, C1, C2, S1, S2);
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        if (s1 != c1 || s2 != c2) {
            m = 0;
        }

        auto batch_fun = [&](const size_t first, const size_t last) {
            if (s1 == c1 && s2 == c2) {
                for (size_t p = first; p < last; ++p) {
                    for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d(errors, m, p, q, i, j, c1, c2, s1, s2);
                            }
                        }
                    }
                }
            } else {
                for (size_t p = first; p < last; ++p) {
                    for (size_t q = 0; q < etl::dim<1>(out); ++q) {
                        for (size_t i = 0; i < etl::dim<2>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(out); ++j) {
                                pool_block_4d_upsample(errors, m, q, i, j, c1, c2, s1, s2);
                            }
                        }

                        for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                            for (size_t j = 0; j < etl::dim<3>(in); ++j) {
                                pool_block_4d_select(in, out, m, q, i, j, c1, c2, s1, s2);
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, S1, S2>(in(i), out(i), errors(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t s1, size_t s2) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, s1, s2);
        }
    }
};

/*!
 * \brief Functor for the derivative of 3D Avg Pooling
 */
struct avg_pool_upsample_3d {
    /*!
     * \brief Pool a 3D block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename C, typename M>
    static void pool_block_3d(const C& errors, M& m, size_t i, size_t j, size_t k) {
        auto error = errors(i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = error / (C1 * C2 * C3);
                }
            }
        }
    }

    /*!
     * \brief Pool a 4D block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename C, typename M>
    static void pool_block_4d(const C& errors, M& m, size_t n, size_t i, size_t j, size_t k) {
        auto error = errors(n, i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    m(n, i * C1 + ii, j * C2 + jj, k * C3 + kk) = error / (C1 * C2 * C3);
                }
            }
        }
    }

    /*!
     * \brief Pool a 3D block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_3d(const C& errors, M& m, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto error = errors(i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = error / (c1 * c2 * c3);
                }
            }
        }
    }

    /*!
     * \brief Pool a 4D block of the sub expression
     * \param errors The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename C, typename M>
    static void pool_block_4d(const C& errors, M& m, size_t n, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto error = errors(n, i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    m(n, i * c1 + ii, j * c2 + jj, k * c3 + kk) = error / (c1 * c2 * c3);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_block_3d<C1, C2, C3>(errors, m, i, j, k);
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M&& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_block_3d(errors, m, i, j, k, c1, c2, c3);
                }
            }
        }
    }

    /*
     * 4D handling
     *
     * This is especially optimized because this is the most common
     * case in machine learning. Moreover, this is also easy to
     * parallelize and optimize
     */

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            for (size_t n = first; n < last; ++n) {
                for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                            avg_pool_upsample_3d::pool_block_4d<C1, C2, C3>(errors, m, n, i, j, k);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply([[maybe_unused]] A&& in, [[maybe_unused]] B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            for (size_t n = first; n < last; ++n) {
                for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                            avg_pool_upsample_3d::pool_block_4d(errors, m, n, i, j, k, c1, c2, c3);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_iff(!is_3d<A> && !is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3>(in(i), out(i), errors(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(!is_3d<A> && !is_4d<A>)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, c3);
        }
    }
};

} //end of namespace etl::impl::standard
