//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/concepts.hpp"
namespace etl::impl::standard {

/*!
 * \brief Functor for 2D Average Pooling
 */
struct avg_pool_2d {
    /*!
     * \brief Pool a block of the sub expression around the border (with padding)
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param s1 The first dimension stride
     * \param s2 The second dimension stride
     * \param p1 The first dimension padding
     * \param p2 The second dimension padding
     */
    template <typename A>
    static auto pool_block_border(const A& sub, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto avg = value_t<A>(0);

        const auto s_j = j * s1;
        const auto s_k = k * s2;

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                if (s_j + jj >= p1 && (s_j + jj) - p1 < etl::dim<0>(sub) && s_k + kk >= p2 && (s_k + kk) - p2 < etl::dim<1>(sub)) {
                    avg += sub(s_j + jj - p1, s_k + kk - p2);
                }
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename A>
    static auto pool_block_2d(const A& sub, size_t j, size_t k) {
        const auto s_j = j * S1 - P1;
        const auto s_k = k * S2 - P2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                avg += sub(s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(C1 * C2);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A>
    static auto pool_block_3d(const A& sub, size_t n, size_t j, size_t k) {
        const auto s_j = j * S1;
        const auto s_k = k * S2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                avg += sub(n, s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(C1 * C2);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, typename A>
    static auto pool_block_4d(const A& sub, size_t m, size_t n, size_t j, size_t k) {
        const auto s_j = j * S1;
        const auto s_k = k * S2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                avg += sub(m, n, s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(C1 * C2);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_2d A, typename M>
    static void apply(const A& sub, M&& m) {
        const size_t o1 = (etl::dim<0>(sub) - C1 + 2 * P1) / S1 + 1;
        const size_t o2 = (etl::dim<1>(sub) - C2 + 2 * P2) / S2 + 1;

        if (P1 || P2) {
            for (size_t i = 0; i < P1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    m(i, j) = pool_block_border(sub, i, j, C1, C2, S1, S2, P1, P2);
                }
            }

            for (size_t i = o1 - P1; i < o1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    m(i, j) = pool_block_border(sub, i, j, C1, C2, S1, S2, P1, P2);
                }
            }

            for (size_t j = 0; j < P2; ++j) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    m(i, j) = pool_block_border(sub, i, j, C1, C2, S1, S2, P1, P2);
                }
            }

            for (size_t j = o2 - P2; j < o2; ++j) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    m(i, j) = pool_block_border(sub, i, j, C1, C2, S1, S2, P1, P2);
                }
            }
        }

        for (size_t j = P1; j < o1 - P1; ++j) {
            for (size_t k = P1; k < o2 - P2; ++k) {
                m(j, k) = pool_block_2d<C1, C2, S1, S2, P1, P2>(sub, j, k);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A>
    static auto pool_block_2d(const A& sub, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        const auto s_j = j * s1 - p1;
        const auto s_k = k * s2 - p2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                avg += sub(s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A>
    static auto pool_block_3d(const A& sub, size_t n, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2) {
        const auto s_j = j * s1;
        const auto s_k = k * s2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                avg += sub(n, s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A>
    static auto pool_block_4d(const A& sub, size_t m, size_t n, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2) {
        const auto s_j = j * s1;
        const auto s_k = k * s2;

        value_t<A> avg = 0;

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                avg += sub(m, n, s_j + jj, s_k + kk);
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <etl_2d A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        const size_t o1 = (etl::dim<0>(sub) - c1 + 2 * p1) / s1 + 1;
        const size_t o2 = (etl::dim<1>(sub) - c2 + 2 * p2) / s2 + 1;

        if (p1 || p2) {
            for (size_t i = 0; i < p1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    m(i, j) = pool_block_border(sub, i, j, c1, c2, s1, s2, p1, p2);
                }
            }

            for (size_t i = o1 - p1; i < o1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    m(i, j) = pool_block_border(sub, i, j, c1, c2, s1, s2, p1, p2);
                }
            }

            for (size_t j = 0; j < p2; ++j) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    m(i, j) = pool_block_border(sub, i, j, c1, c2, s1, s2, p1, p2);
                }
            }

            for (size_t j = o2 - p2; j < o2; ++j) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    m(i, j) = pool_block_border(sub, i, j, c1, c2, s1, s2, p1, p2);
                }
            }
        }

        for (size_t j = p1; j < o1 - p1; ++j) {
            for (size_t k = p2; k < o2 - p2; ++k) {
                m(j, k) = pool_block_2d(sub, j, k, c1, c2, s1, s2, p1, p2);
            }
        }
    }

    /*
     * 3D handling
     *
     * This is especially optimized because this is the most common
     * case in machine learning. Moreover, this is also easy to
     * parallelize and optimize
     */

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_3d A, typename M>
    static void apply(const A& sub, M&& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!P1 && !P2)) {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t j = 0; j < etl::dim<1>(m); ++j) {
                            for (size_t k = 0; k < etl::dim<2>(m); ++k) {
                                m(n, j, k) = pool_block_3d<C1, C2, S1, S2>(sub, n, j, k);
                            }
                        }
                    }
                } else {
                    // In the general case, we use the regular algorithm
                    for (size_t n = first; n < last; ++n) {
                        apply<C1, C2, S1, S2, P1, P2>(sub(n), m(n));
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <etl_3d A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!p1 && !p2)) {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t j = 0; j < etl::dim<1>(m); ++j) {
                            for (size_t k = 0; k < etl::dim<2>(m); ++k) {
                                m(n, j, k) = pool_block_3d(sub, n, j, k, c1, c2, s1, s2);
                            }
                        }
                    }
                } else {
                    for (size_t n = first; n < last; ++n) {
                        apply(sub(n), m(n), c1, c2, s1, s2, p1, p2);
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
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
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_4d A, typename M>
    static void apply(const A& sub, M&& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!P1 && !P2)) {
                    for (size_t mm = first; mm < last; ++mm) {
                        for (size_t n = 0; n < etl::dim<1>(m); ++n) {
                            for (size_t j = 0; j < etl::dim<2>(m); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(m); ++k) {
                                    m(mm, n, j, k) = pool_block_4d<C1, C2, S1, S2>(sub, mm, n, j, k);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t mm = first; mm < last; ++mm) {
                        for (size_t n = 0; n < etl::dim<1>(m); ++n) {
                            apply<C1, C2, S1, S2, P1, P2>(sub(mm)(n), m(mm)(n));
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <etl_4d A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!p1 && !p2)) {
                    for (size_t mm = first; mm < last; ++mm) {
                        for (size_t n = 0; n < etl::dim<1>(m); ++n) {
                            for (size_t j = 0; j < etl::dim<2>(m); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(m); ++k) {
                                    m(mm, n, j, k) = pool_block_4d(sub, mm, n, j, k, c1, c2, s1, s2);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t mm = first; mm < last; ++mm) {
                        for (size_t n = 0; n < etl::dim<1>(m); ++n) {
                            apply(sub(mm)(n), m(mm)(n), c1, c2, s1, s2, p1, p2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam S1 The first dimension stride
     * \tparam S2 The second dimension stride
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, etl_5d_and_plus A, typename M>
    static void apply(const A& sub, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(sub); ++i) {
            apply<C1, C2, S1, S2, P1, P2>(sub(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <etl_5d_and_plus A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        for (size_t i = 0; i < etl::dim<0>(sub); ++i) {
            apply(sub(i), m(i), c1, c2, s1, s2, p1, p2);
        }
    }
};

/*!
 * \brief Functor for 3D Average Pooling
 */
struct avg_pool_3d {
    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     * \param s1 The first dimension stride
     * \param s2 The second dimension stride
     * \param s3 The third dimension stride
     * \param p1 The first dimension padding
     * \param p2 The second dimension padding
     * \param p3 The third dimension padding
     */
    template <typename A>
    static auto pool_block_border(
        const A& sub, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        auto avg = value_t<A>(0);

        const auto s_i = i * s1;
        const auto s_j = j * s2;
        const auto s_k = k * s3;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    if (s_i + ii >= p1 && (s_i + ii) - p1 < etl::dim<0>(sub) && s_j + jj >= p2 && (s_j + jj) - p2 < etl::dim<1>(sub) && s_k + kk >= p3
                        && (s_k + kk) - p3 < etl::dim<2>(sub)) {
                        avg += sub(s_i + ii - p1, s_j + jj - p2, s_k + kk - p3);
                    }
                }
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2 * c3);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A>
    static auto pool_block_3d(const A& sub, size_t i, size_t j, size_t k) {
        const auto s_i = i * S1 - P1;
        const auto s_j = j * S2 - P2;
        const auto s_k = k * S3 - P3;

        value_t<A> avg = 0;

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    avg += sub(s_i + ii, s_j + jj, s_k + kk);
                }
            }
        }

        return avg / static_cast<value_t<A>>(C1 * C2 * C3);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, typename A>
    static auto pool_block_4d(const A& sub, size_t n, size_t i, size_t j, size_t k) {
        const auto s_i = i * S1;
        const auto s_j = j * S2;
        const auto s_k = k * S3;

        value_t<A> avg = 0;

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    avg += sub(n, s_i + ii, s_j + jj, s_k + kk);
                }
            }
        }

        return avg / static_cast<value_t<A>>(C1 * C2 * C3);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1,
              size_t C2,
              size_t C3,
              size_t S1,
              size_t S2,
              size_t S3,
              size_t P1,
              size_t P2,
              size_t P3,
              etl_3d A,
              typename M>
    static void apply(const A& sub, M&& m) {
        const size_t o1 = (etl::dim<0>(sub) - C1 + 2 * P1) / S1 + 1;
        const size_t o2 = (etl::dim<1>(sub) - C2 + 2 * P2) / S2 + 1;
        const size_t o3 = (etl::dim<2>(sub) - C3 + 2 * P3) / S3 + 1;

        if (P1 || P2 || P3) {
            for (size_t i = 0; i < P1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }

            for (size_t i = o1 - P1; i < o1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }

            for (size_t j = 0; j < P2; ++j) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }

            for (size_t j = o2 - P2; j < o2; ++j) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }

            for (size_t k = 0; k < P3; ++k) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    for (size_t j = P2; j < o2 - P2; ++j) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }

            for (size_t k = o3 - P3; k < o3; ++k) {
                for (size_t i = P1; i < o1 - P1; ++i) {
                    for (size_t j = P2; j < o2 - P2; ++j) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, C1, C2, C3, S1, S2, S3, P1, P2, P3);
                    }
                }
            }
        }

        for (size_t i = P1; i < o1 - P1; ++i) {
            for (size_t j = P2; j < o2 - P2; ++j) {
                for (size_t k = P3; k < o3 - P3; ++k) {
                    m(i, j, k) = pool_block_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(sub, i, j, k);
                }
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A>
    static auto pool_block_3d(
        const A& sub, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        const auto s_i = i * s1 - p1;
        const auto s_j = j * s2 - p2;
        const auto s_k = k * s3 - p3;

        value_t<A> avg = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    avg += sub(s_i + ii, s_j + jj, s_k + kk);
                }
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2 * c3);
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A>
    static auto pool_block_4d(const A& sub, size_t n, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3) {
        const auto s_i = i * s1;
        const auto s_j = j * s2;
        const auto s_k = k * s3;

        value_t<A> avg = 0;

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    avg += sub(n, s_i + ii, s_j + jj, s_k + kk);
                }
            }
        }

        return avg / static_cast<value_t<A>>(c1 * c2 * c3);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <etl_3d A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        const size_t o1 = (etl::dim<0>(sub) - c1 + 2 * p1) / s1 + 1;
        const size_t o2 = (etl::dim<1>(sub) - c2 + 2 * p2) / s2 + 1;
        const size_t o3 = (etl::dim<2>(sub) - c3 + 2 * p3) / s3 + 1;

        if (p1 || p2 || p3) {
            for (size_t i = 0; i < p1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }

            for (size_t i = o1 - p1; i < o1; ++i) {
                for (size_t j = 0; j < o2; ++j) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }

            for (size_t j = 0; j < p2; ++j) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }

            for (size_t j = o2 - p2; j < o2; ++j) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    for (size_t k = 0; k < o3; ++k) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }

            for (size_t k = 0; k < p3; ++k) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    for (size_t j = p2; j < o2 - p2; ++j) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }

            for (size_t k = o3 - p3; k < o3; ++k) {
                for (size_t i = p1; i < o1 - p1; ++i) {
                    for (size_t j = p2; j < o2 - p2; ++j) {
                        m(i, j, k) = pool_block_border(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }
        }

        for (size_t i = p1; i < o1 - p1; ++i) {
            for (size_t j = p2; j < o2 - p2; ++j) {
                for (size_t k = p3; k < o3 - p3; ++k) {
                    m(i, j, k) = pool_block_3d(sub, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
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
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1,
              size_t C2,
              size_t C3,
              size_t S1,
              size_t S2,
              size_t S3,
              size_t P1,
              size_t P2,
              size_t P3,
              etl_4d A,
              typename M>
    static void apply(const A& sub, M&& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!P1 && !P2 && !P3)) {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t i = 0; i < etl::dim<1>(m); ++i) {
                            for (size_t j = 0; j < etl::dim<2>(m); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(m); ++k) {
                                    m(n, i, j, k) = pool_block_4d<C1, C2, C3, S1, S2, S3>(sub, n, i, j, k);
                                }
                            }
                        }
                    }
                } else {
                    // In the general case, we use the regular algorithm
                    for (size_t n = first; n < last; ++n) {
                        apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(sub(n), m(n));
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <etl_4d A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                if (cpp_likely(!p1 && !p2 && !p3)) {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t i = 0; i < etl::dim<1>(m); ++i) {
                            for (size_t j = 0; j < etl::dim<2>(m); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(m); ++k) {
                                    m(n, i, j, k) = pool_block_4d(sub, n, i, j, k, c1, c2, c3, s1, s2, s3);
                                }
                            }
                        }
                    }
                } else {
                    for (size_t n = first; n < last; ++n) {
                        apply(sub(n), m(n), c1, c2, c3, s1, s2, s3, p1, p2, p3);
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(m);

        engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1,
              size_t C2,
              size_t C3,
              size_t S1,
              size_t S2,
              size_t S3,
              size_t P1,
              size_t P2,
              size_t P3,
              etl_5d_and_plus A,
              typename M>
    static void apply(const A& sub, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(sub); ++i) {
            apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(sub(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <etl_5d_and_plus A, typename M>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        for (size_t i = 0; i < etl::dim<0>(sub); ++i) {
            apply(sub(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }
    }
};

} //end of namespace etl::impl::standard
