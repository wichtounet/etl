//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_2d {
    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam S1 The first dimension stride
     * \tparam S2 The second dimension stride
     */
    template <std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2, typename A>
    static auto pool_block(const A& sub, std::size_t j, std::size_t k) {
        auto max = sub(j * S1, k * S2);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                max = std::max(max, sub(j * S1 + jj, k * S2 + kk));
            }
        }

        return max;
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam S1 The first dimension stride
     * \tparam S2 The second dimension stride
     */
    template <std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2, typename A, typename M>
    static void apply(const A& sub, M&& m) {
        const std::size_t o1 = (etl::dim<0>(sub) - C1) / S1 + 1;
        const std::size_t o2 = (etl::dim<1>(sub) - C2) / S2 + 1;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block<C1, C2, S1, S2>(sub, j, k);
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
    static auto pool_block(const A& sub, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2) {
        auto max = sub(j * c1, k * c2);

        for (std::size_t jj = 0; jj < c1; ++jj) {
            for (std::size_t kk = 0; kk < c2; ++kk) {
                max = std::max(max, sub(j * c1 + jj, k * c2 + kk));
            }
        }

        return max;
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M>
    static void apply(A&& sub, M& m, std::size_t c1, std::size_t c2) {
        const std::size_t o1 = etl::dim<0>(sub) / c1;
        const std::size_t o2 = etl::dim<1>(sub) / c2;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block(sub, j, k, c1, c2);
            }
        }
    }
};

/*!
 * \brief Functor for 2D Average Pooling
 */
struct avg_pool_2d {
    /*!
     * \brief Pool a block of the sub expression
     * \param sub The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2, typename A>
    static auto pool_block(const A& sub, std::size_t j, std::size_t k) {
        value_t<A> avg = 0;

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                avg += sub(j * S1 + jj, k * S2 + kk);
            }
        }

        return avg / (C1 * C2);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2, typename A, typename M>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = (etl::dim<0>(sub) - C1) / S1 + 1;
        const std::size_t o2 = (etl::dim<1>(sub) - C2) / S2 + 1;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block<C1, C2, S1, S2>(sub, j, k);
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
    static auto pool_block(const A& sub, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2) {
        value_t<A> avg = 0;

        for (std::size_t jj = 0; jj < c1; ++jj) {
            for (std::size_t kk = 0; kk < c2; ++kk) {
                avg += sub(j * c1 + jj, k * c2 + kk);
            }
        }

        return avg / (c1 * c2);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M>
    static void apply(A&& sub, M& m, std::size_t c1, std::size_t c2) {
        const std::size_t o1 = etl::dim<0>(sub) / c1;
        const std::size_t o2 = etl::dim<1>(sub) / c2;

        for (std::size_t j = 0; j < o1; ++j) {
            for (std::size_t k = 0; k < o2; ++k) {
                m(j, k) = pool_block(sub, j, k, c1, c2);
            }
        }
    }
};

/*!
 * \brief Functor for 3D Max Pooling
 */
struct max_pool_3d {
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
    template <std::size_t C1, std::size_t C2, std::size_t C3, std::size_t S1, std::size_t S2, std::size_t S3, typename A>
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k) {
        auto max = sub(i * S1, j * S2, k * S3);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    max = std::max(max, sub(i * S1 + ii, j * S2 + jj, k * S3 + kk));
                }
            }
        }

        return max;
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t C3,std::size_t S1, std::size_t S2, std::size_t S3, typename A, typename M>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = (etl::dim<0>(sub) - C1) / S1 + 1;
        const std::size_t o2 = (etl::dim<1>(sub) - C2) / S2 + 1;
        const std::size_t o3 = (etl::dim<2>(sub) - C3) / S3 + 1;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block<C1, C2, C3, S1, S2, S3>(sub, i, j, k);
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
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2, std::size_t c3) {
        auto max = sub(i * c1, j * c2, k * c3);

        for (std::size_t ii = 0; ii < c1; ++ii) {
            for (std::size_t jj = 0; jj < c2; ++jj) {
                for (std::size_t kk = 0; kk < c3; ++kk) {
                    max = std::max(max, sub(i * c1 + ii, j * c2 + jj, k * c3 + kk));
                }
            }
        }

        return max;
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename M>
    static void apply(A&& sub, M& m, std::size_t c1, std::size_t c2, std::size_t c3) {
        const std::size_t o1 = etl::dim<0>(sub) / c1;
        const std::size_t o2 = etl::dim<1>(sub) / c2;
        const std::size_t o3 = etl::dim<2>(sub) / c3;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block(sub, i, j, k, c1, c2, c3);
                }
            }
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
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t C3,std::size_t S1, std::size_t S2, std::size_t S3, typename A>
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k) {
        value_t<A> avg = 0;

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    avg += sub(i * S1 + ii, j * S2 + jj, k * S3 + kk);
                }
            }
        }

        return avg / (C1 * C2 * C3);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t C3,std::size_t S1, std::size_t S2, std::size_t S3, typename A, typename M>
    static void apply(A&& sub, M& m) {
        const std::size_t o1 = (etl::dim<0>(sub) - C1) / S1 + 1;
        const std::size_t o2 = (etl::dim<1>(sub) - C2) / S2 + 1;
        const std::size_t o3 = (etl::dim<2>(sub) - C3) / S3 + 1;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block<C1, C2, C3, S1, S2, S3>(sub, i, j, k);
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
    static auto pool_block(const A& sub, std::size_t i, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2, std::size_t c3) {
        value_t<A> avg = 0;

        for (std::size_t ii = 0; ii < c1; ++ii) {
            for (std::size_t jj = 0; jj < c2; ++jj) {
                for (std::size_t kk = 0; kk < c3; ++kk) {
                    avg += sub(i * c1 + ii, j * c2 + jj, k * c3 + kk);
                }
            }
        }

        return avg / (c1 * c2 * c3);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename M>
    static void apply(A&& sub, M& m, std::size_t c1, std::size_t c2, std::size_t c3) {
        const std::size_t o1 = etl::dim<0>(sub) / c1;
        const std::size_t o2 = etl::dim<1>(sub) / c2;
        const std::size_t o3 = etl::dim<2>(sub) / c3;

        for (std::size_t i = 0; i < o1; ++i) {
            for (std::size_t j = 0; j < o2; ++j) {
                for (std::size_t k = 0; k < o3; ++k) {
                    m(i, j, k) = pool_block(sub, i, j, k, c1, c2, c3);
                }
            }
        }
    }
};

/*!
 * \brief Functor for the derivative of 2D Max Pooling
 */
struct max_pool_derivative_2d {
    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t j, std::size_t k) {
        auto max = out(j, k);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                if (max == in(j * C1 + jj, k * C2 + kk)) {
                    m(j * C1 + jj, k * C2 + kk) = 1.0;
                } else {
                    m(j * C1 + jj, k * C2 + kk) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, typename A, typename B, typename M>
    static void apply(A&& in, B&& out, M& m) {
        for (std::size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block<C1, C2>(in, out, m, j, k);
            }
        }
    }

    /*!
     * \brief Pool a block of the sub expression
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2) {
        auto max = out(j, k);

        for (std::size_t jj = 0; jj < c1; ++jj) {
            for (std::size_t kk = 0; kk < c2; ++kk) {
                if (max == in(j * c1 + jj, k * c2 + kk)) {
                    m(j * c1 + jj, k * c2 + kk) = 1.0;
                } else {
                    m(j * c1 + jj, k * c2 + kk) = 0.0;
                }
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void apply(A&& in, B&& out, M& m, std::size_t c1, std::size_t c2) {
        for (std::size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block(in, out, m, j, k, c1, c2);
            }
        }
    }
};

/*!
 * \brief Functor for the derivative of 3D Max Pooling
 */
struct max_pool_derivative_3d {
    /*!
     * \brief Pool a block of the sub expression
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
    template <std::size_t C1, std::size_t C2, std::size_t C3, typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t i, std::size_t j, std::size_t k) {
        auto max = out(i, j, k);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    if (max == in(i * C1 + ii, j * C2 + jj, k * C3 + kk)) {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 1.0;
                    } else {
                        m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 0.0;
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
    template <std::size_t C1, std::size_t C2, std::size_t C3, typename A, typename B, typename M>
    static void apply(A&& in, B&& out, M& m) {
        for (std::size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block<C1, C2, C3>(in, out, m, i, j, k);
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
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, std::size_t i, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2, std::size_t c3) {
        auto max = out(i, j, k);

        for (std::size_t ii = 0; ii < c1; ++ii) {
            for (std::size_t jj = 0; jj < c2; ++jj) {
                for (std::size_t kk = 0; kk < c3; ++kk) {
                    if (max == in(i * c1 + ii, j * c2 + jj, k * c3 + kk)) {
                        m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = 1.0;
                    } else {
                        m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = 0.0;
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
    template <typename A, typename B, typename M>
    static void apply(A&& in, B&& out, M& m, std::size_t c1, std::size_t c2, std::size_t c3) {
        for (std::size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block(in, out, m, i, j, k, c1, c2, c3);
                }
            }
        }
    }
};

/*!
 * \brief Functor for 2D Upsampling
 */
struct upsample_2d {
    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, typename A, typename M>
    static void upsample_block(A&& in, M& m, std::size_t j, std::size_t k) {
        auto value = in(j, k);

        for (std::size_t jj = 0; jj < C1; ++jj) {
            for (std::size_t kk = 0; kk < C2; ++kk) {
                m(j * C1 + jj, k * C2 + kk) = value;
            }
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, typename A, typename M>
    static void apply(A&& in, M& m) {
        for (std::size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block<C1, C2>(in, m, j, k);
            }
        }
    }

    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M>
    static void upsample_block(A&& in, M& m, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2) {
        auto value = in(j, k);

        for (std::size_t jj = 0; jj < c1; ++jj) {
            for (std::size_t kk = 0; kk < c2; ++kk) {
                m(j * c1 + jj, k * c2 + kk) = value;
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
    template <typename A, typename M>
    static void apply(A&& in, M& m, std::size_t c1, std::size_t c2) {
        for (std::size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (std::size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block(in, m, j, k, c1, c2);
            }
        }
    }
};

/*!
 * \brief Functor for 3D Upsampling
 */
struct upsample_3d {
    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <std::size_t C1, std::size_t C2, std::size_t C3, typename A, typename M>
    static void upsample_block(A&& in, M& m, std::size_t i, std::size_t j, std::size_t k) {
        auto value = in(i, j, k);

        for (std::size_t ii = 0; ii < C1; ++ii) {
            for (std::size_t jj = 0; jj < C2; ++jj) {
                for (std::size_t kk = 0; kk < C3; ++kk) {
                    m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = value;
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
    template <std::size_t C1, std::size_t C2, std::size_t C3, typename A, typename M>
    static void apply(A&& in, M& m) {
        for (std::size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block<C1, C2, C3>(in, m, i, j, k);
                }
            }
        }
    }

    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param m The storage matrix
     * \param i The first index of the block
     * \param j The second index of the block
     * \param k The third index of the block
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename M>
    static void upsample_block(A&& in, M& m, std::size_t i, std::size_t j, std::size_t k, std::size_t c1, std::size_t c2, std::size_t c3) {
        auto value = in(i, j, k);

        for (std::size_t ii = 0; ii < c1; ++ii) {
            for (std::size_t jj = 0; jj < c2; ++jj) {
                for (std::size_t kk = 0; kk < c3; ++kk) {
                    m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = value;
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
    template <typename A, typename M>
    static void apply(A&& in, M& m, std::size_t c1, std::size_t c2, std::size_t c3) {
        for (std::size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (std::size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block(in, m, i, j, k, c1, c2, c3);
                }
            }
        }
    }
};

} //end of namespace impl

} //end of namespace etl
