//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

/*!
 * \brief Functor for the derivative of 3D Max Pooling
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
    template <size_t C1, size_t C2, typename A, typename B, typename C, typename M>
    static void pool_block_2d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j) {
        auto max = out(i, j);
        auto error = errors(i, j);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                if (max == in(i * C1 + ii, j * C2 + jj)) {
                    m(i * C1 + ii, j * C2 + jj) = error;
                } else {
                    m(i * C1 + ii, j * C2 + jj) = 0.0;
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
     */
    template <size_t C1, size_t C2, typename A, typename B, typename C, typename M, cpp_enable_if(is_2d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                pool_block_2d<C1, C2>(in, out, errors, m, i, j);
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
    static void pool_block_2d(const A& in, const B& out, const C& errors, M& m, size_t i, size_t j, size_t c1, size_t c2) {
        auto max = out(i, j);
        auto error = errors(i, j);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                if (max == in(i * c1 + ii, j * c2 + jj)) {
                    m(i * c1 + ii, j * c2 + jj) = error;
                } else {
                    m(i * c1 + ii, j * c2 + jj) = 0.0;
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
    template <typename A, typename B, typename C, typename M, cpp_enable_if(is_2d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M&& m, size_t c1, size_t c2) {
        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                pool_block_2d(in, out, errors, m, i, j, c1, c2);
            }
        }
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename B, typename C, typename M, cpp_enable_if(!is_2d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for(size_t i = 0; i < etl::dim<0>(in); ++i){
            apply<C1, C2>(in(i), out(i), errors(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_if(!is_2d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
        for(size_t i = 0; i < etl::dim<0>(in); ++i){
            apply(in(i), out(i), errors(i), m(i), c1, c2);
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
        auto max = out(i, j, k);
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
        auto max = out(n, i, j, k);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_if(is_3d<A>::value)>
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
        auto max = out(i, j, k);
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
        auto max = out(n, i, j, k);
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
    template <typename A, typename B, typename C, typename M, cpp_enable_if(is_3d<A>::value)>
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_if(is_4d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                SERIAL_SECTION {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                                    max_pool_upsample_3d::pool_block_4d<C1, C2, C3>(in, out, errors, m, n, i, j, k);
                                }
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d(batch_fun_n, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_if(is_4d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                SERIAL_SECTION {
                    for (size_t n = first; n < last; ++n) {
                        for (size_t i = 0; i < etl::dim<1>(out); ++i) {
                            for (size_t j = 0; j < etl::dim<2>(out); ++j) {
                                for (size_t k = 0; k < etl::dim<3>(out); ++k) {
                                    max_pool_upsample_3d::pool_block_4d(in, out, errors, m, n, i, j, k, c1, c2, c3);
                                }
                            }
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(out);

        engine_dispatch_1d(batch_fun_n, 0, N, 2UL);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C, typename M, cpp_enable_if(!is_3d<A>::value && !is_4d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m) {
        for(size_t i = 0; i < etl::dim<0>(in); ++i){
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
    template <typename A, typename B, typename C, typename M, cpp_enable_if(!is_3d<A>::value && !is_4d<A>::value)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        for(size_t i = 0; i < etl::dim<0>(in); ++i){
            apply(in(i), out(i), errors(i), m(i), c1, c2, c3);
        }
    }
};

} //end of namespace impl

} //end of namespace etl
