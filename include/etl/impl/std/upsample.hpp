//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace standard {

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
    template <size_t C1, size_t C2, typename A, typename M>
    static void upsample_block_2d(A&& in, M& m, size_t j, size_t k) {
        auto value = in(j, k);

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                m(j * C1 + jj, k * C2 + kk) = value;
            }
        }
    }

    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M>
    static void upsample_block_3d(A&& in, M& m, size_t q, size_t j, size_t k) {
        auto value = in(q, j, k);

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                m(q, j * C1 + jj, k * C2 + kk) = value;
            }
        }
    }

    /*!
     * \brief Upsample a block of the sub expression
     * \param in The sub expression
     * \param j The first index of the block
     * \param k The second index of the block
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M>
    static void upsample_block_4d(A&& in, M& m, size_t p, size_t q, size_t j, size_t k) {
        auto value = in(p, q, j, k);

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                m(p, q, j * C1 + jj, k * C2 + kk) = value;
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
    static void upsample_block_2d(A&& in, M& m, size_t j, size_t k, size_t c1, size_t c2) {
        auto value = in(j, k);

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                m(j * c1 + jj, k * c2 + kk) = value;
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
    static void upsample_block_3d(A&& in, M& m, size_t q, size_t j, size_t k, size_t c1, size_t c2) {
        auto value = in(q, j, k);

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                m(q, j * c1 + jj, k * c2 + kk) = value;
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
    static void upsample_block_4d(A&& in, M& m, size_t p, size_t q, size_t j, size_t k, size_t c1, size_t c2) {
        auto value = in(p, q, j, k);

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
                m(p, q, j * c1 + jj, k * c2 + kk) = value;
            }
        }
    }

    // 2D handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, M&& m) {
        for (size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block_2d<C1, C2>(in, m, j, k);
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
    template <typename A, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2) {
        for (size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block_2d(in, m, j, k, c1, c2);
            }
        }
    }

    // 3D handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, M&& m) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    for (size_t k = 0; k < etl::dim<2>(in); ++k) {
                        upsample_block_3d<C1, C2>(in, m, q, j, k);
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                    for (size_t k = 0; k < etl::dim<2>(in); ++k) {
                        upsample_block_3d(in, m, q, j, k, c1, c2);
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // 4D handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, M&& m) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t p = first; p < last; ++p) {
                for (size_t q = 0; q < etl::dim<1>(in); ++q) {
                    for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(in); ++k) {
                            upsample_block_4d<C1, C2>(in, m, p, q, j, k);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t p = first; p < last; ++p) {
                for (size_t q = 0; q < etl::dim<1>(in); ++q) {
                    for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(in); ++k) {
                            upsample_block_4d(in, m, p, q, j, k, c1, c2);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // Deep Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2>(in(i), m(i));
        }
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, M& m, size_t c1, size_t c2) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), m(i), c1, c2);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename M>
    static void upsample_block_3d(A&& in, M& m, size_t i, size_t j, size_t k) {
        auto value = in(i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = value;
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
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename M>
    static void upsample_block_4d(A&& in, M& m, size_t q, size_t i, size_t j, size_t k) {
        auto value = in(q, i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    m(q, i * C1 + ii, j * C2 + jj, k * C3 + kk) = value;
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
    static void upsample_block_3d(A&& in, M& m, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto value = in(i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = value;
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
    static void upsample_block_4d(A&& in, M& m, size_t q, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
        auto value = in(q, i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    m(q, i * c1 + ii, j * c2 + jj, k * c3 + kk) = value;
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
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block_3d<C1, C2, C3>(in, m, i, j, k);
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
    template <typename A, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block_3d(in, m, i, j, k, c1, c2, c3);
                }
            }
        }
    }

    // 4D Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, M&& m) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(in); ++k) {
                            upsample_block_4d<C1, C2, C3>(in, m, q, i, j, k);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(is_4d<A>)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2, size_t c3) {
        // GPU/CPU Synchronization must not be done in parallel
        safe_ensure_cpu_up_to_date(in);
        safe_ensure_cpu_up_to_date(m);

        auto batch_fun = [&](const size_t first, const size_t last) {
            for (size_t q = first; q < last; ++q) {
                for (size_t i = 0; i < etl::dim<1>(in); ++i) {
                    for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                        for (size_t k = 0; k < etl::dim<3>(in); ++k) {
                            upsample_block_4d(in, m, q, i, j, k, c1, c2, c3);
                        }
                    }
                }
            }
        };

        const size_t N = etl::dim<0>(in);

        engine_dispatch_1d_serial(batch_fun, 0, N, 2UL);
    }

    // Deep Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, M& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3>(in(i), m(i));
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
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions() > 4)>
    static void apply(A&& in, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), m(i), c1, c2, c3);
        }
    }
};

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
