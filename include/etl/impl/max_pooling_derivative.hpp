//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl {

// TODO Optimize max pool derivative like max pooling upsampling was optimized

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
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t j, size_t k) {
        auto max = out(j, k);

        // Slow path for cells with padding
        if constexpr (P1 || P2) {
            if (cpp_unlikely(j < P1 || k < P2 || j >= etl::dim<0>(out) - P1 || k >= etl::dim<1>(out) - P2)) {
                const int64_t base_j = j * S1 - P1;
                const int64_t base_k = k * S2 - P2;

                for (size_t jj = 0; jj < C1; ++jj) {
                    for (size_t kk = 0; kk < C2; ++kk) {
                        if (base_j + jj >= 0 && base_k + kk >= 0 && base_j + jj < etl::dim<0>(m) && base_k + kk < etl::dim<1>(m)) {

                            if (max == in(base_j + jj, base_k + kk)) {
                                m(base_j + jj, base_k + kk) = 1.0;
                            } else {
                                m(base_j + jj, base_k + kk) = 0.0;
                            }
                        }
                    }
                }

                return;
            }
        }

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                if constexpr (C1 == S1 && C2 == S2) {
                    if (max == in(j * S1 + jj, k * S2 + kk)) {
                        m(j * S1 - P1 + jj, k * S2 - P2 + kk) = 1.0;
                    } else {
                        m(j * S1 - P1 + jj, k * S2 - P2 + kk) = 0.0;
                    }
                } else {
                    if (max == in(j * S1 + jj, k * S2 + kk)) {
                        m(j * S1 - P1 + jj, k * S2 - P2 + kk) += 1.0;
                    } else {
                        m(j * S1 - P1 + jj, k * S2 - P2 + kk) += 0.0;
                    }
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        if constexpr (C1 != S1 || C2 != S2) {
            m = 0;
        }

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block<C1, C2, S1, S2, P1, P2>(in, out, m, j, k);
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto max = out(j, k);

        // Slow path for cells with padding
        if (cpp_unlikely(p1 || p2)) {
            if (cpp_unlikely(j < p1 || k < p2 || j >= etl::dim<0>(out) - p1 || k >= etl::dim<1>(out) - p2)) {
                const int64_t base_j = j * s1 - p1;
                const int64_t base_k = k * s2 - p2;

                for (size_t jj = 0; jj < c1; ++jj) {
                    for (size_t kk = 0; kk < c2; ++kk) {
                        if (base_j + jj >= 0 && base_k + kk >= 0 && base_j + jj < etl::dim<0>(m) && base_k + kk < etl::dim<1>(m)) {

                            if (max == in(base_j + jj, base_k + kk)) {
                                m(base_j + jj, base_k + kk) = 1.0;
                            } else {
                                m(base_j + jj, base_k + kk) = 0.0;
                            }
                        }
                    }
                }

                return;
            }
        }

        if (c1 == s1 && c2 == s2) {
            for (size_t jj = 0; jj < c1; ++jj) {
                for (size_t kk = 0; kk < c2; ++kk) {
                    if (max == in(j * s1 + jj, k * s2 + kk)) {
                        m(j * s1 - p1 + jj, k * s2 - p2 + kk) = 1.0;
                    } else {
                        m(j * s1 - p1 + jj, k * s2 - p2 + kk) = 0.0;
                    }
                }
            }
        } else {
            for (size_t jj = 0; jj < c1; ++jj) {
                for (size_t kk = 0; kk < c2; ++kk) {
                    if (max == in(j * s1 + jj, k * s2 + kk)) {
                        m(j * s1 - p1 + jj, k * s2 - p2 + kk) += 1.0;
                    } else {
                        m(j * s1 - p1 + jj, k * s2 - p2 + kk) += 0.0;
                    }
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
    template <typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&&                     in,
                      B&&                     out,
                      M&&                     m,
                      size_t                  c1,
                      size_t                  c2,
                      [[maybe_unused]] size_t c3,
                      size_t                  s1,
                      size_t                  s2,
                      [[maybe_unused]] size_t s3,
                      size_t                  p1,
                      size_t                  p2,
                      [[maybe_unused]] size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        if (c1 != s1 || c2 != s2) {
            m = 0;
        }

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block(in, out, m, j, k, c1, c2, s1, s2, p1, p2);
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(!is_2d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in(i), out(i), m(i));
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M, cpp_enable_iff(!is_2d<A>)>
    static void apply(A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t i, size_t j, size_t k) {
        auto max = out(i, j, k);

        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in, out, m, i, j, k);
                }
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    static void pool_derivative_block(const A&                in,
                                      const B&                out,
                                      M&                      m,
                                      size_t                  i,
                                      size_t                  j,
                                      size_t                  k,
                                      size_t                  c1,
                                      size_t                  c2,
                                      size_t                  c3,
                                      [[maybe_unused]] size_t s1,
                                      [[maybe_unused]] size_t s2,
                                      [[maybe_unused]] size_t s3,
                                      [[maybe_unused]] size_t p1,
                                      [[maybe_unused]] size_t p2,
                                      [[maybe_unused]] size_t p3) {
        auto max = out(i, j, k);

        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
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
    template <typename A, typename B, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block(in, out, m, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                }
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(!is_3d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in(i), out(i), m(i));
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename M, cpp_enable_iff(!is_3d<A>)>
    static void apply(
            A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }
};

/*!
 * \brief Functor for the derivative of 2D Avg Pooling
 */
struct avg_pool_derivative_2d {
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
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename A, typename B, typename M>
    static void pool_derivative_block([[maybe_unused]] const A& in, [[maybe_unused]] const B& out, M& m, size_t j, size_t k) {
        // Slow path for cells with padding
        if constexpr (P1 || P2) {
            if (cpp_unlikely(j < P1 || k < P2 || j >= etl::dim<0>(out) - P1 || k >= etl::dim<1>(out) - P2)) {
                const int64_t base_j = j * S1 - P1;
                const int64_t base_k = k * S2 - P2;

                for (size_t jj = 0; jj < C1; ++jj) {
                    for (size_t kk = 0; kk < C2; ++kk) {
                        if (base_j + jj >= 0 && base_k + kk >= 0 && base_j + jj < etl::dim<0>(m) && base_k + kk < etl::dim<1>(m)) {
                            m(base_j + jj, base_k + kk) = 1.0 / (C1 * C2);
                        }
                    }
                }

                return;
            }
        }

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
                if constexpr (C1 == S1 && C2 == S2) {
                    m(j * S1 - P1 + jj, k * S2 - P2 + kk) = 1.0 / (C1 * C2);
                } else {
                    m(j * S1 - P1 + jj, k * S2 - P2 + kk) += 1.0 / (C1 * C2);
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        if constexpr (C1 != S1 || C2 != S2) {
            m = 0;
        }

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block<C1, C2, S1, S2, P1, P2>(in, out, m, j, k);
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    static void pool_derivative_block(
            [[maybe_unused]] const A& in, [[maybe_unused]] const B& out, M& m, size_t j, size_t k, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        // Slow path for cells with padding
        if (cpp_unlikely(p1 || p2)) {
            if (cpp_unlikely(j < p1 || k < p2 || j >= etl::dim<0>(out) - p1 || k >= etl::dim<1>(out) - p2)) {
                const int64_t base_j = j * s1 - p1;
                const int64_t base_k = k * s2 - p2;

                for (size_t jj = 0; jj < c1; ++jj) {
                    for (size_t kk = 0; kk < c2; ++kk) {
                        if (base_j + jj >= 0 && base_k + kk >= 0 && base_j + jj < etl::dim<0>(m) && base_k + kk < etl::dim<1>(m)) {
                            m(base_j + jj, base_k + kk) = 1.0 / (c1 * c2);
                        }
                    }
                }

                return;
            }
        }

        if (c1 == s1 && c2 == s2) {
            for (size_t jj = 0; jj < c1; ++jj) {
                for (size_t kk = 0; kk < c2; ++kk) {
                    m(j * s1 - p1 + jj, k * s2 - p2 + kk) = 1.0 / (c1 * c2);
                }
            }
        } else {
            for (size_t jj = 0; jj < c1; ++jj) {
                for (size_t kk = 0; kk < c2; ++kk) {
                    m(j * s1 - p1 + jj, k * s2 - p2 + kk) += 1.0 / (c1 * c2);
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
    template <typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, M&& m, size_t c1, size_t c2, [[maybe_unused]] size_t c3, size_t s1, size_t s2, [[maybe_unused]] size_t s3, size_t p1, size_t p2, [[maybe_unused]] size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        if (c1 != s1 || c2 != s2) {
            m = 0;
        }

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block(in, out, m, j, k, c1, c2, s1, s2, p1, p2);
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(!is_2d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in(i), out(i), m(i));
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param out The out matrix
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename M, cpp_enable_iff(!is_2d<A>)>
    static void apply(A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }
};

/*!
 * \brief Functor for the derivative of 3D Avg Pooling
 */
struct avg_pool_derivative_3d {
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M>
    static void pool_derivative_block([[maybe_unused]] const A& in, [[maybe_unused]] const B& out, M& m, size_t i, size_t j, size_t k) {
        for (size_t ii = 0; ii < C1; ++ii) {
            for (size_t jj = 0; jj < C2; ++jj) {
                for (size_t kk = 0; kk < C3; ++kk) {
                    m(i * C1 + ii, j * C2 + jj, k * C3 + kk) = 1.0 / (C1 * C2 * C3);
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in, out, m, i, j, k);
                }
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    static void pool_derivative_block([[maybe_unused]] const A& in,
                                      [[maybe_unused]] const B& out,
                                      M&                        m,
                                      size_t                    i,
                                      size_t                    j,
                                      size_t                    k,
                                      size_t                    c1,
                                      size_t                    c2,
                                      size_t                    c3,
                                      [[maybe_unused]] size_t   s1,
                                      [[maybe_unused]] size_t   s2,
                                      [[maybe_unused]] size_t   s3,
                                      [[maybe_unused]] size_t   p1,
                                      [[maybe_unused]] size_t   p2,
                                      [[maybe_unused]] size_t   p3) {
        for (size_t ii = 0; ii < c1; ++ii) {
            for (size_t jj = 0; jj < c2; ++jj) {
                for (size_t kk = 0; kk < c3; ++kk) {
                    m(i * c1 + ii, j * c2 + jj, k * c3 + kk) = 1.0 / (c1 * c2 * c3);
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
    template <typename A, typename B, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block(in, out, m, i, j, k, c1, c2, c3, s1, s2, s3, p1, p2, p3);
                }
            }
        }

        m.invalidate_gpu();
        m.validate_cpu();
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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename A, typename B, typename M, cpp_enable_iff(!is_3d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(in(i), out(i), m(i));
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename B, typename M, cpp_enable_iff(!is_3d<A>)>
    static void apply(
            A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }
};

} //end of namespace etl::impl
