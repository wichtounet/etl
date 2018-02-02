//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
    template <size_t C1, size_t C2, typename A, typename B, typename M>
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t j, size_t k) {
        auto max = out(j, k);

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block<C1, C2>(in, out, m, j, k);
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
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t j, size_t k, size_t c1, size_t c2) {
        auto max = out(j, k);

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
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
    template <typename A, typename B, typename M, cpp_enable_iff(is_2d<A>)>
    static void apply(A&& in, B&& out, M&& m, size_t c1, size_t c2, size_t c3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        cpp_unused(c3);

        for (size_t j = 0; j < etl::dim<0>(out); ++j) {
            for (size_t k = 0; k < etl::dim<1>(out); ++k) {
                pool_derivative_block(in, out, m, j, k, c1, c2);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename M, cpp_enable_iff(!is_2d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3>(in(i), out(i), m(i));
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
    static void apply(A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename M>
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename M, cpp_enable_iff(is_3d<A>)>
    static void apply(A&& in, B&& out, M&& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block<C1, C2, C3>(in, out, m, i, j, k);
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
    static void pool_derivative_block(const A& in, const B& out, M& m, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
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
    static void apply(A&& in, B&& out, M&& m, size_t c1, size_t c2, size_t c3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(out); ++i) {
            for (size_t j = 0; j < etl::dim<1>(out); ++j) {
                for (size_t k = 0; k < etl::dim<2>(out); ++k) {
                    pool_derivative_block(in, out, m, i, j, k, c1, c2, c3);
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename M, cpp_enable_iff(!is_3d<A>)>
    static void apply(A&& in, B&& out, M& m) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply<C1, C2, C3>(in(i), out(i), m(i));
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
    static void apply(A&& in, B&& out, M& m, size_t c1, size_t c2, size_t c3) {
        in.ensure_cpu_up_to_date();
        out.ensure_cpu_up_to_date();

        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), m(i), c1, c2, c3);
        }

        m.invalidate_gpu();
        m.validate_cpu();
    }
};

} //end of namespace etl::impl
