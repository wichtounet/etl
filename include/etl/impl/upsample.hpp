//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

// TODO Optimize upsampling like max pooling upsampling was optimized

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
    static void upsample_block(A&& in, M& m, size_t j, size_t k) {
        auto value = in(j, k);

        for (size_t jj = 0; jj < C1; ++jj) {
            for (size_t kk = 0; kk < C2; ++kk) {
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
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_if(is_2d<A>::value)>
    static void apply(A&& in, M&& m) {
        for (size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (size_t k = 0; k < etl::dim<1>(in); ++k) {
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
    static void upsample_block(A&& in, M& m, size_t j, size_t k, size_t c1, size_t c2) {
        auto value = in(j, k);

        for (size_t jj = 0; jj < c1; ++jj) {
            for (size_t kk = 0; kk < c2; ++kk) {
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
    template <typename A, typename M, cpp_enable_if(is_2d<A>::value)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2) {
        for (size_t j = 0; j < etl::dim<0>(in); ++j) {
            for (size_t k = 0; k < etl::dim<1>(in); ++k) {
                upsample_block(in, m, j, k, c1, c2);
            }
        }
    }

    // Deep Handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     */
    template <size_t C1, size_t C2, typename A, typename M, cpp_enable_if(!is_2d<A>::value)>
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
    template <typename A, typename M, cpp_enable_if(!is_2d<A>::value)>
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
    static void upsample_block(A&& in, M& m, size_t i, size_t j, size_t k) {
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
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     * \tparam C3 The third dimension pooling ratio
     */
    template <size_t C1, size_t C2, size_t C3, typename A, typename M, cpp_enable_if(is_3d<A>::value)>
    static void apply(A&& in, M&& m) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (size_t k = 0; k < etl::dim<2>(in); ++k) {
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
    static void upsample_block(A&& in, M& m, size_t i, size_t j, size_t k, size_t c1, size_t c2, size_t c3) {
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
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_if(is_3d<A>::value)>
    static void apply(A&& in, M&& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            for (size_t j = 0; j < etl::dim<1>(in); ++j) {
                for (size_t k = 0; k < etl::dim<2>(in); ++k) {
                    upsample_block(in, m, i, j, k, c1, c2, c3);
                }
            }
        }
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
    template <size_t C1, size_t C2, size_t C3, typename A, typename M, cpp_enable_if(!is_3d<A>::value)>
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
    template <typename A, typename M, cpp_enable_if(!is_3d<A>::value)>
    static void apply(A&& in, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), m(i), c1, c2, c3);
        }
    }
};

} //end of namespace impl

} //end of namespace etl
