//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl {

namespace impl {

namespace cudnn {

#ifdef ETL_CUDNN_MODE

/*!
 * \brief Apply the functor on sub and store the result in m
 * \param sub The sub expression
 * \param m The storage matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename X, typename Y>
void pool_2d(cudnnPoolingMode_t mode, const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
    using type = std::remove_const_t<value_t<X>>;

    decltype(auto) handle = start_cudnn();

    auto pooling_desc = create_pooling_desc_wrapper(mode, c1, c2, s1, s2, p1, p2);

    auto x_tensor = create_tensor_wrapper(x);
    auto y_tensor = create_tensor_wrapper(y);

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // Perform pooling

    cudnn_check(cudnnPoolingForward(handle.get(), *pooling_desc, alpha, *x_tensor, x.gpu_memory(), beta, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();
}

/*!
 * \brief Apply the functor on sub and store the result in m
 * \param sub The sub expression
 * \param m The storage matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename X, typename Y>
void pool_3d(cudnnPoolingMode_t mode, const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
    using type = std::remove_const_t<value_t<X>>;

    decltype(auto) handle = start_cudnn();

    auto pooling_desc = create_pooling_desc_wrapper(mode, c1, c2, c3, s1, s2, s3, p1, p2, p3);

    auto x_tensor = create_tensor_wrapper_5d(x);
    auto y_tensor = create_tensor_wrapper_5d(y);

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // Perform pooling

    cudnn_check(cudnnPoolingForward(handle.get(), *pooling_desc, alpha, *x_tensor, x.gpu_memory(), beta, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();
}

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y, cpp_enable_iff(decay_traits<X>::dimensions() < 5)>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        pool_2d(CUDNN_POOLING_MAX, x, y, c1, c2, s1, s2, p1, p2);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        for(size_t i = 0; i < etl::dim<0>(sub); ++i){
            apply(sub(i), m(i), c1, c2, s1, s2, p1, p2);
        }
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct avg_pool_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y, cpp_enable_iff(decay_traits<X>::dimensions() < 5)>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        pool_2d(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, x, y, c1, c2, s1, s2, p1, p2);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        for(size_t i = 0; i < etl::dim<0>(sub); ++i){
            apply(sub(i), m(i), c1, c2, s1, s2, p1, p2);
        }
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y, cpp_enable_iff(decay_traits<X>::dimensions() < 5)>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        pool_3d(CUDNN_POOLING_MAX, x, y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        for(size_t i = 0; i < etl::dim<0>(sub); ++i){
            apply(sub(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct avg_pool_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y, cpp_enable_iff(decay_traits<X>::dimensions() < 5)>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        pool_3d(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, x, y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(const A& sub, M&& m, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        for(size_t i = 0; i < etl::dim<0>(sub); ++i){
            apply(sub(i), m(i), c1, c2, c3, s1, s2, s3, p1, p2, p3);
        }
    }
};

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param x The sub expression
     * \param y The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        cpp_unused(x);
        cpp_unused(y);
        cpp_unused(c1);
        cpp_unused(c2);
        cpp_unused(s1);
        cpp_unused(s2);
        cpp_unused(p1);
        cpp_unused(p2);

        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct avg_pool_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param x The sub expression
     * \param y The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        cpp_unused(x);
        cpp_unused(y);
        cpp_unused(c1);
        cpp_unused(c2);
        cpp_unused(s1);
        cpp_unused(s2);
        cpp_unused(p1);
        cpp_unused(p2);

        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param x The sub expression
     * \param y The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        cpp_unused(x);
        cpp_unused(y);
        cpp_unused(c1);
        cpp_unused(c2);
        cpp_unused(c3);
        cpp_unused(s1);
        cpp_unused(s2);
        cpp_unused(s3);
        cpp_unused(p1);
        cpp_unused(p2);
        cpp_unused(p3);

        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct avg_pool_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param x The sub expression
     * \param y The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename X, typename Y>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        cpp_unused(x);
        cpp_unused(y);
        cpp_unused(c1);
        cpp_unused(c2);
        cpp_unused(c3);
        cpp_unused(s1);
        cpp_unused(s2);
        cpp_unused(s3);
        cpp_unused(p1);
        cpp_unused(p2);
        cpp_unused(p3);

        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
