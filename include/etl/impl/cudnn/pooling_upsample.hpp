//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl::impl::cudnn {

#ifdef ETL_CUDNN_MODE

/*!
 * \brief Apply the functor on sub and store the result in m
 * \param sub The sub expression
 * \param m The storage matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename A, typename B, typename C, typename M>
void unpool_2d(cudnnPoolingMode_t mode, A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
    using type = std::remove_const_t<value_t<A>>;

    decltype(auto) handle = start_cudnn();

    auto pooling_desc = create_pooling_desc_wrapper(mode, c1, c2, c1, c2, 0, 0);

    auto in_tensor     = create_tensor_wrapper(in);
    auto out_tensor    = create_tensor_wrapper(out);
    auto errors_tensor = create_tensor_wrapper(errors);
    auto m_tensor      = create_tensor_wrapper(m);

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    // Allocate GPU memory, if necessary

    in.ensure_gpu_up_to_date();
    out.ensure_gpu_allocated();
    errors.ensure_gpu_up_to_date();
    m.ensure_gpu_allocated();

    // Perform pooling

    cudnn_check(cudnnPoolingBackward(handle.get(), *pooling_desc, alpha, *out_tensor, out.gpu_memory(), *errors_tensor, errors.gpu_memory(), *in_tensor,
                                     in.gpu_memory(), beta, *m_tensor, m.gpu_memory()));

    m.validate_gpu();
    m.invalidate_cpu();
}

/*!
 * \brief Apply the functor on sub and store the result in m
 * \param sub The sub expression
 * \param m The storage matrix
 * \param c1 The first dimension pooling ratio
 * \param c2 The second dimension pooling ratio
 */
template <typename A, typename B, typename C, typename M>
void unpool_3d(cudnnPoolingMode_t mode, A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
    using type = std::remove_const_t<value_t<A>>;

    decltype(auto) handle = start_cudnn();

    auto pooling_desc = create_pooling_desc_wrapper(mode, c1, c2, c3, c1, c2, c3, 0, 0, 0);

    auto in_tensor     = create_tensor_wrapper_5d(in);
    auto out_tensor    = create_tensor_wrapper_5d(out);
    auto errors_tensor = create_tensor_wrapper_5d(errors);
    auto m_tensor      = create_tensor_wrapper_5d(m);

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    // Allocate GPU memory, if necessary

    in.ensure_gpu_up_to_date();
    out.ensure_gpu_allocated();
    errors.ensure_gpu_up_to_date();
    m.ensure_gpu_allocated();

    // Perform pooling

    cudnn_check(cudnnPoolingBackward(handle.get(), *pooling_desc, alpha, *out_tensor, out.gpu_memory(), *errors_tensor, errors.gpu_memory(), *in_tensor,
                                     in.gpu_memory(), beta, *m_tensor, m.gpu_memory()));

    m.validate_gpu();
    m.invalidate_cpu();
}

/*!
 * \brief Functor for 2D Max Pooling Upsample
 */
struct max_pool_upsample_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() < 5)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
        unpool_2d(CUDNN_POOLING_MAX, in, out, errors, m, c1, c2);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2);
        }
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_upsample_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() < 5)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        unpool_3d(CUDNN_POOLING_MAX, in, out, errors, m, c1, c2, c3);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, c3);
        }
    }
};

/*!
 * \brief Functor for 2D Avg Pooling Upsample
 */
struct avg_pool_upsample_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() < 5)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
        unpool_2d(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, in, out, errors, m, c1, c2);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2);
        }
    }
};

/*!
 * \brief Functor for 2D Max Pooling
 */
struct avg_pool_upsample_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions() < 5)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        unpool_3d(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, in, out, errors, m, c1, c2, c3);
    }

    // Deep handling

    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param sub The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M, cpp_enable_iff(decay_traits<A>::dimensions > 4)>
    static void apply(A&& in, B&& out, C&& errors, M& m, size_t c1, size_t c2, size_t c3) {
        for (size_t i = 0; i < etl::dim<0>(in); ++i) {
            apply(in(i), out(i), errors(i), m(i), c1, c2, c3);
        }
    }
};

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Functor for 2D Max Pooling Upsample
 */
struct max_pool_upsample_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void apply([[maybe_unused]] A&& in,
                      [[maybe_unused]] B&& out,
                      [[maybe_unused]] C&& errors,
                      [[maybe_unused]] M& m,
                      [[maybe_unused]] size_t c1,
                      [[maybe_unused]] size_t c2) {
        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 3D Max Pooling Upsample
 */
struct max_pool_upsample_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void apply([[maybe_unused]] A&& in,
                      [[maybe_unused]] B&& out,
                      [[maybe_unused]] C&& errors,
                      [[maybe_unused]] M& m,
                      [[maybe_unused]] size_t c1,
                      [[maybe_unused]] size_t c2,
                      [[maybe_unused]] size_t c3) {
        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 2D Avg Pooling Upsample
 */
struct avg_pool_upsample_2d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void apply([[maybe_unused]] A&& in,
                      [[maybe_unused]] B&& out,
                      [[maybe_unused]] C&& errors,
                      [[maybe_unused]] M& m,
                      [[maybe_unused]] size_t c1,
                      [[maybe_unused]] size_t c2) {
        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

/*!
 * \brief Functor for 3D Avg Pooling Upsample
 */
struct avg_pool_upsample_3d {
    /*!
     * \brief Apply the functor on sub and store the result in m
     * \param in The sub expression
     * \param m The storage matrix
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    template <typename A, typename B, typename C, typename M>
    static void apply([[maybe_unused]] A&& in,
                      [[maybe_unused]] B&& out,
                      [[maybe_unused]] C&& errors,
                      [[maybe_unused]] M& m,
                      [[maybe_unused]] size_t c1,
                      [[maybe_unused]] size_t c2,
                      [[maybe_unused]] size_t c3) {
        cpp_unreachable("Unsupported feature called: cudnn pool");
    }
};

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::cudnn
