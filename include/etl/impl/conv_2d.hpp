//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 2D convolution operations
 */

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief The functor impl for 2D full conv
 */
struct conv2_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv2_full_fft(i, k, c);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv2_full(i, k, c);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_full");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_full");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0) + etl::dim(kernel, 0) - 1, "Invalid dimensions for conv2_full");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1) + etl::dim(kernel, 1) - 1, "Invalid dimensions for conv2_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_full");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_full");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>() + etl::dim<0,K>() - 1, "Invalid dimensions for conv2_full");
        static_assert(etl::dim<1,C>() == etl::dim<1,I>() + etl::dim<1,K>() - 1, "Invalid dimensions for conv2_full");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        return etl::dim(input, d) + etl::dim(kernel, d) - 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>() + etl::dim<D, K>() - 1;
    }
};

/*!
 * \brief The functor impl for 2D full conv
 */
struct conv2_full_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full_flipped(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv2_full_fft_flipped(i, k, c);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv2_full_flipped(i, k, c);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv2_full_flipped(i, k, c);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv2_full_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv2_full_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        return etl::dim(input, d) + etl::dim(kernel, d) - 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>() + etl::dim<D, K>() - 1;
    }
};

/*!
 * \brief The functor impl for 2D same conv
 */
struct conv2_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_same");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_same");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_same");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_same");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_same");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<1,C>() == etl::dim<1,I>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<0,I>() >= etl::dim<0,K>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<1,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_same");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");
        cpp_unused(kernel);

        return etl::dim(input, d);
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>();
    }
};

/*!
 * \brief The functor impl for 2D same conv
 */
struct conv2_same_flipped_impl : conv2_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_same_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same";
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
struct conv2_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid(i, k, c, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        cpp_assert(etl::dim(conv, 0) == (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * P1) / S1 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(conv, 1) == (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * P2) / S2 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        static_assert(etl::dim<0,C>() == (etl::dim<0,I>() - etl::dim<0,K>() + 2 * P1) / S1 + 1, "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<1,C>() == (etl::dim<1,I>() - etl::dim<1,K>() + 2 * P2) / S2 + 1, "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<0,I>() >= etl::dim<0,K>(), "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<1,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        if(d == 0){
            return (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * P1) / S1 + 1;
        } else {
            return (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * P2) / S2 + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return D == 0 ? (etl::dim<D, I>() - etl::dim<D, K>() + 2 * P1) / S1 + 1
                      : (etl::dim<D, I>() - etl::dim<D, K>() + 2 * P2) / S2 + 1;
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
struct conv2_valid_flipped_impl : conv2_valid_impl<S1, S2, P1, P2> {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped(i, k, c, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid_flipped";
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct dyn_conv2_valid_impl {
    size_t s1; ///< The first dimension stride
    size_t s2; ///< The second dimension stride
    size_t p1; ///< The first dimension padding (left and right)
    size_t p2; ///< The second dimension padding (top and bottom)

    /*!
     * \brief Construct a new dyn_conv2_valid_impl
     * \param s1 The first dimension stride
     * \param s2 The second dimension stride
     * \param p1 The first dimension padding (left and right)
     * \param p2 The second dimension padding (top and bottom)
     */
    dyn_conv2_valid_impl(size_t s1, size_t s2, size_t p1, size_t p2) : s1(s1), s2(s2), p1(p1), p2(p2) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    void apply(const I& input, const K& kernel, C& conv) const {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid(i, k, c, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    void check(const I& input, const K& kernel, const C& conv) const {
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        cpp_assert(etl::dim(conv, 0) == (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * p1) / s1 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(conv, 1) == (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * p2) / s2 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    size_t dim(size_t d, const I& input, const K& kernel) const {
        cpp_assert(d < 2, "Invalid dimensions access");

        if(d == 0){
            return (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * p1) / s1 + 1;
        } else {
            return (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * p2) / s2 + 1;
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct dyn_conv2_valid_flipped_impl {
    size_t s1; ///< The first dimension stride
    size_t s2; ///< The second dimension stride
    size_t p1; ///< The first dimension padding (left and right)
    size_t p2; ///< The second dimension padding (top and bottom)

    /*!
     * \brief Construct a new dyn_conv2_valid_impl
     * \param s1 The first dimension stride
     * \param s2 The second dimension stride
     * \param p1 The first dimension padding (left and right)
     * \param p2 The second dimension padding (top and bottom)
     */
    dyn_conv2_valid_flipped_impl(size_t s1, size_t s2, size_t p1, size_t p2) : s1(s1), s2(s2), p1(p1), p2(p2) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    void apply(const I& input, const K& kernel, C& conv) const {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped(i, k, c, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    void check(const I& input, const K& kernel, const C& conv) const {
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        cpp_assert(etl::dim(conv, 0) == (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * p1) / s1 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(conv, 1) == (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * p2) / s2 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    size_t dim(size_t d, const I& input, const K& kernel) const {
        cpp_assert(d < 2, "Invalid dimensions access");

        if(d == 0){
            return (etl::dim(input, 0) - etl::dim(kernel, 0) + 2 * p1) / s1 + 1;
        } else {
            return (etl::dim(input, 1) - etl::dim(kernel, 1) + 2 * p2) / s2 + 1;
        }
    }
};

} //end of namespace detail

} //end of namespace etl
