//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 4D convolution operations
 */

namespace etl {

namespace detail {

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
struct conv4_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(conv, 1) == etl::dim(kernel, 0), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 1) == etl::dim(kernel, 1), "Invalid dimensions for conv4_valid");

        cpp_assert(etl::dim(conv, 2) == (etl::dim(input, 2) - etl::dim(kernel, 2) + 2 * P1) / S1 + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(conv, 3) == (etl::dim(input, 3) - etl::dim(kernel, 3) + 2 * P2) / S2 + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 2) >= etl::dim(kernel, 2), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 3) >= etl::dim(kernel, 3), "Invalid dimensions for conv4_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<1,C>() == etl::dim<0,K>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<1,I>() == etl::dim<1,K>(), "Invalid dimensions for conv4_valid");

        static_assert(etl::dim<2,C>() == (etl::dim<2,I>() - etl::dim<2,K>() + 2 * P1) / S1 + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,C>() == (etl::dim<3,I>() - etl::dim<3,K>() + 2 * P2) / S2 + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<2,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,I>() >= etl::dim<3,K>(), "Invalid dimensions for conv4_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(input, 0);
        } else if(d == 1){
            return etl::dim(kernel, 0);
        } else if(d == 2){
            return (etl::dim(input, d) - etl::dim(kernel, d) + 2 * P1) / S1 + 1;
        } else {
            return (etl::dim(input, d) - etl::dim(kernel, d) + 2 * P2) / S2 + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<0,K>()
            :  D == 2 ? (etl::dim<D,I>() - etl::dim<D,K>() + 2 * P1) / S1+ 1
            : (etl::dim<D,I>() - etl::dim<D,K>() + 2 * P2) / S2+ 1;
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
struct conv4_valid_flipped_impl : conv4_valid_impl<S1, S2, P1, P2> {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_flipped";
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_filter_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_filter";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        cpp_assert(etl::dim(conv, 0) == etl::dim(kernel, 1), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 0) == etl::dim(kernel, 0), "Invalid dimensions for conv4_valid");

        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) - etl::dim(kernel, 2) + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(conv, 3) == etl::dim(input, 3) - etl::dim(kernel, 3) + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 2) >= etl::dim(kernel, 2), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 3) >= etl::dim(kernel, 3), "Invalid dimensions for conv4_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        static_assert(etl::dim<0,C>() == etl::dim<1,K>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<1,C>() == etl::dim<1,I>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<0,I>() == etl::dim<0,K>(), "Invalid dimensions for conv4_valid");

        static_assert(etl::dim<2,C>() == etl::dim<2,I>() - etl::dim<2,K>() + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,C>() == etl::dim<3,I>() - etl::dim<3,K>() + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<2,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,I>() >= etl::dim<3,K>(), "Invalid dimensions for conv4_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 1);
        } else if(d == 1){
            return etl::dim(input, 1);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<1,K>()
            :  D == 1 ? etl::dim<1,I>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_filter_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_filter_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_valid_filter_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_valid_filter_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 1);
        } else if(d == 1){
            return etl::dim(input, 1);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<1,K>()
            :  D == 1 ? etl::dim<1,I>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
    }
};

/*!
 * \brief The functor impl for 4D full conv
 */
struct conv4_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_STD) {
            impl::standard::conv4_full_fft(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_MKL) {
            impl::blas::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_CUFFT) {
            impl::cufft::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_full");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_full");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_full");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(conv, 1) == etl::dim(kernel, 1), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 1) == etl::dim(kernel, 0), "Invalid dimensions for conv4_full");

        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) + etl::dim(kernel, 2) - 1, "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(conv, 3) == etl::dim(input, 3) + etl::dim(kernel, 3) - 1, "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 2) >= etl::dim(kernel, 2), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 3) >= etl::dim(kernel, 3), "Invalid dimensions for conv4_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_full");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_full");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_full");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<1,C>() == etl::dim<1,K>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<1,I>() == etl::dim<0,K>(), "Invalid dimensions for conv4_full");

        static_assert(etl::dim<2,C>() == etl::dim<2,I>() + etl::dim<2,K>() - 1, "Invalid dimensions for conv4_full");
        static_assert(etl::dim<3,C>() == etl::dim<3,I>() + etl::dim<3,K>() - 1, "Invalid dimensions for conv4_full");
        static_assert(etl::dim<2,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<3,I>() >= etl::dim<3,K>(), "Invalid dimensions for conv4_full");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(input, 0);
        } else if(d == 1){
            return etl::dim(kernel, 1);
        } else {
            return etl::dim(input, d) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<1,K>()
            : etl::dim<D,I>() + etl::dim<D,K>() - 1;
    }
};

/*!
 * \brief The functor impl for 4D full conv
 */
struct conv4_full_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_full_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_full_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_full_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(input, 0);
        } else if(d == 1){
            return etl::dim(kernel, 1);
        } else {
            return etl::dim(input, d) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<1,K>()
            : etl::dim<D,I>() + etl::dim<D,K>() - 1;
    }
};

} //end of namespace detail

} //end of namespace etl
