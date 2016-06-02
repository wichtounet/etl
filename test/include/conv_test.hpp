//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_VECTORIZE_IMPL
#ifdef __AVX__
#define TEST_AVX
#endif
#ifdef __SSE3__
#define TEST_SSE
#endif
#endif

#ifdef ETL_CUDNN_MODE
#define TEST_CUDNN
#endif

#define CONV_FUNCTOR(name, ...)                       \
    struct name {                                     \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C&& c) {      \
            __VA_ARGS__;                              \
        }                                             \
    };

CONV_FUNCTOR(default_conv1_full, c = etl::conv_1d_full(a, b))
CONV_FUNCTOR(std_conv1_full, c = selected_helper(etl::conv_impl::STD, etl::conv_1d_full(a, b)))
CONV_FUNCTOR(fft_std_conv1_full, c = selected_helper(etl::conv_impl::FFT_STD, etl::conv_1d_full(a, b)))

CONV_FUNCTOR(default_conv1_same, c = etl::conv_1d_same(a, b))
CONV_FUNCTOR(std_conv1_same, c = selected_helper(etl::conv_impl::STD, etl::conv_1d_same(a, b)))

CONV_FUNCTOR(default_conv1_valid, c = etl::conv_1d_valid(a, b))
CONV_FUNCTOR(std_conv1_valid, c = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)))

CONV_FUNCTOR(default_conv2_full, c = etl::conv_2d_full(a, b))
CONV_FUNCTOR(std_conv2_full, c = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)))
CONV_FUNCTOR(fft_std_conv2_full, c = selected_helper(etl::conv_impl::FFT_STD, etl::conv_2d_full(a, b)))

CONV_FUNCTOR(default_conv2_full_flipped, c = etl::conv_2d_full_flipped(a, b))
CONV_FUNCTOR(std_conv2_full_flipped, etl::impl::standard::conv2_full_flipped(a, b, c))

CONV_FUNCTOR(default_conv2_same, c = etl::conv_2d_same(a, b))
CONV_FUNCTOR(std_conv2_same, etl::impl::standard::conv2_same(a, b, c))

CONV_FUNCTOR(default_conv2_valid, c = etl::conv_2d_valid(a, b))
CONV_FUNCTOR(std_conv2_valid, etl::impl::standard::conv2_valid(a, b, c))

CONV_FUNCTOR(default_conv2_valid_flipped, c = etl::conv_2d_valid_flipped(a, b))
CONV_FUNCTOR(std_conv2_valid_flipped, etl::impl::standard::conv2_valid_flipped(a, b, c))

CONV_FUNCTOR(default_conv4_valid, c = etl::conv_4d_valid(a, b))
CONV_FUNCTOR(std_conv4_valid, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid(a, b)))

CONV_FUNCTOR(default_conv4_valid_flipped, c = etl::conv_4d_valid_flipped(a, b))
CONV_FUNCTOR(std_conv4_valid_flipped, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid_flipped(a, b)))

CONV_FUNCTOR(default_conv4_valid_filter, c = etl::conv_4d_valid_filter(a, b))
CONV_FUNCTOR(std_conv4_valid_filter, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid_filter(a, b)))

CONV_FUNCTOR(default_conv4_valid_filter_flipped, c = etl::conv_4d_valid_filter_flipped(a, b))
CONV_FUNCTOR(std_conv4_valid_filter_flipped, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid_filter_flipped(a, b)))

CONV_FUNCTOR(default_conv4_full, c = etl::conv_4d_full(a, b))
CONV_FUNCTOR(std_conv4_full, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_full(a, b)))

CONV_FUNCTOR(default_conv4_full_flipped, c = etl::conv_4d_full_flipped(a, b))
CONV_FUNCTOR(std_conv4_full_flipped, c = selected_helper(etl::conv4_impl::STD, etl::conv_4d_full_flipped(a, b)))

CONV_FUNCTOR(default_conv2_valid_multi, c = etl::conv_2d_valid_multi(a, b))
CONV_FUNCTOR(std_conv2_valid_multi, c = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi(a, b)))
CONV_FUNCTOR(fft_conv2_valid_multi, c = selected_helper(etl::conv_multi_impl::FFT, etl::conv_2d_valid_multi(a, b)))
CONV_FUNCTOR(blas_conv2_valid_multi, c = selected_helper(etl::conv_multi_impl::BLAS, etl::conv_2d_valid_multi(a, b)))

CONV_FUNCTOR(default_conv2_valid_multi_flipped, c = etl::conv_2d_valid_multi_flipped(a, b))
CONV_FUNCTOR(std_conv2_valid_multi_flipped, c = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi_flipped(a, b)))
CONV_FUNCTOR(fft_conv2_valid_multi_flipped, c = selected_helper(etl::conv_multi_impl::FFT, etl::conv_2d_valid_multi_flipped(a, b)))
CONV_FUNCTOR(blas_conv2_valid_multi_flipped, c = selected_helper(etl::conv_multi_impl::BLAS, etl::conv_2d_valid_multi_flipped(a, b)))

CONV_FUNCTOR(default_conv2_full_multi, c = etl::conv_2d_full_multi(a, b))
CONV_FUNCTOR(std_conv2_full_multi, c = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_full_multi(a, b)))

CONV_FUNCTOR(default_conv2_full_multi_flipped, c = etl::conv_2d_full_multi_flipped(a, b))
CONV_FUNCTOR(std_conv2_full_multi_flipped, c = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_full_multi_flipped(a, b)))

#define CONV1_FULL_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv1_full)
#define CONV1_FULL_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv1_full)
#define CONV1_FULL_TEST_CASE_SECTION_FFT_STD CONV_TEST_CASE_SECTIONS(fft_std_conv1_full)

#define CONV1_SAME_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv1_same)
#define CONV1_SAME_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv1_same)

#define CONV1_VALID_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv1_valid)
#define CONV1_VALID_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv1_valid)

#define CONV2_FULL_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_full)
#define CONV2_FULL_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_full)
#define CONV2_FULL_TEST_CASE_SECTION_FFT_STD CONV_TEST_CASE_SECTIONS(fft_std_conv2_full)

#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_full_flipped)
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_full_flipped)

#define CONV2_SAME_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_same)
#define CONV2_SAME_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_same)

#define CONV2_VALID_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_valid)
#define CONV2_VALID_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_valid)

#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_valid_flipped)
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_valid_flipped)

#define CONV4_VALID_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_valid)
#define CONV4_VALID_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_valid)

#define CONV4_VALID_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_valid)
#define CONV4_VALID_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_valid)

#define CONV4_VALID_FILTER_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_valid_filter)
#define CONV4_VALID_FILTER_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_valid_filter)

#define CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_valid_filter_flipped)
#define CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_valid_filter_flipped)

#define CONV4_FULL_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_full)
#define CONV4_FULL_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_full)

#define CONV4_FULL_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv4_full_flipped)
#define CONV4_FULL_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv4_full_flipped)

#define CONV2_VALID_MULTI_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_valid_multi)
#define CONV2_VALID_MULTI_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_valid_multi)
#define CONV2_VALID_MULTI_TEST_CASE_SECTION_FFT CONV_TEST_CASE_SECTIONS(fft_conv2_valid_multi)
#define CONV2_VALID_MULTI_TEST_CASE_SECTION_BLAS CONV_TEST_CASE_SECTIONS(blas_conv2_valid_multi)

#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_valid_multi_flipped)
#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_valid_multi_flipped)
#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_FFT CONV_TEST_CASE_SECTIONS(fft_conv2_valid_multi_flipped)
#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_BLAS CONV_TEST_CASE_SECTIONS(blas_conv2_valid_multi_flipped)

#define CONV2_FULL_MULTI_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_full_multi)
#define CONV2_FULL_MULTI_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_full_multi)

#define CONV2_FULL_MULTI_FLIPPED_TEST_CASE_SECTION_DEFAULT CONV_TEST_CASE_SECTIONS(default_conv2_full_multi_flipped)
#define CONV2_FULL_MULTI_FLIPPED_TEST_CASE_SECTION_STD CONV_TEST_CASE_SECTIONS(std_conv2_full_multi_flipped)

#ifdef ETL_MKL_MODE
CONV_FUNCTOR(fft_mkl_conv1_full, c = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_1d_full(a, b)))
CONV_FUNCTOR(fft_mkl_conv2_full, c = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)))
#define CONV1_FULL_TEST_CASE_SECTION_FFT_MKL CONV_TEST_CASE_SECTIONS(fft_mkl_conv1_full)
#define CONV2_FULL_TEST_CASE_SECTION_FFT_MKL CONV_TEST_CASE_SECTIONS(fft_mkl_conv2_full)
#else
#define CONV1_FULL_TEST_CASE_SECTION_FFT_MKL
#define CONV2_FULL_TEST_CASE_SECTION_FFT_MKL
#endif

#ifdef ETL_CUFFT_MODE
CONV_FUNCTOR(fft_cufft_conv1_full, c = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_1d_full(a, b)))
CONV_FUNCTOR(fft_cufft_conv2_full, c = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_2d_full(a, b)))
#define CONV1_FULL_TEST_CASE_SECTION_FFT_CUFFT CONV_TEST_CASE_SECTIONS(fft_cufft_conv1_full)
#define CONV2_FULL_TEST_CASE_SECTION_FFT_CUFFT CONV_TEST_CASE_SECTIONS(fft_cufft_conv2_full)
#else
#define CONV1_FULL_TEST_CASE_SECTION_FFT_CUFFT
#define CONV2_FULL_TEST_CASE_SECTION_FFT_CUFFT
#endif

#ifdef TEST_SSE
CONV_FUNCTOR(sse_conv1_full, c = selected_helper(etl::conv_impl::SSE, etl::conv_1d_full(a, b)))
CONV_FUNCTOR(sse_conv1_same, c = selected_helper(etl::conv_impl::SSE, etl::conv_1d_same(a, b)))
CONV_FUNCTOR(sse_conv1_valid, c = selected_helper(etl::conv_impl::SSE, etl::conv_1d_valid(a, b)))

CONV_FUNCTOR(sse_conv2_full, c = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)))
CONV_FUNCTOR(sse_conv2_full_flipped, c = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full_flipped(a, b)))
CONV_FUNCTOR(sse_conv2_same, c = selected_helper(etl::conv_impl::SSE, etl::conv_2d_same(a, b)))
CONV_FUNCTOR(sse_conv2_valid, c = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)))
CONV_FUNCTOR(sse_conv2_valid_flipped, c = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid_flipped(a, b)))

#define CONV1_FULL_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv1_full)
#define CONV1_SAME_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv1_same)
#define CONV1_VALID_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv1_valid)
#define CONV2_FULL_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv2_full)
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv2_full_flipped)
#define CONV2_SAME_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv2_same)
#define CONV2_VALID_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv2_valid)
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_SSE CONV_TEST_CASE_SECTIONS(sse_conv2_valid_flipped)
#else
#define CONV1_FULL_TEST_CASE_SECTION_SSE
#define CONV1_SAME_TEST_CASE_SECTION_SSE
#define CONV1_VALID_TEST_CASE_SECTION_SSE
#define CONV2_FULL_TEST_CASE_SECTION_SSE
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_SSE
#define CONV2_SAME_TEST_CASE_SECTION_SSE
#define CONV2_VALID_TEST_CASE_SECTION_SSE
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_SSE
#endif

#ifdef TEST_AVX
CONV_FUNCTOR(avx_conv1_full, c = selected_helper(etl::conv_impl::AVX, etl::conv_1d_full(a, b)))
CONV_FUNCTOR(avx_conv1_same, c = selected_helper(etl::conv_impl::AVX, etl::conv_1d_same(a, b)))
CONV_FUNCTOR(avx_conv1_valid, c = selected_helper(etl::conv_impl::AVX, etl::conv_1d_valid(a, b)))

CONV_FUNCTOR(avx_conv2_full, c = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)))
CONV_FUNCTOR(avx_conv2_full_flipped, c = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full_flipped(a, b)))
CONV_FUNCTOR(avx_conv2_same, c = selected_helper(etl::conv_impl::AVX, etl::conv_2d_same(a, b)))
CONV_FUNCTOR(avx_conv2_valid, c = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)))
CONV_FUNCTOR(avx_conv2_valid_flipped, c = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid_flipped(a, b)))

#define CONV1_FULL_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv1_full)
#define CONV1_SAME_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv1_same)
#define CONV1_VALID_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv1_valid)
#define CONV2_FULL_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv2_full)
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv2_full_flipped)
#define CONV2_SAME_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv2_same)
#define CONV2_VALID_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv2_valid)
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_AVX CONV_TEST_CASE_SECTIONS(avx_conv2_valid_flipped)
#else
#define CONV1_FULL_TEST_CASE_SECTION_AVX
#define CONV1_SAME_TEST_CASE_SECTION_AVX
#define CONV1_VALID_TEST_CASE_SECTION_AVX
#define CONV2_FULL_TEST_CASE_SECTION_AVX
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_AVX
#define CONV2_SAME_TEST_CASE_SECTION_AVX
#define CONV2_VALID_TEST_CASE_SECTION_AVX
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_AVX
#endif

#ifdef TEST_CUDNN
CONV_FUNCTOR(cudnn_conv2_full, c = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full(a, b)))
CONV_FUNCTOR(cudnn_conv2_full_flipped, c = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full_flipped(a, b)))
CONV_FUNCTOR(cudnn_conv2_valid, c = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_valid(a, b)))
CONV_FUNCTOR(cudnn_conv2_valid_flipped, c = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_valid_flipped(a, b)))
CONV_FUNCTOR(cudnn_conv4_valid, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid(a, b)))
CONV_FUNCTOR(cudnn_conv4_valid_flipped, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_flipped(a, b)))
CONV_FUNCTOR(cudnn_conv4_valid_filter, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_filter(a, b)))
CONV_FUNCTOR(cudnn_conv4_valid_filter_flipped, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_filter_flipped(a, b)))
CONV_FUNCTOR(cudnn_conv4_full, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full(a, b)))
CONV_FUNCTOR(cudnn_conv4_full_flipped, c = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full_flipped(a, b)))

CONV_FUNCTOR(cudnn_conv2_valid_multi, c = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi(a, b)))
CONV_FUNCTOR(cudnn_conv2_valid_multi_flipped, c = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi_flipped(a, b)))

#define CONV2_FULL_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_full)
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_full_flipped)
#define CONV2_VALID_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_valid)
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_valid_flipped)
#define CONV2_VALID_MULTI_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_valid_multi)
#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv2_valid_multi_flipped)
#define CONV4_VALID_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_valid)
#define CONV4_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_valid_flipped)
#define CONV4_VALID_FILTER_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_valid_filter)
#define CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_valid_filter_flipped)
#define CONV4_FULL_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_full)
#define CONV4_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN CONV_TEST_CASE_SECTIONS(cudnn_conv4_full_flipped)
#else
#define CONV2_FULL_TEST_CASE_SECTION_CUDNN
#define CONV2_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN
#define CONV2_VALID_TEST_CASE_SECTION_CUDNN
#define CONV2_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN
#define CONV2_VALID_MULTI_TEST_CASE_SECTION_CUDNN
#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_CUDNN
#define CONV4_VALID_TEST_CASE_SECTION_CUDNN
#define CONV4_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN
#define CONV4_VALID_FILTER_TEST_CASE_SECTION_CUDNN
#define CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_CUDNN
#define CONV4_FULL_TEST_CASE_SECTION_CUDNN
#define CONV4_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN
#endif

#define CONV_TEST_CASE_DECL(name, description)                                                \
    template <typename T, typename Impl>                                                      \
    static void INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    TEST_CASE(name, description)

#define CONV_TEST_CASE_SECTION(Tn, Impln)                                                        \
    SECTION(#Tn "_" #Impln) {                                                                    \
        INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define CONV_TEST_CASE_DEFN              \
    template <typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define CONV_TEST_CASE_SECTIONS(S1) \
    CONV_TEST_CASE_SECTION(float, S1)   \
    CONV_TEST_CASE_SECTION(double, S1)

#define CONV1_FULL_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {    \
        CONV1_FULL_TEST_CASE_SECTION_DEFAULT    \
        CONV1_FULL_TEST_CASE_SECTION_STD        \
        CONV1_FULL_TEST_CASE_SECTION_SSE        \
        CONV1_FULL_TEST_CASE_SECTION_AVX        \
        CONV1_FULL_TEST_CASE_SECTION_FFT_STD    \
        CONV1_FULL_TEST_CASE_SECTION_FFT_MKL    \
        CONV1_FULL_TEST_CASE_SECTION_FFT_CUFFT  \
    }                                           \
    CONV_TEST_CASE_DEFN

#define CONV1_SAME_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {    \
        CONV1_SAME_TEST_CASE_SECTION_DEFAULT    \
        CONV1_SAME_TEST_CASE_SECTION_STD        \
        CONV1_SAME_TEST_CASE_SECTION_SSE        \
        CONV1_SAME_TEST_CASE_SECTION_AVX        \
    }                                           \
    CONV_TEST_CASE_DEFN

#define CONV1_VALID_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV1_VALID_TEST_CASE_SECTION_DEFAULT    \
        CONV1_VALID_TEST_CASE_SECTION_STD        \
        CONV1_VALID_TEST_CASE_SECTION_SSE        \
        CONV1_VALID_TEST_CASE_SECTION_AVX        \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {    \
        CONV2_FULL_TEST_CASE_SECTION_DEFAULT    \
        CONV2_FULL_TEST_CASE_SECTION_STD        \
        CONV2_FULL_TEST_CASE_SECTION_SSE        \
        CONV2_FULL_TEST_CASE_SECTION_AVX        \
        CONV2_FULL_TEST_CASE_SECTION_FFT_STD    \
        CONV2_FULL_TEST_CASE_SECTION_FFT_MKL    \
        CONV2_FULL_TEST_CASE_SECTION_FFT_CUFFT  \
        CONV2_FULL_TEST_CASE_SECTION_CUDNN      \
    }                                           \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {    \
        CONV2_FULL_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV2_FULL_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV2_FULL_FLIPPED_TEST_CASE_SECTION_SSE        \
        CONV2_FULL_FLIPPED_TEST_CASE_SECTION_AVX        \
        CONV2_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                           \
    CONV_TEST_CASE_DEFN

//Column major version
#define CONV2_FULL_TEST_CASE_CM(name, description) \
    CONV_TEST_CASE_DECL(name, description) {       \
        CONV2_FULL_TEST_CASE_SECTION_DEFAULT       \
        CONV2_FULL_TEST_CASE_SECTION_STD           \
    }                                              \
    CONV_TEST_CASE_DEFN

#define CONV2_SAME_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {    \
        CONV2_SAME_TEST_CASE_SECTION_DEFAULT    \
        CONV2_SAME_TEST_CASE_SECTION_STD        \
        CONV2_SAME_TEST_CASE_SECTION_SSE        \
        CONV2_SAME_TEST_CASE_SECTION_AVX        \
    }                                           \
    CONV_TEST_CASE_DEFN

#define CONV2_VALID_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV2_VALID_TEST_CASE_SECTION_DEFAULT    \
        CONV2_VALID_TEST_CASE_SECTION_STD        \
        CONV2_VALID_TEST_CASE_SECTION_SSE        \
        CONV2_VALID_TEST_CASE_SECTION_AVX        \
        CONV2_VALID_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV2_VALID_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV2_VALID_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV2_VALID_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV2_VALID_FLIPPED_TEST_CASE_SECTION_SSE        \
        CONV2_VALID_FLIPPED_TEST_CASE_SECTION_AVX        \
        CONV2_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_VALID_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_VALID_TEST_CASE_SECTION_DEFAULT    \
        CONV4_VALID_TEST_CASE_SECTION_STD        \
        CONV4_VALID_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_VALID_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_VALID_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV4_VALID_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV4_VALID_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_VALID_FILTER_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_VALID_FILTER_TEST_CASE_SECTION_DEFAULT    \
        CONV4_VALID_FILTER_TEST_CASE_SECTION_STD        \
        CONV4_VALID_FILTER_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_VALID_FILTER_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV4_VALID_FILTER_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_FULL_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_FULL_TEST_CASE_SECTION_DEFAULT    \
        CONV4_FULL_TEST_CASE_SECTION_STD        \
        CONV4_FULL_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV4_FULL_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {     \
        CONV4_FULL_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV4_FULL_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV4_FULL_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    CONV_TEST_CASE_DEFN

#define CONV2_VALID_MULTI_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {           \
        CONV2_VALID_MULTI_TEST_CASE_SECTION_DEFAULT    \
        CONV2_VALID_MULTI_TEST_CASE_SECTION_STD        \
        CONV2_VALID_MULTI_TEST_CASE_SECTION_FFT        \
        CONV2_VALID_MULTI_TEST_CASE_SECTION_BLAS       \
        CONV2_VALID_MULTI_TEST_CASE_SECTION_CUDNN      \
    }                                                  \
    CONV_TEST_CASE_DEFN

#define CONV2_VALID_MULTI_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {                   \
        CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_STD        \
        CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_FFT        \
        CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_BLAS       \
        CONV2_VALID_MULTI_FLIPPED_TEST_CASE_SECTION_CUDNN      \
    }                                                          \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_MULTI_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {          \
        CONV2_FULL_MULTI_TEST_CASE_SECTION_DEFAULT    \
        CONV2_FULL_MULTI_TEST_CASE_SECTION_STD        \
    }                                                 \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_MULTI_FLIPPED_TEST_CASE(name, description) \
    CONV_TEST_CASE_DECL(name, description) {                  \
        CONV2_FULL_MULTI_FLIPPED_TEST_CASE_SECTION_DEFAULT    \
        CONV2_FULL_MULTI_FLIPPED_TEST_CASE_SECTION_STD        \
    }                                                         \
    CONV_TEST_CASE_DEFN
