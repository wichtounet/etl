//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_VECTORIZE
#ifdef __SSE3__
#define TEST_SSE
#endif
#ifdef __AVX__
#define TEST_AVX
#endif
#endif

#define CONV_FUNCTOR( name, ...)                                \
struct name {                                                   \
    template<typename A, typename B, typename C>                \
    static void apply(A&& a, B&& b, C& c){                      \
        __VA_ARGS__;                                            \
    }                                                           \
};

CONV_FUNCTOR( default_conv1_full, c = etl::conv_1d_full(a, b) )
CONV_FUNCTOR( std_conv1_full, etl::impl::standard::conv1_full(a, b, c) )
CONV_FUNCTOR( reduc_conv1_full, etl::impl::reduc::conv1_full(a, b, c) )

CONV_FUNCTOR( default_conv1_same, c = etl::conv_1d_same(a, b) )
CONV_FUNCTOR( std_conv1_same, etl::impl::standard::conv1_same(a, b, c) )

CONV_FUNCTOR( default_conv1_valid, c = etl::conv_1d_valid(a, b) )
CONV_FUNCTOR( std_conv1_valid, etl::impl::standard::conv1_valid(a, b, c) )

CONV_FUNCTOR( default_conv2_full, c = etl::conv_2d_full(a, b) )
CONV_FUNCTOR( std_conv2_full, etl::impl::standard::conv2_full(a, b, c) )
CONV_FUNCTOR( reduc_conv2_full, etl::impl::reduc::conv2_full(a, b, c) )

CONV_FUNCTOR( default_conv2_same, c = etl::conv_2d_same(a, b) )
CONV_FUNCTOR( std_conv2_same, etl::impl::standard::conv2_same(a, b, c) )

CONV_FUNCTOR( default_conv2_valid, c = etl::conv_2d_valid(a, b) )
CONV_FUNCTOR( std_conv2_valid, etl::impl::standard::conv2_valid(a, b, c) )

#define CONV1_FULL_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv1_full, default_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv1_full, std_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_REDUC     CONV_TEST_CASE_SECTIONS( reduc_conv1_full, reduc_conv1_full )

#define CONV1_SAME_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv1_same, default_conv1_same )
#define CONV1_SAME_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv1_same, std_conv1_same )

#define CONV1_VALID_TEST_CASE_SECTION_DEFAULT  CONV_TEST_CASE_SECTIONS( default_conv1_valid, default_conv1_valid )
#define CONV1_VALID_TEST_CASE_SECTION_STD      CONV_TEST_CASE_SECTIONS( std_conv1_valid, std_conv1_valid )

#define CONV2_FULL_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv2_full, default_conv2_full )
#define CONV2_FULL_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv2_full, std_conv2_full )
#define CONV2_FULL_TEST_CASE_SECTION_REDUC     CONV_TEST_CASE_SECTIONS( reduc_conv2_full, reduc_conv2_full )

#define CONV2_SAME_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv2_same, default_conv2_same )
#define CONV2_SAME_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv2_same, std_conv2_same )

#define CONV2_VALID_TEST_CASE_SECTION_DEFAULT  CONV_TEST_CASE_SECTIONS( default_conv2_valid, default_conv2_valid )
#define CONV2_VALID_TEST_CASE_SECTION_STD      CONV_TEST_CASE_SECTIONS( std_conv2_valid, std_conv2_valid )

#ifdef ETL_MKL_MODE
CONV_FUNCTOR( fft_conv1_full, c = etl::fft_conv_1d_full(a, b) )
CONV_FUNCTOR( fft_conv2_full, c = etl::fft_conv_2d_full(a, b) )
#define CONV1_FULL_TEST_CASE_SECTION_FFT       CONV_TEST_CASE_SECTIONS( fft_conv1_full, fft_conv1_full )
#define CONV2_FULL_TEST_CASE_SECTION_FFT       CONV_TEST_CASE_SECTIONS( fft_conv2_full, fft_conv2_full )
#else
#define CONV1_FULL_TEST_CASE_SECTION_FFT
#define CONV2_FULL_TEST_CASE_SECTION_FFT
#endif

#ifdef TEST_SSE
CONV_FUNCTOR( sse_conv1_full_float, etl::impl::sse::sconv1_full(a, b, c) )
CONV_FUNCTOR( sse_conv1_full_double, etl::impl::sse::dconv1_full(a, b, c) )

CONV_FUNCTOR( sse_conv1_same_float, etl::impl::sse::sconv1_same(a, b, c) )
CONV_FUNCTOR( sse_conv1_same_double, etl::impl::sse::dconv1_same(a, b, c) )

CONV_FUNCTOR( sse_conv1_valid_float, etl::impl::sse::sconv1_valid(a, b, c) )
CONV_FUNCTOR( sse_conv1_valid_double, etl::impl::sse::dconv1_valid(a, b, c) )

CONV_FUNCTOR( sse_conv2_full_float, etl::impl::sse::sconv2_full(a, b, c) )
CONV_FUNCTOR( sse_conv2_full_double, etl::impl::sse::dconv2_full(a, b, c) )

CONV_FUNCTOR( sse_conv2_same_float, etl::impl::sse::sconv2_same(a, b, c) )
CONV_FUNCTOR( sse_conv2_same_double, etl::impl::sse::dconv2_same(a, b, c) )

CONV_FUNCTOR( sse_conv2_valid_float, etl::impl::sse::sconv2_valid(a, b, c) )
CONV_FUNCTOR( sse_conv2_valid_double, etl::impl::sse::dconv2_valid(a, b, c) )

#define CONV1_FULL_TEST_CASE_SECTION_SSE   CONV_TEST_CASE_SECTIONS( sse_conv1_full_float, sse_conv1_full_double )
#define CONV1_SAME_TEST_CASE_SECTION_SSE   CONV_TEST_CASE_SECTIONS( sse_conv1_same_float, sse_conv1_same_double )
#define CONV1_VALID_TEST_CASE_SECTION_SSE  CONV_TEST_CASE_SECTIONS( sse_conv1_valid_float, sse_conv1_valid_double )
#define CONV2_FULL_TEST_CASE_SECTION_SSE   CONV_TEST_CASE_SECTIONS( sse_conv2_full_float, sse_conv2_full_double )
#define CONV2_SAME_TEST_CASE_SECTION_SSE   CONV_TEST_CASE_SECTIONS( sse_conv2_same_float, sse_conv2_same_double )
#define CONV2_VALID_TEST_CASE_SECTION_SSE  CONV_TEST_CASE_SECTIONS( sse_conv2_valid_float, sse_conv2_valid_double )
#else
#define CONV1_FULL_TEST_CASE_SECTION_SSE
#define CONV1_SAME_TEST_CASE_SECTION_SSE
#define CONV1_VALID_TEST_CASE_SECTION_SSE
#define CONV2_FULL_TEST_CASE_SECTION_SSE
#define CONV2_SAME_TEST_CASE_SECTION_SSE
#define CONV2_VALID_TEST_CASE_SECTION_SSE
#endif

#ifdef TEST_AVX
CONV_FUNCTOR( avx_conv1_full_float, etl::impl::avx::sconv1_full(a, b, c) )
CONV_FUNCTOR( avx_conv1_full_double, etl::impl::avx::dconv1_full(a, b, c) )

CONV_FUNCTOR( avx_conv1_same_float, etl::impl::avx::sconv1_same(a, b, c) )
CONV_FUNCTOR( avx_conv1_same_double, etl::impl::avx::dconv1_same(a, b, c) )

CONV_FUNCTOR( avx_conv1_valid_float, etl::impl::avx::sconv1_valid(a, b, c) )
CONV_FUNCTOR( avx_conv1_valid_double, etl::impl::avx::dconv1_valid(a, b, c) )

CONV_FUNCTOR( avx_conv2_full_float, etl::impl::avx::sconv2_full(a, b, c) )
CONV_FUNCTOR( avx_conv2_full_double, etl::impl::avx::dconv2_full(a, b, c) )
CONV_FUNCTOR( avx_conv2_same_float, etl::impl::avx::sconv2_same(a, b, c) )
CONV_FUNCTOR( avx_conv2_same_double, etl::impl::avx::dconv2_same(a, b, c) )

CONV_FUNCTOR( avx_conv2_valid_float, etl::impl::avx::sconv2_valid(a, b, c) )
CONV_FUNCTOR( avx_conv2_valid_double, etl::impl::avx::dconv2_valid(a, b, c) )

#define CONV1_FULL_TEST_CASE_SECTION_AVX   CONV_TEST_CASE_SECTIONS( avx_conv1_full_float, avx_conv1_full_double )
#define CONV1_SAME_TEST_CASE_SECTION_AVX   CONV_TEST_CASE_SECTIONS( avx_conv1_same_float, avx_conv1_same_double )
#define CONV1_VALID_TEST_CASE_SECTION_AVX  CONV_TEST_CASE_SECTIONS( avx_conv1_valid_float, avx_conv1_valid_double )
#define CONV2_FULL_TEST_CASE_SECTION_AVX   CONV_TEST_CASE_SECTIONS( avx_conv2_full_float, avx_conv2_full_double )
#define CONV2_SAME_TEST_CASE_SECTION_AVX   CONV_TEST_CASE_SECTIONS( avx_conv2_same_float, avx_conv2_same_double )
#define CONV2_VALID_TEST_CASE_SECTION_AVX  CONV_TEST_CASE_SECTIONS( avx_conv2_valid_float, avx_conv2_valid_double )
#else
#define CONV1_FULL_TEST_CASE_SECTION_AVX
#define CONV1_SAME_TEST_CASE_SECTION_AVX
#define CONV1_VALID_TEST_CASE_SECTION_AVX
#define CONV2_FULL_TEST_CASE_SECTION_AVX
#define CONV2_SAME_TEST_CASE_SECTION_AVX
#define CONV2_VALID_TEST_CASE_SECTION_AVX
#endif

#define CONV_TEST_CASE_DECL( name, description ) \
    template<typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )(); \
    TEST_CASE( name, description )

#define CONV_TEST_CASE_SECTION( Tn, Impln) \
        SECTION( #Tn "_" #Impln ) \
        { \
            INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )<Tn, Impln>(); \
        }

#define CONV_TEST_CASE_DEFN \
    template<typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )()

#define CONV_TEST_CASE_SECTIONS( S1, S2 ) \
    CONV_TEST_CASE_SECTION( float, S1 ) \
    CONV_TEST_CASE_SECTION( double, S2 )

#define CONV1_FULL_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV1_FULL_TEST_CASE_SECTION_DEFAULT \
        CONV1_FULL_TEST_CASE_SECTION_STD \
        CONV1_FULL_TEST_CASE_SECTION_REDUC \
        CONV1_FULL_TEST_CASE_SECTION_FFT \
        CONV1_FULL_TEST_CASE_SECTION_SSE \
        CONV1_FULL_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN

#define CONV1_SAME_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV1_SAME_TEST_CASE_SECTION_DEFAULT \
        CONV1_SAME_TEST_CASE_SECTION_STD \
        CONV1_SAME_TEST_CASE_SECTION_SSE \
        CONV1_SAME_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN

#define CONV1_VALID_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV1_VALID_TEST_CASE_SECTION_DEFAULT \
        CONV1_VALID_TEST_CASE_SECTION_STD \
        CONV1_VALID_TEST_CASE_SECTION_SSE \
        CONV1_VALID_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV2_FULL_TEST_CASE_SECTION_DEFAULT \
        CONV2_FULL_TEST_CASE_SECTION_STD \
        CONV2_FULL_TEST_CASE_SECTION_REDUC \
        CONV2_FULL_TEST_CASE_SECTION_FFT \
        CONV2_FULL_TEST_CASE_SECTION_SSE \
        CONV2_FULL_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN

#define CONV2_SAME_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV2_SAME_TEST_CASE_SECTION_DEFAULT \
        CONV2_SAME_TEST_CASE_SECTION_STD \
        CONV2_SAME_TEST_CASE_SECTION_SSE \
        CONV2_SAME_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN

#define CONV2_VALID_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV2_VALID_TEST_CASE_SECTION_DEFAULT \
        CONV2_VALID_TEST_CASE_SECTION_STD \
        CONV2_VALID_TEST_CASE_SECTION_SSE \
        CONV2_VALID_TEST_CASE_SECTION_AVX \
    } \
    CONV_TEST_CASE_DEFN
