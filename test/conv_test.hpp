//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

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

CONV_FUNCTOR( default_conv2_full, c = etl::conv_2d_full(a, b) )
CONV_FUNCTOR( std_conv2_full, etl::impl::standard::conv2_full(a, b, c) )
CONV_FUNCTOR( reduc_conv2_full, etl::impl::reduc::conv2_full(a, b, c) )

#define CONV1_FULL_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv1_full, default_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv1_full, std_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_REDUC     CONV_TEST_CASE_SECTIONS( reduc_conv1_full, reduc_conv1_full )

#define CONV2_FULL_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv2_full, default_conv2_full )
#define CONV2_FULL_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv2_full, std_conv2_full )
#define CONV2_FULL_TEST_CASE_SECTION_REDUC     CONV_TEST_CASE_SECTIONS( reduc_conv2_full, reduc_conv2_full )

//#ifdef ETL_BLAS_MODE
//MMUL_FUNCTOR( blas_mmul_float, etl::impl::blas::sgemm(a, b, c) )
//MMUL_FUNCTOR( blas_mmul_double, etl::impl::blas::dgemm(a, b, c) )
//#define MMUL_TEST_CASE_SECTION_BLAS  MMUL_TEST_CASE_SECTIONS( blas_mmul_float, blas_mmul_double )
//#else
//#define MMUL_TEST_CASE_SECTION_BLAS
//#endif

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
    } \
    CONV_TEST_CASE_DEFN

#define CONV2_FULL_TEST_CASE( name, description ) \
    CONV_TEST_CASE_DECL( name, description ) \
    { \
        CONV2_FULL_TEST_CASE_SECTION_DEFAULT \
        CONV2_FULL_TEST_CASE_SECTION_STD \
        CONV2_FULL_TEST_CASE_SECTION_REDUC \
    } \
    CONV_TEST_CASE_DEFN
