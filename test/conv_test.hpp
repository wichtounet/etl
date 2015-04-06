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

//MMUL_FUNCTOR( lazy_mmul, c = etl::lazy_mmul(a, b) )
//MMUL_FUNCTOR( strassen_mmul, c = etl::strassen_mmul(a, b) )
//MMUL_FUNCTOR( eblas_mmul_float, etl::impl::eblas::fast_sgemm(a, b, c) )
//MMUL_FUNCTOR( eblas_mmul_double, etl::impl::eblas::fast_dgemm(a, b, c) )

#define CONV1_FULL_TEST_CASE_SECTION_DEFAULT   CONV_TEST_CASE_SECTIONS( default_conv1_full, default_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_STD       CONV_TEST_CASE_SECTIONS( std_conv1_full, std_conv1_full )
#define CONV1_FULL_TEST_CASE_SECTION_REDUC     CONV_TEST_CASE_SECTIONS( reduc_conv1_full, reduc_conv1_full )

//#define MMUL_TEST_CASE_SECTION_LAZY      MMUL_TEST_CASE_SECTIONS( lazy_mmul, lazy_mmul )
//#define MMUL_TEST_CASE_SECTION_STRASSEN  MMUL_TEST_CASE_SECTIONS( strassen_mmul, strassen_mmul )
//#define MMUL_TEST_CASE_SECTION_EBLAS     MMUL_TEST_CASE_SECTIONS( eblas_mmul_float, eblas_mmul_double )

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
        SECTION( #Tn #Impln ) \
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
