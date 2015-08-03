//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define MMUL_FUNCTOR( name, ...)                                \
struct name {                                                   \
    template<typename A, typename B, typename C>                \
    static void apply(A&& a, B&& b, C& c){                      \
        __VA_ARGS__;                                            \
    }                                                           \
};

MMUL_FUNCTOR( default_mmul, c = etl::mul(a, b) )
MMUL_FUNCTOR( lazy_mmul, c = etl::lazy_mul(a, b) )
MMUL_FUNCTOR( strassen_mmul, c = etl::strassen_mul(a, b) )
MMUL_FUNCTOR( std_mmul, etl::impl::standard::mm_mul(a, b, c) )
MMUL_FUNCTOR( eblas_mmul_float, etl::impl::eblas::fast_sgemm(a, b, c) )
MMUL_FUNCTOR( eblas_mmul_double, etl::impl::eblas::fast_dgemm(a, b, c) )

#define MMUL_TEST_CASE_SECTION_DEFAULT   MMUL_TEST_CASE_SECTIONS( default_mmul, default_mmul )
#define MMUL_TEST_CASE_SECTION_LAZY      MMUL_TEST_CASE_SECTIONS( lazy_mmul, lazy_mmul )
#define MMUL_TEST_CASE_SECTION_STD       MMUL_TEST_CASE_SECTIONS( std_mmul, std_mmul )
#define MMUL_TEST_CASE_SECTION_STRASSEN  MMUL_TEST_CASE_SECTIONS( strassen_mmul, strassen_mmul )
#define MMUL_TEST_CASE_SECTION_EBLAS     MMUL_TEST_CASE_SECTIONS( eblas_mmul_float, eblas_mmul_double )

#ifdef ETL_BLAS_MODE
MMUL_FUNCTOR( blas_mmul_float, etl::impl::blas::sgemm(a, b, c) )
MMUL_FUNCTOR( blas_mmul_double, etl::impl::blas::dgemm(a, b, c) )
#define MMUL_TEST_CASE_SECTION_BLAS  MMUL_TEST_CASE_SECTIONS( blas_mmul_float, blas_mmul_double )
#else
#define MMUL_TEST_CASE_SECTION_BLAS
#endif

#ifdef ETL_CUBLAS_MODE
MMUL_FUNCTOR( cublas_mmul_float, etl::impl::cublas::sgemm(a, b, c) )
MMUL_FUNCTOR( cublas_mmul_double, etl::impl::cublas::dgemm(a, b, c) )
#define MMUL_TEST_CASE_SECTION_CUBLAS  MMUL_TEST_CASE_SECTIONS( cublas_mmul_float, cublas_mmul_double )
#else
#define MMUL_TEST_CASE_SECTION_CUBLAS
#endif

#define MMUL_TEST_CASE_DECL( name, description ) \
    template<typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )(); \
    TEST_CASE( name, description )

#define MMUL_TEST_CASE_SECTION( Tn, Impln) \
        SECTION( #Tn "_" #Impln ) \
        { \
            INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )<Tn, Impln>(); \
        }

#define MMUL_TEST_CASE_DEFN \
    template<typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME( ____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____ )()

#define MMUL_TEST_CASE_SECTIONS( S1, S2 ) \
    MMUL_TEST_CASE_SECTION( float, S1 ) \
    MMUL_TEST_CASE_SECTION( double, S2 )

#define MMUL_TEST_CASE( name, description ) \
    MMUL_TEST_CASE_DECL( name, description ) \
    { \
        MMUL_TEST_CASE_SECTION_DEFAULT \
        MMUL_TEST_CASE_SECTION_STD \
        MMUL_TEST_CASE_SECTION_LAZY \
        MMUL_TEST_CASE_SECTION_STRASSEN \
        MMUL_TEST_CASE_SECTION_BLAS \
        MMUL_TEST_CASE_SECTION_CUBLAS \
        MMUL_TEST_CASE_SECTION_EBLAS \
    } \
    MMUL_TEST_CASE_DEFN
