//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define TRANSPOSE_FUNCTOR(name, ...)      \
    struct name {                         \
        template <typename A, typename C> \
        static void apply(A&& a, C& c) {  \
            __VA_ARGS__;                  \
        }                                 \
    };

TRANSPOSE_FUNCTOR(default_trans, c = transpose(a))
TRANSPOSE_FUNCTOR(std_trans, c = selected_helper(etl::transpose_impl::STD, transpose(a)))

#define TRANSPOSE_TEST_CASE_SECTION_DEFAULT TRANSPOSE_TEST_CASE_SECTIONS(default_trans, default_trans)
#define TRANSPOSE_TEST_CASE_SECTION_STD TRANSPOSE_TEST_CASE_SECTIONS(std_trans, std_trans)

#ifdef ETL_MKL_MODE
TRANSPOSE_FUNCTOR(blas_transpose, c = selected_helper(etl::transpose_impl::MKL, transpose(a)))
#define TRANSPOSE_TEST_CASE_SECTION_BLAS TRANSPOSE_TEST_CASE_SECTIONS(blas_transpose, blas_transpose)
#else
#define TRANSPOSE_TEST_CASE_SECTION_BLAS
#endif

#ifdef ETL_CUBLAS_MODE
TRANSPOSE_FUNCTOR(cublas_transpose, c = selected_helper(etl::transpose_impl::CUBLAS, transpose(a)))
#define TRANSPOSE_TEST_CASE_SECTION_CUBLAS TRANSPOSE_TEST_CASE_SECTIONS(cublas_transpose, cublas_transpose)
#else
#define TRANSPOSE_TEST_CASE_SECTION_CUBLAS
#endif

#define TRANSPOSE_TEST_CASE_DECL(name, description)                                  \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define TRANSPOSE_TEST_CASE_SECTION(Tn, Impln)                                          \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define MUL_TEST_CASE_DEFN               \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define TRANSPOSE_TEST_CASE_SECTIONS(S1, S2) \
    TRANSPOSE_TEST_CASE_SECTION(float, S1)   \
    TRANSPOSE_TEST_CASE_SECTION(double, S2)

#define TRANSPOSE_TEST_CASE(name, description) \
    TRANSPOSE_TEST_CASE_DECL(name, description) {    \
        TRANSPOSE_TEST_CASE_SECTION_DEFAULT    \
        TRANSPOSE_TEST_CASE_SECTION_STD        \
        TRANSPOSE_TEST_CASE_SECTION_BLAS       \
        TRANSPOSE_TEST_CASE_SECTION_CUBLAS     \
    }                                          \
    MUL_TEST_CASE_DEFN
