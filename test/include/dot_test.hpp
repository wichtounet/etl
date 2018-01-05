//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_VECTORIZE_IMPL
#ifdef __AVX__
#define TEST_VEC
#elif defined(__SSE3__)
#define TEST_VEC
#endif
#endif

#define DOT_FUNCTOR(name, ...)                        \
    struct name {                                     \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C&& c) {      \
            __VA_ARGS__;                              \
        }                                             \
    };

DOT_FUNCTOR(default_dot, c = etl::dot(a, b))
DOT_FUNCTOR(std_dot, SELECTED_SECTION(etl::dot_impl::STD) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_DEFAULT DOT_TEST_CASE_SECTIONS(default_dot)
#define DOT_TEST_CASE_SECTION_STD DOT_TEST_CASE_SECTIONS(std_dot)

#ifdef TEST_VEC
DOT_FUNCTOR(vec_dot, SELECTED_SECTION(etl::dot_impl::VEC) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_VEC DOT_TEST_CASE_SECTIONS(vec_dot)
#else
#define DOT_TEST_CASE_SECTION_VEC
#endif

#ifdef ETL_BLAS_MODE
DOT_FUNCTOR(blas_dot, SELECTED_SECTION(etl::dot_impl::BLAS) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_BLAS DOT_TEST_CASE_SECTIONS(blas_dot)
#else
#define DOT_TEST_CASE_SECTION_BLAS
#endif

#ifdef ETL_CUBLAS_MODE
DOT_FUNCTOR(cublas_dot, SELECTED_SECTION(etl::dot_impl::CUBLAS) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_CUBLAS DOT_TEST_CASE_SECTIONS(cublas_dot)
#else
#define DOT_TEST_CASE_SECTION_CUBLAS
#endif

#define DOT_TEST_CASE_DECL(name, description)                                 \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define DOT_TEST_CASE_SECTION(Tn, Impln)                                         \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define DOT_TEST_CASE_DEFN              \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define DOT_TEST_CASE_SECTIONS(S1) \
    DOT_TEST_CASE_SECTION(float, S1)   \
    DOT_TEST_CASE_SECTION(double, S1)

#define DOT_TEST_CASE(name, description)     \
    DOT_TEST_CASE_DECL(name, description) { \
        DOT_TEST_CASE_SECTION_DEFAULT        \
        DOT_TEST_CASE_SECTION_STD            \
        DOT_TEST_CASE_SECTION_VEC            \
        DOT_TEST_CASE_SECTION_BLAS           \
        DOT_TEST_CASE_SECTION_CUBLAS           \
    }                                        \
    DOT_TEST_CASE_DEFN
