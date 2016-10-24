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

#ifdef TEST_AVX
#define TEST_VEC
#elif defined(TEST_SSE)
#define TEST_VEC
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

#ifdef TEST_SSE
DOT_FUNCTOR(vec_dot, SELECTED_SECTION(etl::dot_impl::VEC) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_VEC DOT_TEST_CASE_SECTIONS(vec_dot)
#else
#define DOT_TEST_CASE_SECTION_VEC
#endif

#ifdef TEST_SSE
DOT_FUNCTOR(sse_dot, SELECTED_SECTION(etl::dot_impl::SSE) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_SSE DOT_TEST_CASE_SECTIONS(sse_dot)
#else
#define DOT_TEST_CASE_SECTION_SSE
#endif

#ifdef TEST_AVX
DOT_FUNCTOR(avx_dot, SELECTED_SECTION(etl::dot_impl::AVX) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_AVX DOT_TEST_CASE_SECTIONS(avx_dot)
#else
#define DOT_TEST_CASE_SECTION_AVX
#endif

#ifdef ETL_BLAS_MODE
DOT_FUNCTOR(blas_dot, SELECTED_SECTION(etl::dot_impl::BLAS) { c = etl::dot(a, b); })

#define DOT_TEST_CASE_SECTION_BLAS DOT_TEST_CASE_SECTIONS(blas_dot)
#else
#define DOT_TEST_CASE_SECTION_BLAS
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
        DOT_TEST_CASE_SECTION_SSE            \
        DOT_TEST_CASE_SECTION_AVX            \
        DOT_TEST_CASE_SECTION_BLAS           \
    }                                        \
    DOT_TEST_CASE_DEFN
