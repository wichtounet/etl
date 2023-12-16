//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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

#define SUM_FUNCTOR(name, ...)                        \
    struct name {                                     \
        template <typename A, typename C> \
        static void apply(A&& a, C&& c) {      \
            __VA_ARGS__;                              \
        }                                             \
    };

SUM_FUNCTOR(default_sum, c = etl::sum(a))
SUM_FUNCTOR(std_sum, SELECTED_SECTION(etl::sum_impl::STD) { c = etl::sum(a); })

SUM_FUNCTOR(default_asum, c = etl::asum(a))
SUM_FUNCTOR(std_asum, SELECTED_SECTION(etl::sum_impl::STD) { c = etl::asum(a); })

#define SUM_TEST_CASE_SECTION_DEFAULT SUM_TEST_CASE_SECTIONS(default_sum)
#define SUM_TEST_CASE_SECTION_STD SUM_TEST_CASE_SECTIONS(std_sum)

#define ASUM_TEST_CASE_SECTION_DEFAULT SUM_TEST_CASE_SECTIONS(default_asum)
#define ASUM_TEST_CASE_SECTION_STD SUM_TEST_CASE_SECTIONS(std_asum)

#ifdef TEST_VEC
SUM_FUNCTOR(vec_sum, SELECTED_SECTION(etl::sum_impl::VEC) { c = etl::sum(a); })
SUM_FUNCTOR(vec_asum, SELECTED_SECTION(etl::sum_impl::VEC) { c = etl::asum(a); })

#define SUM_TEST_CASE_SECTION_VEC SUM_TEST_CASE_SECTIONS(vec_sum)
#define ASUM_TEST_CASE_SECTION_VEC SUM_TEST_CASE_SECTIONS(vec_asum)
#else
#define SUM_TEST_CASE_SECTION_VEC
#define ASUM_TEST_CASE_SECTION_VEC
#endif

#ifdef ETL_BLAS_MODE
SUM_FUNCTOR(blas_sum, SELECTED_SECTION(etl::sum_impl::BLAS) { c = etl::sum(a); })
SUM_FUNCTOR(blas_asum, SELECTED_SECTION(etl::sum_impl::BLAS) { c = etl::asum(a); })

#define SUM_TEST_CASE_SECTION_BLAS SUM_TEST_CASE_SECTIONS(blas_sum)
#define ASUM_TEST_CASE_SECTION_BLAS SUM_TEST_CASE_SECTIONS(blas_asum)
#else
#define SUM_TEST_CASE_SECTION_BLAS
#define ASUM_TEST_CASE_SECTION_BLAS
#endif

#ifdef ETL_CUBLAS_MODE
SUM_FUNCTOR(cublas_sum, SELECTED_SECTION(etl::sum_impl::CUBLAS) { c = etl::sum(a); })
SUM_FUNCTOR(cublas_asum, SELECTED_SECTION(etl::sum_impl::CUBLAS) { c = etl::asum(a); })

#define SUM_TEST_CASE_SECTION_CUBLAS SUM_TEST_CASE_SECTIONS(cublas_sum)
#define ASUM_TEST_CASE_SECTION_CUBLAS SUM_TEST_CASE_SECTIONS(cublas_asum)
#else
#define SUM_TEST_CASE_SECTION_CUBLAS
#define ASUM_TEST_CASE_SECTION_CUBLAS
#endif

#define SUM_TEST_CASE_DECL(name, description)                                  \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define SUM_TEST_CASE_SECTION(Tn, Impln)                                          \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define SUM_TEST_CASE_DEFN               \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define SUM_TEST_CASE_SECTIONS(S1) \
    SUM_TEST_CASE_SECTION(float, S1)   \
    SUM_TEST_CASE_SECTION(double, S1)

#define SUM_TEST_CASE(name, description)    \
    SUM_TEST_CASE_DECL(name, description) { \
        SUM_TEST_CASE_SECTION_DEFAULT       \
        SUM_TEST_CASE_SECTION_STD           \
        SUM_TEST_CASE_SECTION_VEC           \
        SUM_TEST_CASE_SECTION_BLAS          \
        SUM_TEST_CASE_SECTION_CUBLAS        \
    }                                       \
    SUM_TEST_CASE_DEFN

#define ASUM_TEST_CASE(name, description)   \
    SUM_TEST_CASE_DECL(name, description) { \
        ASUM_TEST_CASE_SECTION_DEFAULT      \
        ASUM_TEST_CASE_SECTION_STD          \
        ASUM_TEST_CASE_SECTION_VEC          \
        ASUM_TEST_CASE_SECTION_BLAS         \
        ASUM_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    SUM_TEST_CASE_DEFN
