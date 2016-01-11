//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define MUL_FUNCTOR(name, ...)                        \
    /*!                                               \
     * \brief name functor                            \
     */                                               \
    struct name {                                     \
        /*!                                           \
         * \brief Apply the functor                   \
         * \param a a                                 \
         * \param b a                                 \
         * \param c target                            \
         */                                           \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C& c) {       \
            __VA_ARGS__;                              \
        }                                             \
    };

MUL_FUNCTOR(default_gemm, c = etl::mul(a, b))
MUL_FUNCTOR(lazy_gemm, c = etl::lazy_mul(a, b))
MUL_FUNCTOR(strassen_gemm, c = etl::strassen_mul(a, b))
MUL_FUNCTOR(std_gemm, etl::impl::standard::mm_mul(a, b, c))
MUL_FUNCTOR(eblas_gemm, etl::impl::eblas::gemm(a, b, c))

MUL_FUNCTOR(default_gemv, c = etl::mul(a, b))
MUL_FUNCTOR(std_gemv, etl::impl::standard::mv_mul(a, b, c))

MUL_FUNCTOR(default_gevm, c = etl::mul(a, b))
MUL_FUNCTOR(std_gevm, etl::impl::standard::vm_mul(a, b, c))

#define GEMM_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemm, default_gemm)
#define GEMM_TEST_CASE_SECTION_LAZY MUL_TEST_CASE_SECTIONS(lazy_gemm, lazy_gemm)
#define GEMM_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemm, std_gemm)
#define GEMM_TEST_CASE_SECTION_STRASSEN MUL_TEST_CASE_SECTIONS(strassen_gemm, strassen_gemm)
#define GEMM_TEST_CASE_SECTION_EBLAS MUL_TEST_CASE_SECTIONS(eblas_gemm, eblas_gemm)

#define GEMV_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemv, default_gemv)
#define GEMV_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemv, std_gemv)

#define GEVM_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gevm, default_gevm)
#define GEVM_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gevm, std_gevm)

#ifdef ETL_BLAS_MODE
MUL_FUNCTOR(blas_gemm, etl::impl::blas::gemm(a, b, c))
MUL_FUNCTOR(blas_gemv, etl::impl::blas::gemv(a, b, c))
MUL_FUNCTOR(blas_gevm, etl::impl::blas::gevm(a, b, c))
#define GEMM_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemm, blas_gemm)
#define GEMV_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemv, blas_gemv)
#define GEVM_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gevm, blas_gevm)
#else
#define GEMM_TEST_CASE_SECTION_BLAS
#define GEMV_TEST_CASE_SECTION_BLAS
#define GEVM_TEST_CASE_SECTION_BLAS
#endif

#ifdef ETL_CUBLAS_MODE
MUL_FUNCTOR(cublas_gemm, etl::impl::cublas::gemm(a, b, c))
MUL_FUNCTOR(cublas_gemv, etl::impl::cublas::gemv(a, b, c))
MUL_FUNCTOR(cublas_gevm, etl::impl::cublas::gevm(a, b, c))
#define GEMM_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemm, cublas_gemm)
#define GEMV_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemv, cublas_gemv)
#define GEVM_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gevm, cublas_gevm)
#else
#define GEMM_TEST_CASE_SECTION_CUBLAS
#define GEMV_TEST_CASE_SECTION_CUBLAS
#define GEVM_TEST_CASE_SECTION_CUBLAS
#endif

#define MUL_TEST_CASE_DECL(name, description)                                                 \
    template <typename T, typename Impl>                                                      \
    static void INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    TEST_CASE(name, description)

#define MUL_TEST_CASE_SECTION(Tn, Impln)                                                         \
    SECTION(#Tn "_" #Impln) {                                                                    \
        INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define MUL_TEST_CASE_DEFN               \
    template <typename T, typename Impl> \
    static void INTERNAL_CATCH_UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define MUL_TEST_CASE_SECTIONS(S1, S2) \
    MUL_TEST_CASE_SECTION(float, S1)   \
    MUL_TEST_CASE_SECTION(double, S2)

#define GEMM_TEST_CASE(name, description)   \
    MUL_TEST_CASE_DECL(name, description) { \
        GEMM_TEST_CASE_SECTION_DEFAULT      \
        GEMM_TEST_CASE_SECTION_STD          \
        GEMM_TEST_CASE_SECTION_LAZY         \
        GEMM_TEST_CASE_SECTION_STRASSEN     \
        GEMM_TEST_CASE_SECTION_BLAS         \
        GEMM_TEST_CASE_SECTION_CUBLAS       \
        GEMM_TEST_CASE_SECTION_EBLAS        \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEMV_TEST_CASE(name, description)   \
    MUL_TEST_CASE_DECL(name, description) { \
        GEMV_TEST_CASE_SECTION_DEFAULT      \
        GEMV_TEST_CASE_SECTION_STD          \
        GEMV_TEST_CASE_SECTION_BLAS         \
        GEMV_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEVM_TEST_CASE(name, description)   \
    MUL_TEST_CASE_DECL(name, description) { \
        GEVM_TEST_CASE_SECTION_DEFAULT      \
        GEVM_TEST_CASE_SECTION_STD          \
        GEVM_TEST_CASE_SECTION_BLAS         \
        GEVM_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    MUL_TEST_CASE_DEFN
