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

#define MUL_FUNCTOR(name, ...)                        \
    struct name {                                     \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C& c) {       \
            __VA_ARGS__;                              \
        }                                             \
    };

// Default

MUL_FUNCTOR(default_gemm, c = a * b)
MUL_FUNCTOR(default_gemm_nt, c = a * transpose(b))
MUL_FUNCTOR(default_gemm_tn, c = transpose(a) * b)
MUL_FUNCTOR(default_gemm_tt, c = transpose(a) * transpose(b))
MUL_FUNCTOR(default_gemv, c = etl::mul(a, b))
MUL_FUNCTOR(default_gemv_t, c = etl::mul(transpose(a), b))
MUL_FUNCTOR(default_gevm, c = etl::mul(a, b))
MUL_FUNCTOR(default_gevm_t, c = a * transpose(b))

#define GEMM_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemm, default_gemm)
#define GEMM_TN_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemm_tn, default_gemm_tn)
#define GEMM_NT_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemm_nt, default_gemm_nt)
#define GEMM_TT_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemm_tt, default_gemm_tt)
#define GEMV_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemv, default_gemv)
#define GEMV_T_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gemv_t, default_gemv_t)
#define GEVM_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gevm, default_gevm)
#define GEVM_T_TEST_CASE_SECTION_DEFAULT MUL_TEST_CASE_SECTIONS(default_gevm_t, default_gevm_t)

// Standard

MUL_FUNCTOR(std_gemm, c = selected_helper(etl::gemm_impl::STD, a * b))
MUL_FUNCTOR(std_gemm_tn, c = selected_helper(etl::gemm_impl::STD, transpose(a) * b))
MUL_FUNCTOR(std_gemm_nt, c = selected_helper(etl::gemm_impl::STD, a * transpose(b)))
MUL_FUNCTOR(std_gemm_tt, c = selected_helper(etl::gemm_impl::STD, transpose(a) * transpose(b)))
MUL_FUNCTOR(std_gemv, c = selected_helper(etl::gemm_impl::STD, a * b))
MUL_FUNCTOR(std_gemv_t, c = selected_helper(etl::gemm_impl::STD, transpose(a) * b))
MUL_FUNCTOR(std_gevm, c = selected_helper(etl::gemm_impl::STD, a * b))
MUL_FUNCTOR(std_gevm_t, c = selected_helper(etl::gemm_impl::STD, a * transpose(b)))

#define GEMM_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemm, std_gemm)
#define GEMM_TN_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemm_tn, std_gemm_tn)
#define GEMM_NT_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemm_nt, std_gemm_nt)
#define GEMM_TT_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemm_tt, std_gemm_tt)
#define GEMV_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemv, std_gemv)
#define GEMV_T_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gemv_t, std_gemv_t)
#define GEVM_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gevm, std_gevm)
#define GEVM_T_TEST_CASE_SECTION_STD MUL_TEST_CASE_SECTIONS(std_gevm_t, std_gevm_t)

// Lazy

MUL_FUNCTOR(lazy_gemm, c = etl::lazy_mul(a, b))
#define GEMM_TEST_CASE_SECTION_LAZY MUL_TEST_CASE_SECTIONS(lazy_gemm, lazy_gemm)

// Strassen

MUL_FUNCTOR(strassen_gemm, c = etl::strassen_mul(a, b))
#define GEMM_TEST_CASE_SECTION_STRASSEN MUL_TEST_CASE_SECTIONS(strassen_gemm, strassen_gemm)

// Vectorized

#ifdef TEST_VEC
MUL_FUNCTOR(vec_gemm, c = selected_helper(etl::gemm_impl::VEC, a * b))
MUL_FUNCTOR(vec_gemm_tn, c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b))
MUL_FUNCTOR(vec_gemm_nt, c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)))
MUL_FUNCTOR(vec_gemm_tt, c = selected_helper(etl::gemm_impl::VEC, transpose(a) * transpose(b)))
MUL_FUNCTOR(vec_gemv, c = selected_helper(etl::gemm_impl::VEC, a * b))
MUL_FUNCTOR(vec_gemv_t, c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b))
MUL_FUNCTOR(vec_gevm, c = selected_helper(etl::gemm_impl::VEC, a * b))
MUL_FUNCTOR(vec_gevm_t, c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)))

#define GEMM_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemm, vec_gemm)
#define GEMM_TN_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemm_tn, vec_gemm_tn)
#define GEMM_NT_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemm_nt, vec_gemm_nt)
#define GEMM_TT_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemm_tt, vec_gemm_tt)
#define GEMV_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemv, vec_gemv)
#define GEMV_T_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gemv_t, vec_gemv_t)
#define GEVM_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gevm, vec_gevm)
#define GEVM_T_TEST_CASE_SECTION_VEC MUL_TEST_CASE_SECTIONS(vec_gevm_t, vec_gevm_t)
#else
#define GEMM_TEST_CASE_SECTION_VEC
#define GEMM_TN_TEST_CASE_SECTION_VEC
#define GEMM_NT_TEST_CASE_SECTION_VEC
#define GEMM_TT_TEST_CASE_SECTION_VEC
#define GEMV_TEST_CASE_SECTION_VEC
#define GEMV_T_TEST_CASE_SECTION_VEC
#define GEVM_TEST_CASE_SECTION_VEC
#define GEVM_T_TEST_CASE_SECTION_VEC
#endif

// BLAS

#ifdef ETL_BLAS_MODE
MUL_FUNCTOR(blas_gemm, c = selected_helper(etl::gemm_impl::BLAS, a * b))
MUL_FUNCTOR(blas_gemm_tn, c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b))
MUL_FUNCTOR(blas_gemm_nt, c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)))
MUL_FUNCTOR(blas_gemm_tt, c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * transpose(b)))
MUL_FUNCTOR(blas_gemv, c = selected_helper(etl::gemm_impl::BLAS, a * b))
MUL_FUNCTOR(blas_gemv_t, c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b))
MUL_FUNCTOR(blas_gevm, c = selected_helper(etl::gemm_impl::BLAS, a * b))
MUL_FUNCTOR(blas_gevm_t, c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)))

#define GEMM_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemm, blas_gemm)
#define GEMM_TN_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemm_tn, blas_gemm_tn)
#define GEMM_NT_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemm_nt, blas_gemm_nt)
#define GEMM_TT_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemm_tt, blas_gemm_tt)
#define GEMV_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemv, blas_gemv)
#define GEMV_T_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gemv_t, blas_gemv_t)
#define GEVM_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gevm, blas_gevm)
#define GEVM_T_TEST_CASE_SECTION_BLAS MUL_TEST_CASE_SECTIONS(blas_gevm_t, blas_gevm_t)
#else
#define GEMM_TEST_CASE_SECTION_BLAS
#define GEMM_TN_TEST_CASE_SECTION_BLAS
#define GEMM_NT_TEST_CASE_SECTION_BLAS
#define GEMM_TT_TEST_CASE_SECTION_BLAS
#define GEMV_TEST_CASE_SECTION_BLAS
#define GEMV_T_TEST_CASE_SECTION_BLAS
#define GEVM_TEST_CASE_SECTION_BLAS
#define GEVM_T_TEST_CASE_SECTION_BLAS
#endif

// CUBLAS

#ifdef ETL_CUBLAS_MODE
MUL_FUNCTOR(cublas_gemm, c = selected_helper(etl::gemm_impl::CUBLAS, a * b))
MUL_FUNCTOR(cublas_gemm_tn, c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b))
MUL_FUNCTOR(cublas_gemm_nt, c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)))
MUL_FUNCTOR(cublas_gemm_tt, c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * transpose(b)))
MUL_FUNCTOR(cublas_gemv, c = selected_helper(etl::gemm_impl::CUBLAS, a * b))
MUL_FUNCTOR(cublas_gemv_t, c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b))
MUL_FUNCTOR(cublas_gevm, c = selected_helper(etl::gemm_impl::CUBLAS, a * b))
MUL_FUNCTOR(cublas_gevm_t, c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)))

#define GEMM_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemm, cublas_gemm)
#define GEMM_TN_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemm_tn, cublas_gemm_tn)
#define GEMM_NT_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemm_nt, cublas_gemm_nt)
#define GEMM_TT_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemm_tt, cublas_gemm_tt)
#define GEMV_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemv, cublas_gemv)
#define GEMV_T_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gemv_t, cublas_gemv_t)
#define GEVM_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gevm, cublas_gevm)
#define GEVM_T_TEST_CASE_SECTION_CUBLAS MUL_TEST_CASE_SECTIONS(cublas_gevm_t, cublas_gevm_t)
#else
#define GEMM_TEST_CASE_SECTION_CUBLAS
#define GEMM_TN_TEST_CASE_SECTION_CUBLAS
#define GEMM_NT_TEST_CASE_SECTION_CUBLAS
#define GEMM_TT_TEST_CASE_SECTION_CUBLAS
#define GEMV_TEST_CASE_SECTION_CUBLAS
#define GEMV_T_TEST_CASE_SECTION_CUBLAS
#define GEVM_TEST_CASE_SECTION_CUBLAS
#define GEVM_T_TEST_CASE_SECTION_CUBLAS
#endif

#define MUL_TEST_CASE_DECL(name, description)                                  \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define MUL_TEST_CASE_SECTION(Tn, Impln)                                          \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define MUL_TEST_CASE_DEFN               \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

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
        GEMM_TEST_CASE_SECTION_VEC          \
        GEMM_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEMM_TEST_CASE_FAST(name, description) \
    MUL_TEST_CASE_DECL(name, description) {    \
        GEMM_TEST_CASE_SECTION_DEFAULT         \
        GEMM_TEST_CASE_SECTION_STD             \
        GEMM_TEST_CASE_SECTION_LAZY            \
        GEMM_TEST_CASE_SECTION_BLAS            \
        GEMM_TEST_CASE_SECTION_VEC             \
        GEMM_TEST_CASE_SECTION_CUBLAS          \
    }                                          \
    MUL_TEST_CASE_DEFN

#define GEMM_NT_TEST_CASE(name, description) \
    MUL_TEST_CASE_DECL(name, description) {  \
        GEMM_NT_TEST_CASE_SECTION_DEFAULT    \
        GEMM_NT_TEST_CASE_SECTION_STD        \
        GEMM_NT_TEST_CASE_SECTION_BLAS       \
        GEMM_NT_TEST_CASE_SECTION_VEC        \
        GEMM_NT_TEST_CASE_SECTION_CUBLAS     \
    }                                        \
    MUL_TEST_CASE_DEFN

#define GEMM_TT_TEST_CASE(name, description) \
    MUL_TEST_CASE_DECL(name, description) {  \
        GEMM_TT_TEST_CASE_SECTION_DEFAULT    \
        GEMM_TT_TEST_CASE_SECTION_STD        \
        GEMM_TT_TEST_CASE_SECTION_BLAS       \
        GEMM_TT_TEST_CASE_SECTION_VEC        \
        GEMM_TT_TEST_CASE_SECTION_CUBLAS     \
    }                                        \
    MUL_TEST_CASE_DEFN

#define GEMM_TN_TEST_CASE(name, description) \
    MUL_TEST_CASE_DECL(name, description) {  \
        GEMM_TN_TEST_CASE_SECTION_DEFAULT    \
        GEMM_TN_TEST_CASE_SECTION_STD        \
        GEMM_TN_TEST_CASE_SECTION_BLAS       \
        GEMM_TN_TEST_CASE_SECTION_VEC        \
        GEMM_TN_TEST_CASE_SECTION_CUBLAS     \
    }                                        \
    MUL_TEST_CASE_DEFN

#define GEMV_TEST_CASE(name, description)   \
    MUL_TEST_CASE_DECL(name, description) { \
        GEMV_TEST_CASE_SECTION_DEFAULT      \
        GEMV_TEST_CASE_SECTION_STD          \
        GEMV_TEST_CASE_SECTION_VEC          \
        GEMV_TEST_CASE_SECTION_BLAS         \
        GEMV_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEMV_T_TEST_CASE(name, description) \
    MUL_TEST_CASE_DECL(name, description) { \
        GEMV_T_TEST_CASE_SECTION_DEFAULT    \
        GEMV_T_TEST_CASE_SECTION_STD        \
        GEMV_T_TEST_CASE_SECTION_VEC        \
        GEMV_T_TEST_CASE_SECTION_BLAS       \
        GEMV_T_TEST_CASE_SECTION_CUBLAS     \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEVM_TEST_CASE(name, description)   \
    MUL_TEST_CASE_DECL(name, description) { \
        GEVM_TEST_CASE_SECTION_DEFAULT      \
        GEVM_TEST_CASE_SECTION_STD          \
        GEVM_TEST_CASE_SECTION_VEC          \
        GEVM_TEST_CASE_SECTION_BLAS         \
        GEVM_TEST_CASE_SECTION_CUBLAS       \
    }                                       \
    MUL_TEST_CASE_DEFN

#define GEVM_T_TEST_CASE(name, description) \
    MUL_TEST_CASE_DECL(name, description) { \
        GEVM_T_TEST_CASE_SECTION_DEFAULT    \
        GEVM_T_TEST_CASE_SECTION_STD        \
        GEVM_T_TEST_CASE_SECTION_VEC        \
        GEVM_T_TEST_CASE_SECTION_BLAS       \
        GEVM_T_TEST_CASE_SECTION_CUBLAS     \
    }                                       \
    MUL_TEST_CASE_DEFN
