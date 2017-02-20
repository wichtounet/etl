//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define SCALAR_FUNCTOR(name, ...)                     \
    struct name {                                     \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C&& c) {      \
            __VA_ARGS__;                              \
        }                                             \
    };

SCALAR_FUNCTOR(default_scalar_add, c = a + b)
SCALAR_FUNCTOR(default_scalar_sub, c = a - b)
SCALAR_FUNCTOR(default_scalar_mul, c = a * b)
SCALAR_FUNCTOR(default_scalar_div, c = a / b)

#define SCALAR_ADD_TEST_CASE_SECTION_DEFAULT SCALAR_TEST_CASE_SECTIONS(default_scalar_add)
#define SCALAR_SUB_TEST_CASE_SECTION_DEFAULT SCALAR_TEST_CASE_SECTIONS(default_scalar_sub)
#define SCALAR_MUL_TEST_CASE_SECTION_DEFAULT SCALAR_TEST_CASE_SECTIONS(default_scalar_mul)
#define SCALAR_DIV_TEST_CASE_SECTION_DEFAULT SCALAR_TEST_CASE_SECTIONS(default_scalar_div)

SCALAR_FUNCTOR(std_scalar_add, c = selected_helper(etl::scalar_impl::STD, a + b))
SCALAR_FUNCTOR(std_scalar_sub, c = selected_helper(etl::scalar_impl::STD, a - b))
SCALAR_FUNCTOR(std_scalar_mul, c = selected_helper(etl::scalar_impl::STD, a * b))
SCALAR_FUNCTOR(std_scalar_div, c = selected_helper(etl::scalar_impl::STD, a / b))

#define SCALAR_ADD_TEST_CASE_SECTION_STD SCALAR_TEST_CASE_SECTIONS(std_scalar_add)
#define SCALAR_SUB_TEST_CASE_SECTION_STD SCALAR_TEST_CASE_SECTIONS(std_scalar_sub)
#define SCALAR_MUL_TEST_CASE_SECTION_STD SCALAR_TEST_CASE_SECTIONS(std_scalar_mul)
#define SCALAR_DIV_TEST_CASE_SECTION_STD SCALAR_TEST_CASE_SECTIONS(std_scalar_div)

#ifdef ETL_BLAS_MODE
SCALAR_FUNCTOR(blas_scalar_add, c = selected_helper(etl::scalar_impl::BLAS, a + b))
SCALAR_FUNCTOR(blas_scalar_sub, c = selected_helper(etl::scalar_impl::BLAS, a - b))
SCALAR_FUNCTOR(blas_scalar_mul, c = selected_helper(etl::scalar_impl::BLAS, a * b))
SCALAR_FUNCTOR(blas_scalar_div, c = selected_helper(etl::scalar_impl::BLAS, a / b))

#define SCALAR_ADD_TEST_CASE_SECTION_BLAS SCALAR_TEST_CASE_SECTIONS(blas_scalar_add)
#define SCALAR_SUB_TEST_CASE_SECTION_BLAS SCALAR_TEST_CASE_SECTIONS(blas_scalar_sub)
#define SCALAR_MUL_TEST_CASE_SECTION_BLAS SCALAR_TEST_CASE_SECTIONS(blas_scalar_mul)
#define SCALAR_DIV_TEST_CASE_SECTION_BLAS SCALAR_TEST_CASE_SECTIONS(blas_scalar_div)
#else
#define SCALAR_ADD_TEST_CASE_SECTION_BLAS
#define SCALAR_SUB_TEST_CASE_SECTION_BLAS
#define SCALAR_MUL_TEST_CASE_SECTION_BLAS
#define SCALAR_DIV_TEST_CASE_SECTION_BLAS
#endif

#define SCALAR_TEST_CASE_DECL(name, description)                               \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define SCALAR_TEST_CASE_SECTION(Tn, Impln)                                       \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define SCALAR_TEST_CASE_DEFN            \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define SCALAR_TEST_CASE_SECTIONS(S1)   \
    SCALAR_TEST_CASE_SECTION(float, S1) \
    SCALAR_TEST_CASE_SECTION(double, S1)

#define SCALAR_ADD_TEST_CASE(name, description) \
    SCALAR_TEST_CASE_DECL(name, description) {  \
        SCALAR_ADD_TEST_CASE_SECTION_DEFAULT    \
        SCALAR_ADD_TEST_CASE_SECTION_STD        \
        SCALAR_ADD_TEST_CASE_SECTION_BLAS       \
    }                                           \
    SCALAR_TEST_CASE_DEFN

#define SCALAR_SUB_TEST_CASE(name, description) \
    SCALAR_TEST_CASE_DECL(name, description) {  \
        SCALAR_SUB_TEST_CASE_SECTION_DEFAULT    \
        SCALAR_SUB_TEST_CASE_SECTION_STD        \
        SCALAR_SUB_TEST_CASE_SECTION_BLAS       \
    }                                           \
    SCALAR_TEST_CASE_DEFN

#define SCALAR_MUL_TEST_CASE(name, description) \
    SCALAR_TEST_CASE_DECL(name, description) {  \
        SCALAR_MUL_TEST_CASE_SECTION_DEFAULT    \
        SCALAR_MUL_TEST_CASE_SECTION_STD        \
        SCALAR_MUL_TEST_CASE_SECTION_BLAS       \
    }                                           \
    SCALAR_TEST_CASE_DEFN

#define SCALAR_DIV_TEST_CASE(name, description) \
    SCALAR_TEST_CASE_DECL(name, description) {  \
        SCALAR_DIV_TEST_CASE_SECTION_DEFAULT    \
        SCALAR_DIV_TEST_CASE_SECTION_STD        \
        SCALAR_DIV_TEST_CASE_SECTION_BLAS       \
    }                                           \
    SCALAR_TEST_CASE_DEFN
