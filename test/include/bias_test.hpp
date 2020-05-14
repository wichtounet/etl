//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#define BIAS_FUNCTOR(name, ...)                       \
    struct name {                                     \
        template <typename A, typename B, typename C> \
        static void apply(A&& a, B&& b, C&& c) {      \
            __VA_ARGS__;                              \
        }                                             \
    };

BIAS_FUNCTOR(default_bias_add_2d, c = etl::bias_add_2d(a, b))
BIAS_FUNCTOR(default_bias_add_4d, c = etl::bias_add_4d(a, b))

#define BIAS_ADD_2D_TEST_CASE_SECTION_DEFAULT BIAS_ADD_TEST_CASE_SECTIONS(default_bias_add_2d)
#define BIAS_ADD_4D_TEST_CASE_SECTION_DEFAULT BIAS_ADD_TEST_CASE_SECTIONS(default_bias_add_4d)

BIAS_FUNCTOR(std_bias_add_2d, SELECTED_SECTION(etl::bias_add_impl::STD) { c = etl::bias_add_2d(a, b); })
BIAS_FUNCTOR(std_bias_add_4d, SELECTED_SECTION(etl::bias_add_impl::STD) { c = etl::bias_add_4d(a, b); })

#define BIAS_ADD_2D_TEST_CASE_SECTION_STD BIAS_ADD_TEST_CASE_SECTIONS(std_bias_add_2d)
#define BIAS_ADD_4D_TEST_CASE_SECTION_STD BIAS_ADD_TEST_CASE_SECTIONS(std_bias_add_4d)

#ifdef TEST_VEC
BIAS_FUNCTOR(vec_bias_add_2d, SELECTED_SECTION(etl::bias_add_impl::VEC) { c = etl::bias_add_2d(a, b); })
BIAS_FUNCTOR(vec_bias_add_4d, SELECTED_SECTION(etl::bias_add_impl::VEC) { c = etl::bias_add_4d(a, b); })

#define BIAS_ADD_2D_TEST_CASE_SECTION_VEC BIAS_ADD_TEST_CASE_SECTIONS(vec_bias_add_2d)
#define BIAS_ADD_4D_TEST_CASE_SECTION_VEC BIAS_ADD_TEST_CASE_SECTIONS(vec_bias_add_4d)
#else
#define BIAS_ADD_2D_TEST_CASE_SECTION_VEC
#define BIAS_ADD_4D_TEST_CASE_SECTION_VEC
#endif

#ifdef ETL_CUDNN_MODE
BIAS_FUNCTOR(cudnn_bias_add_2d, SELECTED_SECTION(etl::bias_add_impl::CUDNN) { c = etl::bias_add_2d(a, b); })
BIAS_FUNCTOR(cudnn_bias_add_4d, SELECTED_SECTION(etl::bias_add_impl::CUDNN) { c = etl::bias_add_4d(a, b); })

#define BIAS_ADD_2D_TEST_CASE_SECTION_CUDNN BIAS_ADD_TEST_CASE_SECTIONS(cudnn_bias_add_2d)
#define BIAS_ADD_4D_TEST_CASE_SECTION_CUDNN BIAS_ADD_TEST_CASE_SECTIONS(cudnn_bias_add_4d)
#else
#define BIAS_ADD_2D_TEST_CASE_SECTION_CUDNN
#define BIAS_ADD_4D_TEST_CASE_SECTION_CUDNN
#endif

#define BIAS_ADD_TEST_CASE_DECL(name, description)                             \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define BIAS_ADD_TEST_CASE_SECTION(Tn, Impln)                                     \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define BIAS_ADD_TEST_CASE_DEFN          \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define BIAS_ADD_TEST_CASE_SECTIONS(S1)   \
    BIAS_ADD_TEST_CASE_SECTION(float, S1) \
    BIAS_ADD_TEST_CASE_SECTION(double, S1)

#define BIAS_ADD_2D_TEST_CASE(name, description) \
    BIAS_ADD_TEST_CASE_DECL(name, description) { \
        BIAS_ADD_2D_TEST_CASE_SECTION_DEFAULT    \
        BIAS_ADD_2D_TEST_CASE_SECTION_STD        \
        BIAS_ADD_2D_TEST_CASE_SECTION_VEC        \
        BIAS_ADD_2D_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    BIAS_ADD_TEST_CASE_DEFN

#define BIAS_ADD_4D_TEST_CASE(name, description) \
    BIAS_ADD_TEST_CASE_DECL(name, description) { \
        BIAS_ADD_4D_TEST_CASE_SECTION_DEFAULT    \
        BIAS_ADD_4D_TEST_CASE_SECTION_STD        \
        BIAS_ADD_4D_TEST_CASE_SECTION_VEC        \
        BIAS_ADD_4D_TEST_CASE_SECTION_CUDNN      \
    }                                            \
    BIAS_ADD_TEST_CASE_DEFN
