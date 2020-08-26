//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_CUDNN_MODE
#define TEST_CUDNN
#endif

#define POOL_2D_FUNCTOR(name, ...)                                                                          \
    struct name {                                                                                           \
        template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename X, typename Y> \
        static void apply(X&& x, Y&& y) {                                                                   \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
    };

#define DYN_POOL_2D_FUNCTOR(name, ...)                                                                      \
    struct name {                                                                                           \
        template <typename X, typename Y>                                                                   \
        static void apply(X&& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) { \
            __VA_ARGS__;                                                                                    \
        }                                                                                                   \
    };

POOL_2D_FUNCTOR(default_mp2_valid, y = etl::max_pool_2d<C1, C2, S1, S2, P1, P2>(x))
POOL_2D_FUNCTOR(std_mp2_valid, y = selected_helper(etl::pool_impl::STD, (etl::max_pool_2d<C1, C2, S1, S2, P1, P2>(x))))

POOL_2D_FUNCTOR(default_avgp2_valid, y = etl::avg_pool_2d<C1, C2, S1, S2, P1, P2>(x))
POOL_2D_FUNCTOR(std_avgp2_valid, y = selected_helper(etl::pool_impl::STD, (etl::avg_pool_2d<C1, C2, S1, S2, P1, P2>(x))))

DYN_POOL_2D_FUNCTOR(default_dyn_mp2_valid, y = etl::max_pool_2d(x, c1, c2, s1, s2, p1, p2))
DYN_POOL_2D_FUNCTOR(std_dyn_mp2_valid, y = selected_helper(etl::pool_impl::STD, (etl::max_pool_2d(x, c1, c2, s1, s2, p1, p2))))

DYN_POOL_2D_FUNCTOR(default_dyn_avgp2_valid, y = etl::avg_pool_2d(x, c1, c2, s1, s2, p1, p2))
DYN_POOL_2D_FUNCTOR(std_dyn_avgp2_valid, y = selected_helper(etl::pool_impl::STD, (etl::avg_pool_2d(x, c1, c2, s1, s2, p1, p2))))

#define MP2_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_mp2_valid)
#define MP2_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_mp2_valid)

#define AVGP2_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_avgp2_valid)
#define AVGP2_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_avgp2_valid)

#define DYN_MP2_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_dyn_mp2_valid)
#define DYN_MP2_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_dyn_mp2_valid)

#define DYN_AVGP2_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_dyn_avgp2_valid)
#define DYN_AVGP2_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_dyn_avgp2_valid)

#ifdef TEST_CUDNN
POOL_2D_FUNCTOR(cudnn_mp2_valid, y = selected_helper(etl::pool_impl::CUDNN, (etl::max_pool_2d<C1, C2, S1, S2, P1, P2>(x))))
POOL_2D_FUNCTOR(cudnn_avgp2_valid, y = selected_helper(etl::pool_impl::CUDNN, (etl::avg_pool_2d<C1, C2, S1, S2, P1, P2>(x))))

DYN_POOL_2D_FUNCTOR(cudnn_dyn_mp2_valid, y = selected_helper(etl::pool_impl::CUDNN, (etl::max_pool_2d(x, c1, c2, s1, s2, p1, p2))))
DYN_POOL_2D_FUNCTOR(cudnn_dyn_avgp2_valid, y = selected_helper(etl::pool_impl::CUDNN, (etl::avg_pool_2d(x, c1, c2, s1, s2, p1, p2))))

#define MP2_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_mp2_valid)
#define AVGP2_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_avgp2_valid)

#define DYN_MP2_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_dyn_mp2_valid)
#define DYN_AVGP2_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_dyn_avgp2_valid)
#else
#define MP2_TEST_CASE_SECTION_CUDNN
#define AVGP2_TEST_CASE_SECTION_CUDNN

#define DYN_MP2_TEST_CASE_SECTION_CUDNN
#define DYN_AVGP2_TEST_CASE_SECTION_CUDNN
#endif

#define POOL_TEST_CASE_DECL(name, description)                                 \
    template <typename T, typename Impl>                                       \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define POOL_TEST_CASE_SECTION(Tn, Impln)                                         \
    ETL_SECTION(#Tn "_" #Impln) {                                                 \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define POOL_TEST_CASE_DEFN              \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define POOL_TEST_CASE_SECTIONS(S1) \
    POOL_TEST_CASE_SECTION(float, S1)   \
    POOL_TEST_CASE_SECTION(double, S1)

#define MP2_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {   \
        MP2_TEST_CASE_SECTION_DEFAULT    \
        MP2_TEST_CASE_SECTION_STD        \
        MP2_TEST_CASE_SECTION_CUDNN      \
    }                                          \
    POOL_TEST_CASE_DEFN

#define DYN_MP2_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {       \
        DYN_MP2_TEST_CASE_SECTION_DEFAULT    \
        DYN_MP2_TEST_CASE_SECTION_STD        \
        DYN_MP2_TEST_CASE_SECTION_CUDNN      \
    }                                              \
    POOL_TEST_CASE_DEFN

#define AVGP2_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {   \
        AVGP2_TEST_CASE_SECTION_DEFAULT    \
        AVGP2_TEST_CASE_SECTION_STD        \
        AVGP2_TEST_CASE_SECTION_CUDNN      \
    }                                          \
    POOL_TEST_CASE_DEFN

#define DYN_AVGP2_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {       \
        DYN_AVGP2_TEST_CASE_SECTION_DEFAULT    \
        DYN_AVGP2_TEST_CASE_SECTION_STD        \
        DYN_AVGP2_TEST_CASE_SECTION_CUDNN      \
    }                                              \
    POOL_TEST_CASE_DEFN
