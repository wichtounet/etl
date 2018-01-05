//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifdef ETL_CUDNN_MODE
#define TEST_CUDNN
#endif

#define POOL_3D_FUNCTOR(name, ...)                                                                                                           \
    struct name {                                                                                                                            \
        template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename X, typename Y> \
        static void apply(X&& x, Y&& y) {                                                                                                    \
            __VA_ARGS__;                                                                                                                     \
        }                                                                                                                                    \
    };

#define DYN_POOL_3D_FUNCTOR(name, ...)                                                                                                       \
    struct name {                                                                                                                            \
        template <typename X, typename Y>                                                                                                    \
        static void apply(X&& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) { \
            __VA_ARGS__;                                                                                                                     \
        }                                                                                                                                    \
    };

POOL_3D_FUNCTOR(default_mp3_valid, y = etl::max_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))
POOL_3D_FUNCTOR(std_mp3_valid, y = selected_helper(etl::conv_impl::STD, (etl::max_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))))

POOL_3D_FUNCTOR(default_avgp3_valid, y = etl::avg_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))
POOL_3D_FUNCTOR(std_avgp3_valid, y = selected_helper(etl::conv_impl::STD, (etl::avg_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))))

DYN_POOL_3D_FUNCTOR(default_dyn_mp3_valid, y = etl::max_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))
DYN_POOL_3D_FUNCTOR(std_dyn_mp3_valid, y = selected_helper(etl::conv_impl::STD, (etl::max_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))))

DYN_POOL_3D_FUNCTOR(default_dyn_avgp3_valid, y = etl::avg_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))
DYN_POOL_3D_FUNCTOR(std_dyn_avgp3_valid, y = selected_helper(etl::conv_impl::STD, (etl::avg_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))))

#define MP3_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_mp3_valid)
#define MP3_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_mp3_valid)

#define AVGP3_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_avgp3_valid)
#define AVGP3_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_avgp3_valid)

#define DYN_MP3_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_dyn_mp3_valid)
#define DYN_MP3_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_dyn_mp3_valid)

#define DYN_AVGP3_TEST_CASE_SECTION_DEFAULT POOL_TEST_CASE_SECTIONS(default_dyn_avgp3_valid)
#define DYN_AVGP3_TEST_CASE_SECTION_STD POOL_TEST_CASE_SECTIONS(std_dyn_avgp3_valid)

#ifdef TEST_CUDNN
POOL_3D_FUNCTOR(cudnn_mp3_valid, y = selected_helper(etl::conv_impl::CUDNN, (etl::max_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))))
POOL_3D_FUNCTOR(cudnn_avgp3_valid, y = selected_helper(etl::conv_impl::CUDNN, (etl::avg_pool_3d<C1, C2, C3, S1, S2, S3, P1, P2, P3>(x))))

DYN_POOL_3D_FUNCTOR(cudnn_dyn_mp3_valid, y = selected_helper(etl::conv_impl::CUDNN, (etl::max_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))))
DYN_POOL_3D_FUNCTOR(cudnn_dyn_avgp3_valid, y = selected_helper(etl::conv_impl::CUDNN, (etl::avg_pool_3d(x, c1, c2, c3, s1, s2, s3, p1, p2, p3))))

#define MP3_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_mp3_valid)
#define AVGP3_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_avgp3_valid)

#define DYN_MP3_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_dyn_mp3_valid)
#define DYN_AVGP3_TEST_CASE_SECTION_CUDNN POOL_TEST_CASE_SECTIONS(cudnn_dyn_avgp3_valid)
#else
#define MP3_TEST_CASE_SECTION_CUDNN
#define AVGP3_TEST_CASE_SECTION_CUDNN

#define DYN_MP3_TEST_CASE_SECTION_CUDNN
#define DYN_AVGP3_TEST_CASE_SECTION_CUDNN
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

#define MP3_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {   \
        MP3_TEST_CASE_SECTION_DEFAULT    \
        MP3_TEST_CASE_SECTION_STD        \
        MP3_TEST_CASE_SECTION_CUDNN      \
    }                                          \
    POOL_TEST_CASE_DEFN

#define DYN_MP3_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {       \
        DYN_MP3_TEST_CASE_SECTION_DEFAULT    \
        DYN_MP3_TEST_CASE_SECTION_STD        \
        DYN_MP3_TEST_CASE_SECTION_CUDNN      \
    }                                              \
    POOL_TEST_CASE_DEFN

#define AVGP3_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {   \
        AVGP3_TEST_CASE_SECTION_DEFAULT    \
        AVGP3_TEST_CASE_SECTION_STD        \
        AVGP3_TEST_CASE_SECTION_CUDNN      \
    }                                          \
    POOL_TEST_CASE_DEFN

#define DYN_AVGP3_TEST_CASE(name, description) \
    POOL_TEST_CASE_DECL(name, description) {       \
        DYN_AVGP3_TEST_CASE_SECTION_DEFAULT    \
        DYN_AVGP3_TEST_CASE_SECTION_STD        \
        DYN_AVGP3_TEST_CASE_SECTION_CUDNN      \
    }                                              \
    POOL_TEST_CASE_DEFN
