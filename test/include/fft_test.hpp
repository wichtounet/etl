//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define FFT_FUNCTOR(name, ...)            \
    struct name {                         \
        template <typename A, typename C> \
        static void apply(A&& a, C& c) {  \
            __VA_ARGS__;                  \
        }                                 \
    };

FFT_FUNCTOR(default_fft1, c = etl::fft_1d(a))
FFT_FUNCTOR(std_fft1, c = selected_helper(etl::fft_impl::STD, etl::fft_1d(a)))

FFT_FUNCTOR(default_fft1_many, c = etl::fft_1d_many(a))
FFT_FUNCTOR(std_fft1_many, c = selected_helper(etl::fft_impl::STD, etl::fft_1d_many(a)))

FFT_FUNCTOR(default_ifft1_many, c = etl::ifft_1d_many(a))
FFT_FUNCTOR(std_ifft1_many, c = selected_helper(etl::fft_impl::STD, etl::ifft_1d_many(a)))

FFT_FUNCTOR(default_ifft1, c = etl::ifft_1d(a))
FFT_FUNCTOR(std_ifft1, c = selected_helper(etl::fft_impl::STD, etl::ifft_1d(a)))

FFT_FUNCTOR(default_ifft1_real, c = etl::ifft_1d_real(a))
FFT_FUNCTOR(std_ifft1_real, c = selected_helper(etl::fft_impl::STD, etl::ifft_1d_real(a)))

FFT_FUNCTOR(default_fft2, c = etl::fft_2d(a))
FFT_FUNCTOR(std_fft2, c = selected_helper(etl::fft_impl::STD, etl::fft_2d(a)))

FFT_FUNCTOR(default_ifft2, c = etl::ifft_2d(a))
FFT_FUNCTOR(std_ifft2, c = selected_helper(etl::fft_impl::STD, etl::ifft_2d(a)))

FFT_FUNCTOR(default_ifft2_real, c = etl::ifft_2d_real(a))
FFT_FUNCTOR(std_ifft2_real, c = selected_helper(etl::fft_impl::STD, etl::ifft_2d_real(a)))

FFT_FUNCTOR(default_fft2_many, c = etl::fft_2d_many(a))
FFT_FUNCTOR(std_fft2_many, c = selected_helper(etl::fft_impl::STD, etl::fft_2d_many(a)))

FFT_FUNCTOR(default_ifft2_many, c = etl::ifft_2d_many(a))
FFT_FUNCTOR(std_ifft2_many, c = selected_helper(etl::fft_impl::STD, etl::ifft_2d_many(a)))

#define FFT1_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_fft1)
#define FFT1_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_fft1)

#define FFT1_MANY_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_fft1_many)
#define FFT1_MANY_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_fft1_many)

#define IFFT1_MANY_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft1_many)
#define IFFT1_MANY_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft1_many)

#define IFFT1_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft1)
#define IFFT1_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft1)

#define IFFT1_REAL_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft1_real)
#define IFFT1_REAL_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft1_real)

#define FFT2_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_fft2)
#define FFT2_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_fft2)

#define FFT2_MANY_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_fft2_many)
#define FFT2_MANY_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_fft2_many)

#define IFFT2_MANY_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft2_many)
#define IFFT2_MANY_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft2_many)

#define IFFT2_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft2)
#define IFFT2_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft2)

#define IFFT2_REAL_TEST_CASE_SECTION_DEFAULT FFT_TEST_CASE_SECTIONS(default_ifft2_real)
#define IFFT2_REAL_TEST_CASE_SECTION_STD FFT_TEST_CASE_SECTIONS(std_ifft2_real)

#ifdef ETL_MKL_MODE
FFT_FUNCTOR(mkl_fft1, c = selected_helper(etl::fft_impl::MKL, etl::fft_1d(a)))
FFT_FUNCTOR(mkl_fft1_many, c = selected_helper(etl::fft_impl::MKL, etl::fft_1d_many(a)))
FFT_FUNCTOR(mkl_ifft1_many, c = selected_helper(etl::fft_impl::MKL, etl::ifft_1d_many(a)))
FFT_FUNCTOR(mkl_ifft1, c = selected_helper(etl::fft_impl::MKL, etl::ifft_1d(a)))
FFT_FUNCTOR(mkl_ifft1_real, c = selected_helper(etl::fft_impl::MKL, etl::ifft_1d_real(a)))
FFT_FUNCTOR(mkl_fft2, c = selected_helper(etl::fft_impl::MKL, etl::fft_2d(a)))
FFT_FUNCTOR(mkl_ifft2, c = selected_helper(etl::fft_impl::MKL, etl::ifft_2d(a)))
FFT_FUNCTOR(mkl_ifft2_real, c = selected_helper(etl::fft_impl::MKL, etl::ifft_2d_real(a)))
FFT_FUNCTOR(mkl_fft2_many, c = selected_helper(etl::fft_impl::MKL, etl::fft_2d_many(a)))
FFT_FUNCTOR(mkl_ifft2_many, c = selected_helper(etl::fft_impl::MKL, etl::ifft_2d_many(a)))
#define FFT1_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_fft1)
#define FFT2_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_fft2)
#define FFT1_MANY_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_fft1_many)
#define FFT2_MANY_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_fft2_many)
#define IFFT1_MANY_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft1_many)
#define IFFT2_MANY_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft2_many)
#define IFFT1_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft1)
#define IFFT2_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft2)
#define IFFT1_REAL_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft1_real)
#define IFFT2_REAL_TEST_CASE_SECTION_MKL FFT_TEST_CASE_SECTIONS(mkl_ifft2_real)
#else
#define FFT1_TEST_CASE_SECTION_MKL
#define FFT2_TEST_CASE_SECTION_MKL
#define FFT1_MANY_TEST_CASE_SECTION_MKL
#define FFT2_MANY_TEST_CASE_SECTION_MKL
#define IFFT1_MANY_TEST_CASE_SECTION_MKL
#define IFFT2_MANY_TEST_CASE_SECTION_MKL
#define IFFT1_TEST_CASE_SECTION_MKL
#define IFFT2_TEST_CASE_SECTION_MKL
#define IFFT1_REAL_TEST_CASE_SECTION_MKL
#define IFFT2_REAL_TEST_CASE_SECTION_MKL
#endif

#ifdef ETL_CUFFT_MODE
FFT_FUNCTOR(cufft_fft1, c = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d(a)))
FFT_FUNCTOR(cufft_fft1_many, c = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d_many(a)))
FFT_FUNCTOR(cufft_ifft1_many, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_1d_many(a)))
FFT_FUNCTOR(cufft_ifft1, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_1d(a)))
FFT_FUNCTOR(cufft_ifft1_real, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_1d_real(a)))
FFT_FUNCTOR(cufft_fft2, c = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d(a)))
FFT_FUNCTOR(cufft_ifft2, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d(a)))
FFT_FUNCTOR(cufft_ifft2_real, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d_real(a)))
FFT_FUNCTOR(cufft_fft2_many, c = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d_many(a)))
FFT_FUNCTOR(cufft_ifft2_many, c = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d_many(a)))
#define FFT1_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_fft1)
#define FFT2_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_fft2)
#define FFT1_MANY_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_fft1_many)
#define FFT2_MANY_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_fft2_many)
#define IFFT1_MANY_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft1_many)
#define IFFT2_MANY_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft2_many)
#define IFFT1_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft1)
#define IFFT2_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft2)
#define IFFT1_REAL_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft1_real)
#define IFFT2_REAL_TEST_CASE_SECTION_CUFFT FFT_TEST_CASE_SECTIONS(cufft_ifft2_real)
#else
#define FFT1_TEST_CASE_SECTION_CUFFT
#define FFT2_TEST_CASE_SECTION_CUFFT
#define FFT1_MANY_TEST_CASE_SECTION_CUFFT
#define FFT2_MANY_TEST_CASE_SECTION_CUFFT
#define IFFT1_MANY_TEST_CASE_SECTION_CUFFT
#define IFFT2_MANY_TEST_CASE_SECTION_CUFFT
#define IFFT1_TEST_CASE_SECTION_CUFFT
#define IFFT2_TEST_CASE_SECTION_CUFFT
#define IFFT1_REAL_TEST_CASE_SECTION_CUFFT
#define IFFT2_REAL_TEST_CASE_SECTION_CUFFT
#endif

#define FFT_TEST_CASE_DECL(name, description)                                                 \
    template <typename T, typename Impl>                                                      \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)(); \
    ETL_TEST_CASE(name, description)

#define FFT_TEST_CASE_SECTION(Tn, Impln)                                                         \
    ETL_SECTION(#Tn "_" #Impln) {                                                                    \
        UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)<Tn, Impln>(); \
    }

#define FFT_TEST_CASE_DEFN               \
    template <typename T, typename Impl> \
    static void UNIQUE_NAME(____C_A_T_C_H____T_E_M_P_L_A_TE____T_E_S_T____)()

#define FFT_TEST_CASE_SECTIONS(S1) \
    FFT_TEST_CASE_SECTION(float, S1)   \
    FFT_TEST_CASE_SECTION(double, S1)

#define FFT1_TEST_CASE(name, description)   \
    FFT_TEST_CASE_DECL(name, description) { \
        FFT1_TEST_CASE_SECTION_DEFAULT      \
        FFT1_TEST_CASE_SECTION_STD          \
        FFT1_TEST_CASE_SECTION_MKL          \
        FFT1_TEST_CASE_SECTION_CUFFT        \
    }                                       \
    FFT_TEST_CASE_DEFN

#define FFT1_MANY_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {    \
        FFT1_MANY_TEST_CASE_SECTION_DEFAULT    \
        FFT1_MANY_TEST_CASE_SECTION_STD        \
        FFT1_MANY_TEST_CASE_SECTION_MKL        \
        FFT1_MANY_TEST_CASE_SECTION_CUFFT      \
    }                                          \
    FFT_TEST_CASE_DEFN

#define IFFT1_MANY_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {     \
        IFFT1_MANY_TEST_CASE_SECTION_DEFAULT    \
        IFFT1_MANY_TEST_CASE_SECTION_STD        \
        IFFT1_MANY_TEST_CASE_SECTION_MKL        \
        IFFT1_MANY_TEST_CASE_SECTION_CUFFT      \
    }                                           \
    FFT_TEST_CASE_DEFN

#define IFFT1_TEST_CASE(name, description)  \
    FFT_TEST_CASE_DECL(name, description) { \
        IFFT1_TEST_CASE_SECTION_DEFAULT     \
        IFFT1_TEST_CASE_SECTION_STD         \
        IFFT1_TEST_CASE_SECTION_MKL         \
        IFFT1_TEST_CASE_SECTION_CUFFT       \
    }                                       \
    FFT_TEST_CASE_DEFN

#define IFFT1_REAL_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {     \
        IFFT1_REAL_TEST_CASE_SECTION_DEFAULT    \
        IFFT1_REAL_TEST_CASE_SECTION_STD        \
        IFFT1_REAL_TEST_CASE_SECTION_MKL        \
        IFFT1_REAL_TEST_CASE_SECTION_CUFFT      \
    }                                           \
    FFT_TEST_CASE_DEFN

#define FFT2_TEST_CASE(name, description)   \
    FFT_TEST_CASE_DECL(name, description) { \
        FFT2_TEST_CASE_SECTION_DEFAULT      \
        FFT2_TEST_CASE_SECTION_STD          \
        FFT2_TEST_CASE_SECTION_MKL          \
        FFT2_TEST_CASE_SECTION_CUFFT        \
    }                                       \
    FFT_TEST_CASE_DEFN

#define FFT2_MANY_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {    \
        FFT2_MANY_TEST_CASE_SECTION_DEFAULT    \
        FFT2_MANY_TEST_CASE_SECTION_STD        \
        FFT2_MANY_TEST_CASE_SECTION_MKL        \
        FFT2_MANY_TEST_CASE_SECTION_CUFFT      \
    }                                          \
    FFT_TEST_CASE_DEFN

#define IFFT2_MANY_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {     \
        IFFT2_MANY_TEST_CASE_SECTION_DEFAULT    \
        IFFT2_MANY_TEST_CASE_SECTION_STD        \
        IFFT2_MANY_TEST_CASE_SECTION_MKL        \
        IFFT2_MANY_TEST_CASE_SECTION_CUFFT      \
    }                                           \
    FFT_TEST_CASE_DEFN

#define IFFT2_TEST_CASE(name, description)  \
    FFT_TEST_CASE_DECL(name, description) { \
        IFFT2_TEST_CASE_SECTION_DEFAULT     \
        IFFT2_TEST_CASE_SECTION_STD         \
        IFFT2_TEST_CASE_SECTION_MKL         \
        IFFT2_TEST_CASE_SECTION_CUFFT       \
    }                                       \
    FFT_TEST_CASE_DEFN

#define IFFT2_REAL_TEST_CASE(name, description) \
    FFT_TEST_CASE_DECL(name, description) {     \
        IFFT2_REAL_TEST_CASE_SECTION_DEFAULT    \
        IFFT2_REAL_TEST_CASE_SECTION_STD        \
        IFFT2_REAL_TEST_CASE_SECTION_MKL        \
        IFFT2_REAL_TEST_CASE_SECTION_CUFFT      \
    }                                           \
    FFT_TEST_CASE_DEFN
