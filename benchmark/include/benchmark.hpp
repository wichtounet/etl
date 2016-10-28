//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

#define CPM_NO_RANDOMIZATION            //Randomly initialize only once
#define CPM_AUTO_STEPS                  //Enable steps estimation system
#define CPM_STEP_ESTIMATION_MIN 0.025   //Run during 0.025 seconds for estimating steps
#define CPM_RUNTIME_TARGET 0.9          //Run each test during 0.9 seconds

#include "cpm/cpm.hpp"

#ifdef ETL_VECTORIZE_IMPL
#ifdef __SSE3__
#define TEST_SSE
#endif
#ifdef __AVX__
#define TEST_AVX
#endif
#endif

#ifdef ETL_MKL_MODE
#define TEST_MKL
#endif

#ifdef ETL_CUBLAS_MODE
#define TEST_CUBLAS
#endif

#ifdef ETL_CUFFT_MODE
#define TEST_CUFFT
#endif

#ifdef ETL_CUDNN_MODE
#define TEST_CUDNN
#endif

#ifdef ETL_BLAS_MODE
#define TEST_BLAS
#endif

#ifdef ETL_BENCH_STRASSEN
#define TEST_STRASSEN
#endif

using smat_cm = etl::dyn_matrix_cm<float>;
using dmat_cm = etl::dyn_matrix_cm<double>;
using cmat_cm = etl::dyn_matrix_cm<std::complex<float>>;
using zmat_cm = etl::dyn_matrix_cm<std::complex<double>>;

using dvec = etl::dyn_vector<double>;
using dmat = etl::dyn_matrix<double>;
using dmat2 = etl::dyn_matrix<double, 2>;
using dmat3 = etl::dyn_matrix<double, 3>;
using dmat4 = etl::dyn_matrix<double, 4>;
using dmat5 = etl::dyn_matrix<double, 5>;

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;
using smat3 = etl::dyn_matrix<float, 3>;
using smat4 = etl::dyn_matrix<float, 4>;

using cvec = etl::dyn_vector<std::complex<float>>;
using cmat = etl::dyn_matrix<std::complex<float>>;
using cmat3 = etl::dyn_matrix<std::complex<float>, 3>;
using ecvec = etl::dyn_vector<etl::complex<float>>;

using zvec = etl::dyn_vector<std::complex<double>>;
using zmat = etl::dyn_matrix<std::complex<double>>;
using zmat3 = etl::dyn_matrix<std::complex<double>, 3>;
using ezvec = etl::dyn_vector<etl::complex<double>>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000,3000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

using outer_policy = NARY_POLICY(
    VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000),
    VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000)
    );

using dot_policy = VALUES_POLICY(100, 500, 1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 10000000);

using large_vector_policy = VALUES_POLICY(10, 100, 1000, 10000, 1000000, 10000000, 100000000);

using conv_1d_large_policy = NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000));
using conv_2d_large_policy = NARY_POLICY(VALUES_POLICY(100, 105, 110, 115, 120, 125, 130, 135, 140), VALUES_POLICY(50, 50, 55, 55, 60, 60, 65, 65, 70));

// Use some common kernels in machine learning
using conv_2d_real_policy = NARY_POLICY(
    VALUES_POLICY(28, 28, 28, 28, 32, 32, 32, 32, 227, 227, 227, 227),
    VALUES_POLICY(3, 5, 7, 9, 3, 5, 7, 9, 3, 5, 11, 13));

using conv_2d_small_policy = NARY_POLICY(
    VALUES_POLICY(10, 10, 12, 12, 16, 16, 20, 20, 28, 28, 28, 32, 100),
    VALUES_POLICY(5,  9,  5,  9,  5,  9,  5,  9,  5,  9,  17, 16, 24));

using conv_2d_multi_policy = NARY_POLICY(
    VALUES_POLICY(32, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
    VALUES_POLICY(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    );

using conv_2d_multi_multi_policy = NARY_POLICY(
    VALUES_POLICY(32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
    VALUES_POLICY(2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 30),
    VALUES_POLICY(2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 30)
    );

using conv_4d_valid_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 4, 6, 8, 10, 12, 14, 16, 18, 64),
    /* K */ VALUES_POLICY(6, 8, 10, 12, 14, 16, 18, 20, 22, 40),
    /* C */ VALUES_POLICY(3, 5, 7, 9, 11, 13, 15, 17, 19, 2),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16, 12)
    );

using conv_4d_full_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 4, 6, 8, 10, 12, 14, 16, 18),
    /* K */ VALUES_POLICY(6, 8, 10, 12, 14, 16, 18, 20, 22),
    /* C */ VALUES_POLICY(3, 5, 7, 9, 11, 13, 15, 17, 19),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16)
    );

// Policy fairer to AVX
using conv_4d_valid_policy_1 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16, 16)
    );

// Policy fairer to SSE
using conv_4d_valid_policy_2 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(12, 12, 12, 12, 12, 12, 12, 12, 12, 12)
    );

// Policy with very small kernels
using conv_4d_valid_policy_3 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    );

// Policy with very small kernels
using conv_4d_valid_policy_4 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    );

// Policy with very small kernels
using conv_4d_valid_policy_5 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
    );

// Policy with small kernels
using conv_4d_valid_policy_6 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
    );

// Policy with larger images
using conv_4d_valid_policy_7 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    );

// Policy with larger images
using conv_4d_valid_policy_8 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
    );

// Policy with larger images and larger kernels
using conv_4d_valid_policy_9 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(12, 12, 12, 12, 12, 12, 12)
    );

// Policy with larger images and larger kernels
using conv_4d_valid_policy_10 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16)
    );

// Policy with larger images and larger kernels
using conv_4d_valid_policy_11 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(32, 32, 32, 32, 32, 32, 32, 32)
    );

// Policy with larger images and larger kernels
using conv_4d_valid_policy_12 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(33, 33, 33, 33, 33, 33, 33)
    );

// Policy with larger images and larger kernels
using conv_4d_valid_policy_13 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(64, 64, 64, 64, 64, 64, 64)
    );

// Policy with very large images
using conv_4d_valid_policy_14 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(256, 256, 256, 256, 256, 256, 256),
    /* W */ VALUES_POLICY(5, 5, 5, 5, 5, 5, 5)
    );

using pmp_policy = VALUES_POLICY(100, 120, 140, 160, 180, 200);

using fast_policy = VALUES_POLICY(1);

using fft_1d_policy = VALUES_POLICY(10, 100, 1000, 10000, 100000, 500000);
using fft_1d_policy_2 = VALUES_POLICY(16, 64, 256, 1024, 16384, 131072, 1048576, 2097152);
using fft_1d_many_policy = VALUES_POLICY(10, 50, 100, 500, 1000, 5000, 10000, 50000);

using fft_2d_policy = NARY_POLICY(
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048));

using fft_2d_many_policy = NARY_POLICY(
    VALUES_POLICY(4, 8, 16, 24, 32, 64, 96, 128, 256),
    VALUES_POLICY(4, 8, 16, 24, 32, 64, 96, 128, 256));

using square_policy = NARY_POLICY(
    VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));
using small_square_policy = NARY_POLICY(
    VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500),
    VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500));

using gemv_policy = NARY_POLICY(
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000),
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000));

using trans_sub_policy = VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using trans_policy = NARY_POLICY(
    VALUES_POLICY(100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000),
    VALUES_POLICY(100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000));

using large_kernel_policy = NARY_POLICY(
    VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000),
    VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000));

using small_kernel_policy = NARY_POLICY(
    VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000),
    VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using large_kernel_policy_2d = NARY_POLICY(
    VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170),
    VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85));

using small_kernel_policy_2d = NARY_POLICY(
    VALUES_POLICY(100, 150, 200, 250, 300, 350, 400),
    VALUES_POLICY(10, 15, 20, 25, 30, 35, 40));

#ifdef TEST_SSE
#define SSE_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define SSE_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_AVX
#define AVX_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define AVX_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_MKL
#define MKL_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define MKL_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_BLAS
#define BLAS_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define BLAS_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_CUBLAS
#define CUBLAS_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define CUBLAS_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_CUFFT
#define CUFFT_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define CUFFT_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_CUDNN
#define CUDNN_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define CUDNN_SECTION_FUNCTOR(name, ...)
#endif
