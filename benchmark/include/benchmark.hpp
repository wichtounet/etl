//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

// Check if VEC can be benchmarked
#ifdef ETL_VECTORIZE_IMPL
#ifdef __AVX__
#define TEST_VEC
#elif defined(__SSE3__)
#define TEST_VEC
#endif
#endif

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

// TODO Instead of using this fix, do the real fix inside CPM
#if !defined(TEST_VEC) && !defined(TEST_BLAS)
#define TEST_STDFIX
#endif

using i8vec = etl::dyn_vector<int8_t>;
using i16vec = etl::dyn_vector<int16_t>;
using i32vec = etl::dyn_vector<int32_t>;
using i64vec = etl::dyn_vector<int64_t>;

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;
using smat3 = etl::dyn_matrix<float, 3>;
using smat4 = etl::dyn_matrix<float, 4>;

using dvec = etl::dyn_vector<double>;
using dmat = etl::dyn_matrix<double>;
using dmat2 = etl::dyn_matrix<double, 2>;
using dmat3 = etl::dyn_matrix<double, 3>;
using dmat4 = etl::dyn_matrix<double, 4>;
using dmat5 = etl::dyn_matrix<double, 5>;

using cvec = etl::dyn_vector<std::complex<float>>;
using cmat = etl::dyn_matrix<std::complex<float>>;
using cmat3 = etl::dyn_matrix<std::complex<float>, 3>;
using ecvec = etl::dyn_vector<etl::complex<float>>;

using zvec = etl::dyn_vector<std::complex<double>>;
using zmat = etl::dyn_matrix<std::complex<double>>;
using zmat3 = etl::dyn_matrix<std::complex<double>, 3>;
using ezvec = etl::dyn_vector<etl::complex<double>>;

using smat_cm = etl::dyn_matrix_cm<float>;
using dmat_cm = etl::dyn_matrix_cm<double>;
using cmat_cm = etl::dyn_matrix_cm<std::complex<float>>;
using zmat_cm = etl::dyn_matrix_cm<std::complex<double>>;

using smat_rm = etl::dyn_matrix<float>;
using dmat_rm = etl::dyn_matrix<double>;
using cmat_rm = etl::dyn_matrix<std::complex<float>>;
using zmat_rm = etl::dyn_matrix<std::complex<double>>;

using svec_cm = etl::dyn_matrix_cm<float, 1>;
using dvec_cm = etl::dyn_matrix_cm<double, 1>;
using cvec_cm = etl::dyn_matrix_cm<std::complex<float>, 1>;
using zvec_cm = etl::dyn_matrix_cm<std::complex<double>, 1>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000,3000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

using dot_policy = VALUES_POLICY(100, 500, 1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000, 10000000);

using large_vector_policy = VALUES_POLICY(10, 100, 1000, 10000, 1000000, 10000000, 100000000);

using mp_policy = VALUES_POLICY(28, 32, 64, 100, 128, 140, 160, 180, 200, 400, 600);
using pmp_policy = VALUES_POLICY(100, 120, 140, 160, 180, 200, 400, 600, 800, 1000);
using pmp_policy_3 = VALUES_POLICY(10, 20, 30, 40, 50, 60, 80, 90, 100);

using fast_policy = VALUES_POLICY(1);

using fft_1d_policy = VALUES_POLICY(100, 1000, 10000, 100000, 1000000);
using fft_1d_policy_2 = VALUES_POLICY(16, 64, 256, 1024, 16384, 131072, 1048576, 2097152);
using fft_1d_many_policy = VALUES_POLICY(10, 50, 100, 500, 1000, 5000, 10000, 50000);

using fft_2d_policy = NARY_POLICY(
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048));

using fft_2d_many_policy = NARY_POLICY(
    VALUES_POLICY(4, 8, 16, 24, 32, 64, 96, 128, 256),
    VALUES_POLICY(4, 8, 16, 24, 32, 64, 96, 128, 256));

using gemm_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using sgemm_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200));

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

#ifdef TEST_STDFIX
#define STDFIX_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define STDFIX_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_VEC
#define VEC_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define VEC_SECTION_FUNCTOR(name, ...)
#endif

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
