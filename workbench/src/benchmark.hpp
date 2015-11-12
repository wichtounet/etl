//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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
#define CPM_STEP_ESTIMATION_MIN 0.05    //Run during 0.05 seconds for estimating steps
#define CPM_RUNTIME_TARGET 1.0          //Run each test during 1.0 seconds

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

#ifdef ETL_BLAS_MODE
#define TEST_BLAS
#endif

#ifdef ETL_BENCH_STRASSEN
#define TEST_STRASSEN
#endif

#ifdef ETL_BENCH_MMUL_CONV
#define TEST_MMUL_CONV
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

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;

using cvec = etl::dyn_vector<std::complex<float>>;
using cmat = etl::dyn_matrix<std::complex<float>>;

using zvec = etl::dyn_vector<std::complex<double>>;
using zmat = etl::dyn_matrix<std::complex<double>>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000,3000,4000,8000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

using conv_1d_large_policy = NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000));
using conv_2d_large_policy = NARY_POLICY(VALUES_POLICY(100, 105, 110, 115, 120, 125, 130, 135, 140), VALUES_POLICY(50, 50, 55, 55, 60, 60, 65, 65, 70));

using fft_1d_policy = VALUES_POLICY(10, 100, 1000, 10000, 100000, 500000);
using fft_1d_policy_2 = VALUES_POLICY(16, 64, 256, 1024, 16384, 131072, 1048576, 2097152);
using fft_1d_many_policy = VALUES_POLICY(10, 50, 100, 500, 1000, 5000, 10000);

using fft_2d_policy = NARY_POLICY(
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048));

using sigmoid_policy = VALUES_POLICY(250, 500, 750, 1000, 1250, 1500, 1750, 2000);

using small_square_policy = NARY_POLICY(VALUES_POLICY(50, 100, 150, 200, 250, 300, 350, 400, 450, 500), VALUES_POLICY(50, 100, 150, 200, 250, 300, 350, 400, 450, 500));
using square_policy = NARY_POLICY(VALUES_POLICY(50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using gemv_policy = NARY_POLICY(
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000),
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000));

using trans_sub_policy = VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using trans_policy = NARY_POLICY(
    VALUES_POLICY(100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000),
    VALUES_POLICY(100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000, 1100));

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

#ifdef TEST_MMUL_CONV
#define MC_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define MC_SECTION_FUNCTOR(name, ...)
#endif
