//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

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
    /* I */ VALUES_POLICY(32, 32, 32, 32, 32, 32, 32, 32, 32, 32),
    /* K */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
    /* N */ VALUES_POLICY(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    );

using conv_2d_multi_policy_pad = NARY_POLICY(
    /* I */ VALUES_POLICY(30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
    /* K */ VALUES_POLICY(11, 11, 11, 11, 11, 11, 11, 11, 11, 11),
    /* N */ VALUES_POLICY(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
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

using conv_4d_full_policy_1 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 4, 6, 8, 10, 12, 14, 16, 18),
    /* K */ VALUES_POLICY(6, 8, 10, 12, 14, 16, 18, 20, 22),
    /* C */ VALUES_POLICY(3, 5, 7, 9, 11, 13, 15, 17, 19),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(16, 16, 16, 16, 16, 16, 16, 16, 16)
    );

using conv_4d_full_policy_2 = NARY_POLICY(
    /* N */ VALUES_POLICY(5, 10, 20, 30, 40, 60, 80, 100),
    /* K */ VALUES_POLICY(10, 10, 10, 10, 10, 10, 10, 10),
    /* C */ VALUES_POLICY(1, 1, 1, 1, 1, 1, 1, 1),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(5, 5, 5, 5, 5, 5, 5, 5)
    );

using conv_4d_full_policy_3 = NARY_POLICY(
    /* N */ VALUES_POLICY(5, 10, 20, 30, 40, 60, 80, 100),
    /* K */ VALUES_POLICY(24, 24, 24, 24, 24, 24, 24, 24),
    /* C */ VALUES_POLICY(12, 12, 12, 12, 12, 12, 12, 12),
    /* I */ VALUES_POLICY(14, 14, 14, 14, 14, 14, 14, 14),
    /* W */ VALUES_POLICY(3, 3, 3, 3, 3, 3, 3, 3)
    );

using conv_4d_full_policy_4 = NARY_POLICY(
    /* N */ VALUES_POLICY(5, 10, 20, 30, 40, 60, 80, 100),
    /* K */ VALUES_POLICY(12, 12, 12, 12, 12, 12, 12, 12),
    /* C */ VALUES_POLICY(24, 24, 24, 24, 24, 24, 24, 24),
    /* I */ VALUES_POLICY(14, 14, 14, 14, 14, 14, 14, 14),
    /* W */ VALUES_POLICY(3, 3, 3, 3, 3, 3, 3, 3)
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

// Real life config
using conv_4d_valid_policy_15 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(227, 227, 227, 227, 227, 227, 227),
    /* W */ VALUES_POLICY(15, 15, 15, 15, 15, 15, 15)
    );

// Real life config
using conv_4d_valid_policy_16 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30),
    /* I */ VALUES_POLICY(227, 227, 227, 227, 227, 227, 227),
    /* W */ VALUES_POLICY(9, 9, 9, 9, 9, 9, 9)
    );

// Policy fairer to AVX
using conv_4d_valid_policy_17 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
    );

// Policy with very small kernels but large image
using conv_4d_valid_policy_18 = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 8, 12, 16, 20, 24, 30, 40, 50, 64),
    /* K */ VALUES_POLICY(6, 10, 12, 20, 20, 30, 30, 30, 40, 40),
    /* C */ VALUES_POLICY(2, 3, 8, 10, 20, 20, 30, 40, 50, 50),
    /* I */ VALUES_POLICY(128, 128, 128, 128, 128, 128, 128, 128, 128, 128),
    /* W */ VALUES_POLICY(3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    );

// ImageNet forward policy
using imagenet_forward32_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* K */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* C */ VALUES_POLICY(3,   32,  32,  32,  32),
    /* I */ VALUES_POLICY(256, 128, 64,  32,  16),
    /* W */ VALUES_POLICY(3,   3,   3,   3,   3)
    );

// ImageNet forward policy
using imagenet_forward128_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(128,  128,  128,  128,  128),
    /* K */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* C */ VALUES_POLICY(3,   32,  32,  32,  32),
    /* I */ VALUES_POLICY(256, 128, 64,  32,  16),
    /* W */ VALUES_POLICY(3,   3,   3,   3,   3)
    );

// ImageNet backward policy
using imagenet_backward_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* K */ VALUES_POLICY(3,   32,  32,  32,  32),
    /* C */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* I */ VALUES_POLICY(256, 128, 64,  32,  16),
    /* W */ VALUES_POLICY(3,   3,   3,   3,   3)
    );

// ImageNet gradients policy
using imagenet_gradients_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(32,  32,  32,  32,  32),
    /* K */ VALUES_POLICY(32,   32,  32,  32,  32),
    /* C */ VALUES_POLICY(3,   32,  32,  32,  32),
    /* I */ VALUES_POLICY(256, 128, 64,  32,  16),
    /* W */ VALUES_POLICY(256, 128, 64,  32,  16)
    );

using conv_4d_valid_5x5_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 4, 6, 8, 10, 12, 14, 16, 18, 64),
    /* K */ VALUES_POLICY(6, 8, 10, 12, 14, 16, 18, 20, 22, 40),
    /* C */ VALUES_POLICY(3, 5, 7, 9, 11, 13, 15, 17, 19, 2),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    );

using conv_4d_valid_3x3_policy = NARY_POLICY(
    /* N */ VALUES_POLICY(2, 4, 6, 8, 10, 12, 14, 16, 18, 64),
    /* K */ VALUES_POLICY(6, 8, 10, 12, 14, 16, 18, 20, 22, 40),
    /* C */ VALUES_POLICY(3, 5, 7, 9, 11, 13, 15, 17, 19, 2),
    /* I */ VALUES_POLICY(28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    /* W */ VALUES_POLICY(3, 3, 3, 3, 3, 3, 3, 3, 3, 3)
    );

#define CONV4_BENCH(Name, Policy, Function) \
CPM_DIRECT_SECTION_TWO_PASS_NS_PF(Name, Policy, \
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }), \
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){ \
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i - w + 1, i - w + 1)); }), \
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = Function(a, b); }) \
    STDFIX_SECTION_FUNCTOR("std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::STD, Function(a, b)); }) \
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, Function(a, b)); }) \
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, Function(a, b)); }) \
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, Function(a, b)); }) \
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, Function(a, b)); }) \
)

