//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Highly inspired from the "GEMM: From Pure C to SSE Optimized Micro Kernels" course

#ifdef ETL_VECTORIZE_IMPL
#ifdef __SSE3__
#define DGEMM_SSE
#endif //__SSE3__
#endif //ETL_VECTORIZE_IMPL

#ifdef DGEMM_SSE
#include <immintrin.h>
#endif

#include "cpp_utils/likely.hpp"

#include "etl/allocator.hpp"

namespace etl {

namespace impl {

namespace eblas {

template<typename T>
struct gemm_config {
    static constexpr const std::size_t MC = 384;
    static constexpr const std::size_t KC = 384;
    static constexpr const std::size_t NC = 4096;

    static constexpr const std::size_t MR = 4;
    static constexpr const std::size_t NR = 4;
};

template<typename D>
void pack_MRxk(std::size_t k, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, double* buffer){
    constexpr const std::size_t MR = gemm_config<D>::MR;

    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = 0; i < MR; ++i) {
            buffer[j * MR + i] = A[j * a_col_stride + i * a_row_stride];
        }
    }
}

template<typename D>
void pack_A(std::size_t mc, std::size_t kc, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, double* buffer){
    constexpr const std::size_t MR = gemm_config<D>::MR;

    for (std::size_t i=0; i < mc / MR; ++i) {
        pack_MRxk(kc, A, a_row_stride, a_col_stride, buffer);
        buffer += kc*MR;
        A += MR*a_row_stride;
    }

    std::size_t mr = mc % MR;
    if (mr>0) {
        for (std::size_t j=0; j<kc; ++j) {
            for (std::size_t i=0; i<mr; ++i) {
                buffer[i] = A[i*a_row_stride];
            }

            std::fill(buffer + mr, buffer + mr + MR, D(0));
            buffer += MR;
            A += a_col_stride;
        }
    }
}

template<typename D>
void pack_kxNR(std::size_t k, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, double* buffer){
    constexpr const std::size_t NR = gemm_config<D>::NR;

    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j  =0; j < NR; ++j) {
            decltype(auto) lhs = buffer[i * NR + j];
            auto rhs = B[i * b_row_stride + j * b_col_stride];
            lhs = rhs;
        }
    }
}

template<typename D>
void pack_B(std::size_t kc, std::size_t nc, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, double* buffer){
    constexpr const std::size_t NR = gemm_config<D>::NR;

    for (std::size_t j=0; j<nc / NR; ++j) {
        pack_kxNR(kc, B, b_row_stride, b_col_stride, buffer);
        buffer += kc*NR;
        B += NR*b_col_stride;
    }

    std::size_t nr = nc % NR;
    if (nr>0) {
        for (std::size_t i=0; i<kc; ++i) {
            for (std::size_t j=0; j<nr; ++j) {
                buffer[j] = B[j*b_col_stride];
            }
            std::fill(buffer + nr, buffer + nr + NR, D(0));
            buffer += NR;
            B += b_row_stride;
        }
    }
}

#ifdef DGEMM_SSE

template<typename D>
void gemm_micro_kernel(std::size_t kc, D alpha, const double* A, const double* B, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    constexpr const std::size_t MR = gemm_config<double>::MR;
    constexpr const std::size_t NR = gemm_config<double>::NR;

    double AB[MR*NR] __attribute__ ((aligned (16)));

    __m128d tmp0 = _mm_load_pd(A);
    __m128d tmp1 = _mm_load_pd(A+2);
    __m128d tmp2 = _mm_load_pd(B);
    __m128d tmp3;
    __m128d tmp4;
    __m128d tmp5;
    __m128d tmp6;
    __m128d tmp7;

    __m128d ab_00_11 = _mm_setzero_pd();
    __m128d ab_20_31 = _mm_setzero_pd();
    __m128d ab_01_10 = _mm_setzero_pd();
    __m128d ab_21_30 = _mm_setzero_pd();
    __m128d ab_02_13 = _mm_setzero_pd();
    __m128d ab_22_33 = _mm_setzero_pd();
    __m128d ab_03_12 = _mm_setzero_pd();
    __m128d ab_23_32 = _mm_setzero_pd();

    for (std::size_t l=0; l<kc; ++l) {
        tmp3 = _mm_load_pd(B+2);

        tmp4 = _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0, 1));
        tmp5 = _mm_shuffle_pd(tmp3, tmp3, _MM_SHUFFLE2(0, 1));

        tmp6 = tmp2;
        tmp2 = _mm_mul_pd(tmp2, tmp0);
        tmp6 = _mm_mul_pd(tmp6, tmp1);
        ab_00_11 = _mm_add_pd(ab_00_11, tmp2);
        ab_20_31 = _mm_add_pd(ab_20_31, tmp6);

        tmp7 = tmp4;
        tmp4 = _mm_mul_pd(tmp4, tmp0);
        tmp7 = _mm_mul_pd(tmp7, tmp1);
        ab_01_10 = _mm_add_pd(ab_01_10, tmp4);
        ab_21_30 = _mm_add_pd(ab_21_30, tmp7);

        tmp2 = _mm_load_pd(B+4);                                // (6)
        tmp6 = tmp3;
        tmp3 = _mm_mul_pd(tmp3, tmp0);
        tmp6 = _mm_mul_pd(tmp6, tmp1);
        ab_02_13 = _mm_add_pd(ab_02_13, tmp3);
        ab_22_33 = _mm_add_pd(ab_22_33, tmp6);

        tmp7 = tmp5;
        tmp5 = _mm_mul_pd(tmp5, tmp0);
        tmp0 = _mm_load_pd(A+4);                                // (4)
        tmp7 = _mm_mul_pd(tmp7, tmp1);
        tmp1 = _mm_load_pd(A+6);                                // (5)
        ab_03_12 = _mm_add_pd(ab_03_12, tmp5);
        ab_23_32 = _mm_add_pd(ab_23_32, tmp7);

        A += 4;
        B += 4;
    }

    _mm_storel_pd(&AB[0+0*4], ab_00_11);
    _mm_storeh_pd(&AB[1+0*4], ab_01_10);
    _mm_storel_pd(&AB[2+0*4], ab_20_31);
    _mm_storeh_pd(&AB[3+0*4], ab_21_30);

    _mm_storel_pd(&AB[0+1*4], ab_01_10);
    _mm_storeh_pd(&AB[1+1*4], ab_00_11);
    _mm_storel_pd(&AB[2+1*4], ab_21_30);
    _mm_storeh_pd(&AB[3+1*4], ab_20_31);

    _mm_storel_pd(&AB[0+2*4], ab_02_13);
    _mm_storeh_pd(&AB[1+2*4], ab_03_12);
    _mm_storel_pd(&AB[2+2*4], ab_22_33);
    _mm_storeh_pd(&AB[3+2*4], ab_23_32);

    _mm_storel_pd(&AB[0+3*4], ab_03_12);
    _mm_storeh_pd(&AB[1+3*4], ab_02_13);
    _mm_storel_pd(&AB[2+3*4], ab_23_32);
    _mm_storeh_pd(&AB[3+3*4], ab_22_33);

    if (cpp_likely(beta==0.0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] *= beta;
            }
        }
    }

    if (cpp_likely(alpha==1.0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] += AB[i+j*MR];
            }
        }
    } else {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] += alpha*AB[i+j*MR];
            }
        }
    }
}

#else //!DGEMM_SSE

template<typename D>
void gemm_micro_kernel(std::size_t kc, D alpha, const double* A, const double* B, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    constexpr const std::size_t MR = gemm_config<D>::MR;
    constexpr const std::size_t NR = gemm_config<D>::NR;

    double AB[MR*NR];

    std::fill(AB, AB + MR * NR, D(0));

    for (std::size_t l=0; l<kc; ++l) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                AB[i+j*MR] += A[i]*B[j];
            }
        }
        A += MR;
        B += NR;
    }

    if (cpp_likely(beta==0.0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] = D(0);
            }
        }
    } else if (beta!=1.0) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] *= beta;
            }
        }
    }

    if (cpp_likely(alpha==1.0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] += AB[i+j*MR];
            }
        }
    } else {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*c_row_stride+j*c_col_stride] += alpha*AB[i+j*MR];
            }
        }
    }
}

#endif

template<typename D>
void dgeaxpy(std::size_t m, std::size_t n, D alpha, const D  *X, std::size_t incRowX, std::size_t incColX, D* Y, std::size_t incRowY, std::size_t incColY){
    if (cpp_likely(alpha==D(1))) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    }
}

template<typename D>
void dgescal(std::size_t m, std::size_t n, D beta, D* X, std::size_t incRowX, std::size_t incColX){
    if (cpp_likely(beta==D(0))) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = D(0);
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= beta;
            }
        }
    }
}

template<typename D>
void gemm_macro_kernel(std::size_t mc, std::size_t nc, std::size_t kc, D alpha, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride, double* _A, double* _B, D* _C){
    constexpr const std::size_t MR = gemm_config<D>::MR;
    constexpr const std::size_t NR = gemm_config<D>::NR;

    auto mp = (mc+MR-1) / MR;
    auto np = (nc+NR-1) / NR;

    auto _mr = mc % MR;
    auto _nr = nc % NR;

    for (std::size_t j=0; j<np; ++j) {
        auto nr = (j!=np-1 || _nr==0) ? NR : _nr;

        for (std::size_t i=0; i<mp; ++i) {
            auto mr = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                gemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], beta, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
            } else {
                gemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], D(0), _C, 1, MR);
                dgescal(mr, nr, beta, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
                dgeaxpy(mr, nr, D(1), _C, 1, MR, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
            }
        }
    }
}

template<typename D>
void gemm_nn(std::size_t m, std::size_t n, std::size_t k, D alpha, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    if (alpha==0.0 || k==0) {
        dgescal(m, n, beta, C, c_row_stride, c_col_stride);
        return;
    }

    constexpr const std::size_t MC = gemm_config<D>::MC;
    constexpr const std::size_t NC = gemm_config<D>::NC;
    constexpr const std::size_t KC = gemm_config<D>::KC;

    constexpr const std::size_t MR = gemm_config<D>::MR;
    constexpr const std::size_t NR = gemm_config<D>::NR;

    auto _A = allocate<double>(MC * KC);
    auto _B = allocate<double>(KC * NC);
    auto _C = allocate<D>(MR * NR);

    auto mb = (m+MC-1) / MC;
    auto nb = (n+NC-1) / NC;
    auto kb = (k+KC-1) / KC;

    std::size_t _mc = m % MC;
    std::size_t _nc = n % NC;
    std::size_t _kc = k % KC;

    for (std::size_t j=0; j<nb; ++j) {
        auto nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (std::size_t l=0; l<kb; ++l) {
            auto kc = (l!=kb-1 || _kc==0) ? KC   : _kc;
            auto _beta = (l==0) ? beta : D(1);

            pack_B(kc, nc, &B[l*KC*b_row_stride+j*NC*b_col_stride], b_row_stride, b_col_stride, _B.get());

            for (std::size_t i=0; i<mb; ++i) {
                auto mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc, &A[i*MC*a_row_stride+l*KC*a_col_stride], a_row_stride, a_col_stride, _A.get());

                gemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i*MC*c_row_stride+j*NC*c_col_stride], c_row_stride, c_col_stride, _A.get(), _B.get(), _C.get());
            }
        }
    }
}

template<typename A, typename B, typename C, cpp_enable_if(all_dma<A,B,C>::value && !is_complex<A>::value)>
void gemm(A&& a, B&& b, C&& c){
    gemm_nn(
        etl::dim<0>(a), etl::dim<1>(b), etl::dim<1>(a),
        value_t<A>(1.0),
        a.memory_start(), row_stride(a), col_stride(a),
        b.memory_start(), row_stride(b), col_stride(b),
        value_t<A>(0.0),
        c.memory_start(), row_stride(c), col_stride(c)
    );
}

template<typename A, typename B, typename C, cpp_enable_if(!all_dma<A,B,C>::value || is_complex<A>::value)>
void gemm(A&& /*a*/, B&& /*b*/, C&& /*c*/){}

} //end of namespace eblas

} //end of namespace impl

} //end of namespace etl
