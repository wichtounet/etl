//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Highly inspired from the "GEMM: From Pure C to SSE Optimized Micro Kernels" course

#ifndef ETL_DGEMM_HPP
#define ETL_DGEMM_HPP

#include "cpp_utils/likely.hpp"

namespace etl {

namespace dgemm_detail {

static constexpr const std::size_t MC = 384;
static constexpr const std::size_t KC = 384;
static constexpr const std::size_t NC = 4096;

static constexpr const std::size_t MR = 4;
static constexpr const std::size_t NR = 4;

//TODO Should not be double but templated
thread_local static double _A[MC*KC];
thread_local static double _B[KC*NC];
thread_local static double _C[MR*NR];

template<typename D>
void pack_MRxk(std::size_t k, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, D* buffer){
    for (std::size_t j=0; j<k; ++j) {
        for (std::size_t i=0; i<MR; ++i) {
            buffer[i] = A[i*a_row_stride];
        }

        buffer += MR;
        A += a_col_stride;
    }
}

template<typename D>
void pack_A(std::size_t mc, std::size_t kc, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, D* buffer){
    for (std::size_t i=0; i < mc / MR; ++i) {
        pack_MRxk(kc, A, a_row_stride, a_col_stride, buffer);
        buffer += kc*MR;
        A += MR*a_row_stride;
    }

    auto mr = mc % MR;
    if (mr>0) {
        for (std::size_t j=0; j<kc; ++j) {
            for (std::size_t i=0; i<mr; ++i) {
                buffer[i] = A[i*a_row_stride];
            }
            std::fill(buffer + mr, buffer + mr + MR, 0.0);
            buffer += MR;
            A += a_col_stride;
        }
    }
}

template<typename D>
void pack_kxNR(std::size_t k, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, D* buffer){
    for (std::size_t i=0; i<k; ++i) {
        for (std::size_t j=0; j<NR; ++j) {
            buffer[j] = B[j*b_col_stride];
        }
        buffer += NR;
        B += b_row_stride;
    }
}

template<typename D>
void pack_B(std::size_t kc, std::size_t nc, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, D* buffer){
    for (std::size_t j=0; j<nc / NR; ++j) {
        pack_kxNR(kc, B, b_row_stride, b_col_stride, buffer);
        buffer += kc*NR;
        B += NR*b_col_stride;
    }

    auto nr = nc % NR;
    if (nr>0) {
        for (std::size_t i=0; i<kc; ++i) {
            for (std::size_t j=0; j<nr; ++j) {
                buffer[j] = B[j*b_col_stride];
            }
            std::fill(buffer + nr, buffer + nr + NR, 0.0);
            buffer += NR;
            B += b_row_stride;
        }
    }
}

template<typename D>
void dgemm_micro_kernel(std::size_t kc, D alpha, const D* A, const D* B, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    D AB[MR*NR];

    std::fill(AB, AB + MR * MR, 0.0);

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

template<typename D>
void dgeaxpy(std::size_t m, std::size_t n, D alpha, const D  *X, std::size_t incRowX, std::size_t incColX, D* Y, std::size_t incRowY, std::size_t incColY){
    if (cpp_likely(alpha==1.0)) {
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
    if (cpp_likely(beta==0.0)) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
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
void dgemm_macro_kernel(std::size_t mc, std::size_t nc, std::size_t kc, D alpha, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    auto mp = (mc+MR-1) / MR;
    auto np = (nc+NR-1) / NR;

    auto _mr = mc % MR;
    auto _nr = nc % NR;

    for (std::size_t j=0; j<np; ++j) {
        auto nr = (j!=np-1 || _nr==0) ? NR : _nr;

        for (std::size_t i=0; i<mp; ++i) {
            auto mr = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], beta, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
            } else {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], 0.0, _C, 1, MR);
                dgescal(mr, nr, beta, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
                dgeaxpy(mr, nr, 1.0, _C, 1, MR, &C[i*MR*c_row_stride+j*NR*c_col_stride], c_row_stride, c_col_stride);
            }
        }
    }
}

template<typename D>
void dgemm_nn(std::size_t m, std::size_t n, std::size_t k, D alpha, const D* A, std::size_t a_row_stride, std::size_t a_col_stride, const D* B, std::size_t b_row_stride, std::size_t b_col_stride, D beta, D* C, std::size_t c_row_stride, std::size_t c_col_stride){
    if (alpha==0.0 || k==0) {
        dgescal(m, n, beta, C, c_row_stride, c_col_stride);
        return;
    }

    auto mb = (m+MC-1) / MC;
    auto nb = (n+NC-1) / NC;
    auto kb = (k+KC-1) / KC;

    auto _mc = m % MC;
    auto _nc = n % NC;
    auto _kc = k % KC;

    for (std::size_t j=0; j<nb; ++j) {
        auto nc = (j!=nb-1 || _nc==0) ? NC : _nc;

        for (std::size_t l=0; l<kb; ++l) {
            auto kc = (l!=kb-1 || _kc==0) ? KC   : _kc;
            auto _beta = (l==0) ? beta : 1.0;

            pack_B(kc, nc, &B[l*KC*b_row_stride+j*NC*b_col_stride], b_row_stride, b_col_stride, _B);

            for (std::size_t i=0; i<mb; ++i) {
                auto mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc, &A[i*MC*a_row_stride+l*KC*a_col_stride], a_row_stride, a_col_stride, _A);

                dgemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i*MC*c_row_stride+j*NC*c_col_stride], c_row_stride, c_col_stride);
            }
        }
    }
}

} //end of namespace dgemm_detail

template<typename A, typename B, typename C>
void fast_dgemm(A&& a, B&& b, C&& c){
    dgemm_detail::dgemm_nn(
        etl::rows(a), etl::columns(b), etl::columns(a),
        1.0,
        a.memory_start(), etl::dim<1>(a), 1,
        b.memory_start(), etl::dim<1>(b), 1,
        0.0,
        c.memory_start(), etl::dim<1>(c), 1
    );
};

} //end of namespace etl

#endif
