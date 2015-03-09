//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

//Highly inspired from the "GEMM: From Pure C to SSE Optimized Micro Kernels" course

#ifndef ETL_DGEMM_HPP
#define ETL_DGEMM_HPP

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
void pack_MRxk(std::size_t k, const D* A, std::size_t incRowA, std::size_t incColA, D* buffer){
    for (std::size_t j=0; j<k; ++j) {
        for (std::size_t i=0; i<MR; ++i) {
            buffer[i] = A[i*incRowA];
        }

        buffer += MR;
        A += incColA;
    }
}

template<typename D>
void pack_A(std::size_t mc, std::size_t kc, const D* A, std::size_t incRowA, std::size_t incColA, D* buffer){
    auto mp  = mc / MR;
    auto _mr = mc % MR;

    for (std::size_t i=0; i<mp; ++i) {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc*MR;
        A += MR*incRowA;
    }

    if (_mr>0) {
        for (std::size_t j=0; j<kc; ++j) {
            for (std::size_t i=0; i<_mr; ++i) {
                buffer[i] = A[i*incRowA];
            }
            for (std::size_t i=_mr; i<MR; ++i) {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A += incColA;
        }
    }
}

template<typename D>
void pack_kxNR(std::size_t k, const D* B, std::size_t incRowB, std::size_t incColB, D* buffer){
    for (std::size_t i=0; i<k; ++i) {
        for (std::size_t j=0; j<NR; ++j) {
            buffer[j] = B[j*incColB];
        }
        buffer += NR;
        B += incRowB;
    }
}

template<typename D>
void pack_B(std::size_t kc, std::size_t nc, const D* B, std::size_t incRowB, std::size_t incColB, D* buffer){
    auto np  = nc / NR;
    auto _nr = nc % NR;

    for (std::size_t j=0; j<np; ++j) {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc*NR;
        B += NR*incColB;
    }

    if (_nr>0) {
        for (std::size_t i=0; i<kc; ++i) {
            for (std::size_t j=0; j<_nr; ++j) {
                buffer[j] = B[j*incColB];
            }
            for (std::size_t j=_nr; j<NR; ++j) {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B += incRowB;
        }
    }
}

template<typename D>
void dgemm_micro_kernel(std::size_t kc, D alpha, const D* A, const D* B, D beta, D* C, std::size_t incRowC, std::size_t incColC){
    D AB[MR*NR];

    for (std::size_t l=0; l<MR*NR; ++l) {
        AB[l] = 0;
    }

    for (std::size_t l=0; l<kc; ++l) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                AB[i+j*MR] += A[i]*B[j];
            }
        }
        A += MR;
        B += NR;
    }

    if (beta==0.0) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = 0.0;
            }
        }
    } else if (beta!=1.0) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
            }
        }
    }

    if (alpha==1.0) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += AB[i+j*MR];
            }
        }
    } else {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] += alpha*AB[i+j*MR];
            }
        }
    }
}

template<typename D>
void dgeaxpy(std::size_t m, std::size_t n, D alpha, const D  *X, std::size_t incRowX, std::size_t incColX, D* Y, std::size_t incRowY, std::size_t incColY){
    if (alpha!=1.0) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += X[i*incRowX+j*incColX];
            }
        }
    }
}

template<typename D>
void dgescal(std::size_t m, std::size_t n, D alpha, D* X, std::size_t incRowX, std::size_t incColX){
    if (alpha!=0.0) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] *= alpha;
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                X[i*incRowX+j*incColX] = 0.0;
            }
        }
    }
}

template<typename D>
void dgemm_macro_kernel(std::size_t mc, std::size_t nc, std::size_t kc, D alpha, D beta, D* C, std::size_t incRowC, std::size_t incColC){
    auto mp = (mc+MR-1) / MR;
    auto np = (nc+NR-1) / NR;

    auto _mr = mc % MR;
    auto _nr = nc % NR;

    for (std::size_t j=0; j<np; ++j) {
        auto nr = (j!=np-1 || _nr==0) ? NR : _nr;

        for (std::size_t i=0; i<mp; ++i) {
            auto mr = (i!=mp-1 || _mr==0) ? MR : _mr;

            if (mr==MR && nr==NR) {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            } else {
                dgemm_micro_kernel(kc, alpha, &_A[i*kc*MR], &_B[j*kc*NR], 0.0, _C, 1, MR);
                dgescal(mr, nr, beta, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                dgeaxpy(mr, nr, 1.0, _C, 1, MR, &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

template<typename D>
void dgemm_nn(std::size_t m, std::size_t n, std::size_t k, D alpha, const D* A, std::size_t incRowA, std::size_t incColA, const D* B, std::size_t incRowB, std::size_t incColB, D beta, D* C, std::size_t incRowC, std::size_t incColC){
    if (alpha==0.0 || k==0) {
        dgescal(m, n, beta, C, incRowC, incColC);
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

            pack_B(kc, nc, &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB, _B);

            for (std::size_t i=0; i<mb; ++i) {
                auto mc = (i!=mb-1 || _mc==0) ? MC : _mc;

                pack_A(mc, kc, &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA, _A);

                dgemm_macro_kernel(mc, nc, kc, alpha, _beta, &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC);
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
