//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "../../config.hpp"

#ifdef ETL_BLAS_MODE

extern "C"
{
#include "cblas.h"
}

#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_dgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        0.0,
        c.memory_start(), major_stride(c)
    );
}

template<typename A, typename B, typename C>
void sgemm(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_sgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        1.0f,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        0.0f,
        c.memory_start(), major_stride(c)
    );
}

template<typename A, typename B, typename C>
void dgemv(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_dgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        0.0,
        c.memory_start(), 1
    );
}

template<typename A, typename B, typename C>
void sgemv(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_sgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        0.0,
        c.memory_start(), 1
    );
}

template<typename A, typename B, typename C>
void dgevm(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    cblas_dgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        1.0,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        0.0,
        c.memory_start(), 1
    );
}

template<typename A, typename B, typename C>
void sgevm(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    cblas_sgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        1.0,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        0.0,
        c.memory_start(), 1
    );
}

#else

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void sgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void sgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgevm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void sgevm(A&& a, B&& b, C&& c);

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl
