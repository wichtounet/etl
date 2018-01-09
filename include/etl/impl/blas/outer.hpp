//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of the outer product
 */

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Compute the outer product of a and b and store the result in c
 * \param a The lhs expression
 * \param b The rhs expression
 * \param c The output expression
 */
template <typename A, typename B, typename C, cpp_enable_iff(all_single_precision<A, B, C>)>
void outer(const A& a, const B& b, C&& c) {
    c = 0;

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();
    c.ensure_cpu_up_to_date();

    cblas_sger(
        CblasRowMajor,
        etl::dim<0>(a), etl::dim<0>(b),
        1.0,
        a.memory_start(), 1,
        b.memory_start(), 1,
        c.memory_start(), etl::dim<0>(b));

    c.invalidate_gpu();
}

/*!
 * \copydoc outer
 */
template <typename A, typename B, typename C, cpp_enable_iff(all_double_precision<A, B, C>)>
void outer(const A& a, const B& b, C&& c) {
    c = 0;

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();
    c.ensure_cpu_up_to_date();

    cblas_dger(
        CblasRowMajor,
        etl::dim<0>(a), etl::dim<0>(b),
        1.0,
        a.memory_start(), 1,
        b.memory_start(), 1,
        c.memory_start(), etl::dim<0>(b));

    c.invalidate_gpu();
}

/*!
 * \brief Compute the batch_outer product of a and b and store the result in c
 * \param a The lhs expression
 * \param b The rhs expression
 * \param c The output expression
 */
template <typename A, typename B, typename C, cpp_enable_iff(all_single_precision<A, B, C>)>
void batch_outer(const A& a, const B& b, C&& c) {
    const size_t m = etl::rows(c);
    const size_t n = etl::columns(c);
    const size_t k = etl::rows(a);

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    cblas_sgemm(
        CblasRowMajor,
        CblasTrans, CblasNoTrans,
        m, n, k,
        1.0f,
        a.memory_start(), m,
        b.memory_start(), n,
        0.0f,
        c.memory_start(), n);

    c.invalidate_gpu();
}

/*!
 * \copydoc batch_outer
 */
template <typename A, typename B, typename C, cpp_enable_iff(all_double_precision<A, B, C>)>
void batch_outer(const A& a, const B& b, C&& c) {
    const size_t m = etl::rows(c);
    const size_t n = etl::columns(c);
    const size_t k = etl::rows(a);

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    cblas_dgemm(
        CblasRowMajor,
        CblasTrans, CblasNoTrans,
        m, n, k,
        1.0,
        a.memory_start(), m,
        b.memory_start(), n,
        0.0,
        c.memory_start(), n);

    c.invalidate_gpu();
}

#else

/*!
 * \copydoc outer
 */
template <typename A, typename B, typename C>
void outer(const A& /*a*/, const B& /*b*/, C&& /*c*/) {
    cpp_unreachable("BLAS not enabled/available");
}

/*!
 * \copydoc batch_outer
 */
template <typename A, typename B, typename C>
void batch_outer(const A& /*a*/, const B& /*b*/, C&& /*c*/) {
    cpp_unreachable("BLAS not enabled/available");
}

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
