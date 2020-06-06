//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Standard implementation of a matrix-matrix multiplication
 * \param a The left input matrix
 * \param b The right input matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C, typename T>
static void mm_mul(A&& a, B&& b, C&& c, T alpha) {
    static constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    c = 0;

    if constexpr (row_major) {
        for (size_t i = 0; i < rows(a); i++) {
            for (size_t k = 0; k < columns(a); k++) {
                for (size_t j = 0; j < columns(b); j++) {
                    c(i, j) += alpha * a(i, k) * b(k, j);
                }
            }
        }
    } else {
        for (size_t j = 0; j < columns(b); j++) {
            for (size_t k = 0; k < columns(a); k++) {
                for (size_t i = 0; i < rows(a); i++) {
                    c(i, j) += alpha * a(i, k) * b(k, j);
                }
            }
        }
    }
}

/*!
 * \brief Performs the computation c = c + a * b
 * \param c The output
 * \param a The lhs of the multiplication
 * \param b The rhs of the multiplication
 */
inline void add_mul(float& c, float a, float b) {
    c += a * b;
}

/*!
 * \brief Performs the computation c = c + a * b
 * \param c The output
 * \param a The lhs of the multiplication
 * \param b The rhs of the multiplication
 */
inline void add_mul(double& c, double a, double b) {
    c += a * b;
}

/*!
 * \brief Performs the computation c = c + a * b
 * \param c The output
 * \param a The lhs of the multiplication
 * \param b The rhs of the multiplication
 */
template <typename T>
inline void add_mul(etl::complex<T>& c, etl::complex<T> a, etl::complex<T> b) {
    c += a * b;
}

/*!
 * \brief Performs the computation c = c + a * b
 * \param c The output
 * \param a The lhs of the multiplication
 * \param b The rhs of the multiplication
 *
 * Note: For some reason, compilers have a real hard time
 * inlining/vectorizing std::complex operations
 * This helper improves performance by more than 50% on some cases
 */
template <typename T>
inline void add_mul(std::complex<T>& c, std::complex<T> a, std::complex<T> b) {
    T ac = a.real() * b.real();
    T bd = a.imag() * b.imag();

    T abcd = (a.real() + a.imag()) * (b.real() + b.imag());

    c.real(c.real() + ac - bd);
    c.imag(c.imag() + abcd - ac - bd);
}

/*!
 * \brief Standard implementation of a vector-matrix multiplication
 * \param a The left vector matrix
 * \param b The right input matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
static void vm_mul(A&& a, B&& b, C&& c) {
    static constexpr bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    c = 0;

    if constexpr (row_major) {
        for (size_t k = 0; k < etl::dim<0>(a); k++) {
            for (size_t j = 0; j < columns(b); j++) {
                //optimized compound add of the multiplication
                add_mul(c(j), a(k), b(k, j));
            }
        }
    } else {
        for (size_t j = 0; j < columns(b); j++) {
            for (size_t k = 0; k < etl::dim<0>(a); k++) {
                //optimized compound add of the multiplication
                add_mul(c(j), a(k), b(k, j));
            }
        }
    }
}

/*!
 * \brief Standard implementation of a matrix-vector multiplication
 * \param a The left input matrix
 * \param b The right vector matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
static void mv_mul(A&& a, B&& b, C&& c) {
    static constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    c = 0;

    if constexpr (row_major) {
        for (size_t i = 0; i < rows(a); i++) {
            for (size_t k = 0; k < columns(a); k++) {
                //optimized compound add of the multiplication
                add_mul(c(i), a(i, k), b(k));
            }
        }
    } else {
        for (size_t k = 0; k < columns(a); k++) {
            for (size_t i = 0; i < rows(a); i++) {
                //optimized compound add of the multiplication
                add_mul(c(i), a(i, k), b(k));
            }
        }
    }
}

} //end of namespace etl::impl::standard
