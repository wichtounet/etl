//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Strassen multiplication step
 */
template <typename A, typename B, typename C>
void strassen_mm_mul_r(const A& a, const B& b, C& c) {
    using value_type = value_t<A>;

    size_t n = dim<0>(a);

    //1x1 matrix mul
    if (n == 1) {
        c(0, 0) = a(0, 0) * b(0, 0);
    } else if (n == 2) {
        value_type a11 = a(0, 0);
        value_type a12 = a(0, 1);
        value_type a21 = a(1, 0);
        value_type a22 = a(1, 1);

        value_type b11 = b(0, 0);
        value_type b12 = b(0, 1);
        value_type b21 = b(1, 0);
        value_type b22 = b(1, 1);

        value_type p1 = (a11 + a22) * (b11 + b22);
        value_type p2 = (a12 - a22) * (b21 + b22);
        value_type p3 = a11 * (b12 - b22);
        value_type p4 = a22 * (b21 - b11);
        value_type p5 = (a11 + a12) * b22;
        value_type p6 = (a21 + a22) * b11;
        value_type p7 = (a21 - a11) * (b11 + b12);

        c(0, 0) = p1 + p4 + p2 - p5;
        c(0, 1) = p3 + p5;
        c(1, 0) = p6 + p4;
        c(1, 1) = p1 + p3 + p7 - p6;
    } else if (n == 4) {
        //This is entirely done on stack

        size_t new_n = n / 2;

        etl::fast_matrix<value_type, 2, 2> a11;
        etl::fast_matrix<value_type, 2, 2> a12;
        etl::fast_matrix<value_type, 2, 2> a21;
        etl::fast_matrix<value_type, 2, 2> a22;

        etl::fast_matrix<value_type, 2, 2> b11;
        etl::fast_matrix<value_type, 2, 2> b12;
        etl::fast_matrix<value_type, 2, 2> b21;
        etl::fast_matrix<value_type, 2, 2> b22;

        etl::fast_matrix<value_type, 2, 2> p1;
        etl::fast_matrix<value_type, 2, 2> p2;
        etl::fast_matrix<value_type, 2, 2> p3;
        etl::fast_matrix<value_type, 2, 2> p4;
        etl::fast_matrix<value_type, 2, 2> p5;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                a11(i, j) = a(i, j);
                a12(i, j) = a(i, j + new_n);
                a21(i, j) = a(i + new_n, j);
                a22(i, j) = a(i + new_n, j + new_n);

                b11(i, j) = b(i, j);
                b12(i, j) = b(i, j + new_n);
                b21(i, j) = b(i + new_n, j);
                b22(i, j) = b(i + new_n, j + new_n);
            }
        }

        strassen_mm_mul_r(a11 + a22, b11 + b22, p1);
        strassen_mm_mul_r(a12 - a22, b21 + b22, p2);
        strassen_mm_mul_r(a22, b21 - b11, p4);
        strassen_mm_mul_r(a11 + a12, b22, p5);

        auto c11 = p1 + p4 + p2 - p5;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i, j) = c11(i, j);
            }
        }

        strassen_mm_mul_r(a11, b12 - b22, p3);

        auto c12 = p3 + p5;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i, j + new_n) = c12(i, j);
            }
        }

        strassen_mm_mul_r(a21 + a22, b11, p2);
        strassen_mm_mul_r(a21 - a11, b11 + b12, p5);

        auto c21 = p2 + p4;
        auto c22 = p1 + p3 + p5 - p2;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i + new_n, j)         = c21(i, j);
                c(i + new_n, j + new_n) = c22(i, j);
            }
        }
    } else {
        size_t new_n = n / 2;

        etl::dyn_matrix<value_type> a11(new_n, new_n);
        etl::dyn_matrix<value_type> a12(new_n, new_n);
        etl::dyn_matrix<value_type> a21(new_n, new_n);
        etl::dyn_matrix<value_type> a22(new_n, new_n);

        etl::dyn_matrix<value_type> b11(new_n, new_n);
        etl::dyn_matrix<value_type> b12(new_n, new_n);
        etl::dyn_matrix<value_type> b21(new_n, new_n);
        etl::dyn_matrix<value_type> b22(new_n, new_n);

        etl::dyn_matrix<value_type> p1(new_n, new_n);
        etl::dyn_matrix<value_type> p2(new_n, new_n);
        etl::dyn_matrix<value_type> p3(new_n, new_n);
        etl::dyn_matrix<value_type> p4(new_n, new_n);
        etl::dyn_matrix<value_type> p5(new_n, new_n);

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                a11(i, j) = a(i, j);
                a12(i, j) = a(i, j + new_n);
                a21(i, j) = a(i + new_n, j);
                a22(i, j) = a(i + new_n, j + new_n);

                b11(i, j) = b(i, j);
                b12(i, j) = b(i, j + new_n);
                b21(i, j) = b(i + new_n, j);
                b22(i, j) = b(i + new_n, j + new_n);
            }
        }

        strassen_mm_mul_r(a11 + a22, b11 + b22, p1);
        strassen_mm_mul_r(a12 - a22, b21 + b22, p2);
        strassen_mm_mul_r(a22, b21 - b11, p4);
        strassen_mm_mul_r(a11 + a12, b22, p5);

        auto c11 = p1 + p4 + p2 - p5;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i, j) = c11(i, j);
            }
        }

        strassen_mm_mul_r(a11, b12 - b22, p3);

        auto c12 = p3 + p5;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i, j + new_n) = c12(i, j);
            }
        }

        strassen_mm_mul_r(a21 + a22, b11, p2);
        strassen_mm_mul_r(a21 - a11, b11 + b12, p5);

        auto c21 = p2 + p4;
        auto c22 = p1 + p3 + p5 - p2;

        for (size_t i = 0; i < new_n; i++) {
            for (size_t j = 0; j < new_n; j++) {
                c(i + new_n, j)         = c21(i, j);
                c(i + new_n, j + new_n) = c22(i, j);
            }
        }
    }
}

/*!
 * \brief Returns the next power of two of n
 */
inline size_t next_power_of_two(size_t n) {
    return std::pow(2, static_cast<size_t>(std::ceil(std::log2(n))));
}

/*!
 * \brief Strassen multiplication of a and b into c
 * \param a The left hand side of the multiplication
 * \param b The right hand side of the multiplication
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void strassen_mm_mul(const A& a, const B& b, C& c) {
    c = 0;

    //For now, assume matrices are of size 2^nx2^n

    size_t n = std::max(dim<0>(a), std::max(dim<1>(a), dim<1>(b)));
    size_t m = next_power_of_two(n);

    if (dim<0>(a) == m && dim<0>(b) == m && dim<1>(a) == m && dim<1>(b) == m) {
        strassen_mm_mul_r(a, b, c);
    } else {
        using value_type = value_t<A>;

        etl::dyn_matrix<value_type> a_prep(m, m, value_type(0));
        etl::dyn_matrix<value_type> b_prep(m, m, value_type(0));
        etl::dyn_matrix<value_type> c_prep(m, m, value_type(0));

        for (size_t i = 0; i < dim<0>(a); i++) {
            for (size_t j = 0; j < dim<1>(a); j++) {
                a_prep(i, j) = a(i, j);
            }
        }

        for (size_t i = 0; i < dim<0>(b); i++) {
            for (size_t j = 0; j < dim<1>(b); j++) {
                b_prep(i, j) = b(i, j);
            }
        }

        strassen_mm_mul_r(a_prep, b_prep, c_prep);

        for (size_t i = 0; i < dim<0>(c); i++) {
            for (size_t j = 0; j < dim<1>(c); j++) {
                c(i, j) = c_prep(i, j);
            }
        }
    }
}

} //end of namespace etl::impl::standard
