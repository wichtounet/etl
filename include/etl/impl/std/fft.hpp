//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace standard {

namespace detail {

//The efficient sub transforms modules have been taken from the "FFT Algorithms"
//paper by Brian Gough, 1997

/*!
 * \brief Limit on the number of factors of the prime factorization
 * of the FFT size
 */
constexpr std::size_t MAX_FACTORS = 32;

/*!
 * \brief Transform module for a FFT with 2 points
 * \param in The input vector
 * \param out The output vector
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle The first twiddle factors
 */
template <typename T>
void fft_2_point(const etl::complex<T>* in, etl::complex<T>* out, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle) {
    static constexpr std::size_t factor = 2;

    const std::size_t m      = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc    = (factor - 1) * offset;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc) {
        etl::complex<T> w(1.0, 0.0);

        if (k > 0) {
            w = twiddle[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j) {
            etl::complex<T> z0 = in[i];
            etl::complex<T> z1 = in[i + m];

            out[j]          = z0 + z1;
            out[j + offset] = w * (z0 - z1);
        }
    }
}

/*!
 * \brief Transform module for a FFT with 3 points
 * \param in The input vector
 * \param out The output vector
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle1 The first twiddle factors
 * \param twiddle2 The second twiddle factors
 */
template <typename T>
void fft_3_point(const etl::complex<T>* in, etl::complex<T>* out, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle1, const etl::complex<T>* twiddle2) {
    static constexpr std::size_t factor = 3;

    const std::size_t m      = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc    = (factor - 1) * offset;

    static const T tau = std::sqrt(3.0) / 2.0;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; k++, j += inc) {
        etl::complex<T> w1(1.0, 0.0);
        etl::complex<T> w2(1.0, 0.0);

        if (k > 0) {
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j) {
            etl::complex<T> z0 = in[i];
            etl::complex<T> z1 = in[i + m];
            etl::complex<T> z2 = in[i + 2 * m];

            etl::complex<T> t1 = z1 + z2;
            etl::complex<T> t2 = z0 - t1 / T(2.0);
            etl::complex<T> t3 = -tau * (z1 - z2);

            out[j]              = z0 + t1;
            out[j + offset]     = w1 * (t2 + inverse_conj(t3));
            out[j + 2 * offset] = w2 * (t2 + conj_inverse(t3));
        }
    }
}

/*!
 * \brief Transform module for a FFT with 4 points
 * \param in The input vector
 * \param out The output vector
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle1 The first twiddle factors
 * \param twiddle2 The second twiddle factors
 * \param twiddle3 The third twiddle factors
 */
template <typename T>
void fft_4_point(const etl::complex<T>* in, etl::complex<T>* out, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle1, const etl::complex<T>* twiddle2, const etl::complex<T>* twiddle3) {
    static constexpr std::size_t factor = 4;

    const std::size_t m      = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc    = (factor - 1) * offset;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; k++, j += inc) {
        etl::complex<T> w1(1.0, 0.0);
        etl::complex<T> w2(1.0, 0.0);
        etl::complex<T> w3(1.0, 0.0);

        if (k > 0) {
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
            w3 = twiddle3[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j) {
            etl::complex<T> z0 = in[i];
            etl::complex<T> z1 = in[i + 1 * m];
            etl::complex<T> z2 = in[i + 2 * m];
            etl::complex<T> z3 = in[i + 3 * m];

            etl::complex<T> t1 = z0 + z2;
            etl::complex<T> t2 = z1 + z3;
            etl::complex<T> t3 = z0 - z2;
            etl::complex<T> t4 = T(-1.0) * (z1 - z3);

            out[j]              = t1 + t2;
            out[j + 1 * offset] = w1 * (t3 + inverse_conj(t4));
            out[j + 2 * offset] = w2 * (t1 - t2);
            out[j + 3 * offset] = w3 * (t3 + conj_inverse(t4));
        }
    }
}

/*!
 * \brief Transform module for a FFT with 5 points
 * \param in The input vector
 * \param out The output vector
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle1 The first twiddle factors
 * \param twiddle2 The second twiddle factors
 * \param twiddle3 The third twiddle factors
 * \param twiddle4 The fourth twiddle factors
 */
template <typename T>
void fft_5_point(const etl::complex<T>* in, etl::complex<T>* out, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle1, const etl::complex<T>* twiddle2, const etl::complex<T>* twiddle3, const etl::complex<T>* twiddle4) {
    static constexpr std::size_t factor = 5;

    const std::size_t m      = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc    = (factor - 1) * offset;

    static const T theta_1 = -1.0 * std::sin(2.0 * M_PI / 5.0);
    static const T theta_2 = -1.0 * std::sin(2.0 * M_PI / 10.0);

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc) {
        etl::complex<T> w1(1.0, 0.0);
        etl::complex<T> w2(1.0, 0.0);
        etl::complex<T> w3(1.0, 0.0);
        etl::complex<T> w4(1.0, 0.0);

        if (k > 0) {
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
            w3 = twiddle3[k - 1];
            w4 = twiddle4[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j) {
            etl::complex<T> z0 = in[i];
            etl::complex<T> z1 = in[i + m];
            etl::complex<T> z2 = in[i + 2 * m];
            etl::complex<T> z3 = in[i + 3 * m];
            etl::complex<T> z4 = in[i + 4 * m];

            etl::complex<T> t1  = z1 + z4;
            etl::complex<T> t2  = z2 + z3;
            etl::complex<T> t3  = z1 - z4;
            etl::complex<T> t4  = z2 - z3;
            etl::complex<T> t5  = t1 + t2;
            etl::complex<T> t6  = T(std::sqrt(5.0) / 4.0) * (t1 - t2);
            etl::complex<T> t7  = z0 - (t5 / T(4));
            etl::complex<T> t8  = t7 + t6;
            etl::complex<T> t9  = t7 - t6;
            etl::complex<T> t10 = theta_1 * t3 + theta_2 * t4;
            etl::complex<T> t11 = theta_2 * t3 - theta_1 * t4;

            out[j]              = z0 + t5;
            out[j + offset]     = w1 * (t8 + inverse_conj(t10));
            out[j + 2 * offset] = w2 * (t9 + inverse_conj(t11));
            out[j + 3 * offset] = w3 * (t9 - inverse_conj(t11));
            out[j + 4 * offset] = w4 * (t8 - inverse_conj(t10));
        }
    }
}

/*!
 * \brief Transform module for a FFT with 7 points
 * \param in The input vector
 * \param out The output vector
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle1 The first twiddle factors
 * \param twiddle2 The second twiddle factors
 * \param twiddle3 The third twiddle factors
 * \param twiddle4 The fourth twiddle factors
 * \param twiddle5 The fifth twiddle factors
 * \param twiddle6 The sixth twiddle factors
 */
template <typename T>
void fft_7_point(const etl::complex<T>* in, etl::complex<T>* out, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle1, const etl::complex<T>* twiddle2, const etl::complex<T>* twiddle3, const etl::complex<T>* twiddle4, const etl::complex<T>* twiddle5, const etl::complex<T>* twiddle6) {
    static constexpr std::size_t factor = 7;

    const std::size_t m      = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc    = (factor - 1) * offset;

    static constexpr T theta_0 = 2.0 * M_PI / 7.0;

    static const T theta_1 = (std::cos(theta_0) + std::cos(2.0 * theta_0) + std::cos(3.0 * theta_0)) / 3.0 - 1.0;
    static const T theta_2 = (2.0 * std::cos(theta_0) - std::cos(2.0 * theta_0) - std::cos(3.0 * theta_0)) / 3.0;
    static const T theta_3 = (std::cos(theta_0) - 2.0 * std::cos(2.0 * theta_0) + std::cos(3.0 * theta_0)) / 3.0;
    static const T theta_4 = (std::cos(theta_0) + std::cos(2.0 * theta_0) - 2.0 * std::cos(3.0 * theta_0)) / 3.0;

    static const T theta_5 = (std::sin(theta_0) + std::sin(2.0 * theta_0) - std::sin(3.0 * theta_0)) / 3.0;
    static const T theta_6 = (2.0 * std::sin(theta_0) - std::sin(2.0 * theta_0) + std::sin(3.0 * theta_0)) / 3.0;
    static const T theta_7 = (std::sin(theta_0) - 2.0 * std::sin(2.0 * theta_0) - std::sin(3.0 * theta_0)) / 3.0;
    static const T theta_8 = (std::sin(theta_0) + std::sin(2.0 * theta_0) + 2.0 * std::sin(3.0 * theta_0)) / 3.0;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc) {
        etl::complex<T> w1(1.0, 0.0);
        etl::complex<T> w2(1.0, 0.0);
        etl::complex<T> w3(1.0, 0.0);
        etl::complex<T> w4(1.0, 0.0);
        etl::complex<T> w5(1.0, 0.0);
        etl::complex<T> w6(1.0, 0.0);

        if (k > 0) {
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
            w3 = twiddle3[k - 1];
            w4 = twiddle4[k - 1];
            w5 = twiddle5[k - 1];
            w6 = twiddle6[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; k1++, ++i, ++j) {
            etl::complex<T> z0 = in[i];
            etl::complex<T> z1 = in[i + m];
            etl::complex<T> z2 = in[i + 2 * m];
            etl::complex<T> z3 = in[i + 3 * m];
            etl::complex<T> z4 = in[i + 4 * m];
            etl::complex<T> z5 = in[i + 5 * m];
            etl::complex<T> z6 = in[i + 6 * m];

            etl::complex<T> t0  = z1 + z6;
            etl::complex<T> t1  = z1 - z6;
            etl::complex<T> t2  = z2 + z5;
            etl::complex<T> t3  = z2 - z5;
            etl::complex<T> t4  = z4 + z3;
            etl::complex<T> t5  = z4 - z3;
            etl::complex<T> t6  = t2 + t0;
            etl::complex<T> t8  = z0 + t6 + t4;
            etl::complex<T> t10 = theta_2 * (t0 - t4);
            etl::complex<T> t11 = theta_3 * (t4 - t2);
            etl::complex<T> t12 = theta_4 * (t2 - t0);
            etl::complex<T> t13 = theta_5 * (t5 + t3 + t1);
            etl::complex<T> t14 = theta_6 * (t1 - t5);
            etl::complex<T> t15 = theta_7 * (t5 - t3);
            etl::complex<T> t16 = theta_8 * (t3 - t1);
            etl::complex<T> t17 = t8 + theta_1 * (t6 + t4);
            etl::complex<T> t18 = t17 + t10 + t11;
            etl::complex<T> t19 = t17 + t12 - t11;
            etl::complex<T> t20 = t17 - t10 - t12;
            etl::complex<T> t21 = t14 + t15 + t13;
            etl::complex<T> t22 = t16 - t15 + t13;
            etl::complex<T> t23 = -t16 - t14 + t13;

            out[j]              = t8;
            out[j + 1 * offset] = w1 * (t18 + conj_inverse(t21));
            out[j + 2 * offset] = w2 * (t20 + conj_inverse(t23));
            out[j + 3 * offset] = w3 * (t19 + inverse_conj(t22));
            out[j + 4 * offset] = w4 * (t19 + conj_inverse(t22));
            out[j + 5 * offset] = w5 * (t20 + inverse_conj(t23));
            out[j + 6 * offset] = w6 * (t18 + inverse_conj(t21));
        }
    }
}

/*!
 * \brief General Transform module for a FFT
 * \param in The input vector
 * \param out The output vector
 * \param factor The factor
 * \param product The current product
 * \param n The size of the transform
 * \param twiddle The twiddle factors
 */
template <typename T>
void fft_n_point(etl::complex<T>* in, etl::complex<T>* out, const std::size_t factor, const std::size_t product, const std::size_t n, const etl::complex<T>* twiddle) {
    const std::size_t m            = n / factor;
    const std::size_t q            = n / product;
    const std::size_t offset       = product / factor;
    const std::size_t inc          = (factor - 1) * offset;
    const std::size_t factor_limit = (factor - 1) / 2 + 1;

    std::copy_n(in, m, out);

    for (std::size_t i = 1; i < (factor - 1) / 2 + 1; i++) {
        std::transform(in + i * m, in + i * m + m, in + (factor - i) * m, out + i * m, std::plus<etl::complex<T>>());
        std::transform(in + i * m, in + i * m + m, in + (factor - i) * m, out + (factor - i) * m, std::minus<etl::complex<T>>());
    }

    std::copy_n(out, m, in);

    for (std::size_t i = 1; i < factor_limit; i++) {
        std::transform(in, in + m, out + i * m, in, std::plus<etl::complex<T>>());
    }

    for (std::size_t e = 1; e < factor_limit; e++) {
        std::copy_n(out, m, in + e * m);
        std::copy_n(out, m, in + (factor - e) * m);

        for (std::size_t k = 1, j = e * q; k < (factor - 1) / 2 + 1; k++) {
            etl::complex<T> w(1.0, 0.0);

            if (j > 0) {
                w = twiddle[j - 1];
            }

            for (std::size_t i = 0; i < m; i++) {
                etl::complex<T> xp = out[i + k * m];
                etl::complex<T> xm = out[i + (factor - k) * m];

                in[i + e * m] += w.real * xp - w.imag * conj_inverse(xm);
                in[i + (factor - e) * m] += w.real * xp + w.imag * conj_inverse(xm);
            }

            j = (j + (e * q)) % (factor * q);
        }
    }

    std::copy_n(in, offset, out);

    for (std::size_t i = 1; i < factor; i++) {
        std::copy_n(in + i * m, offset, out + i * offset);
    }

    for (std::size_t i = offset, j = product; i < offset + (q - 1) * offset; i += offset, j += offset + inc) {
        std::copy_n(in + i, offset, out + j);
    }

    for (std::size_t k = 1, i = offset, j = product; k < q; ++k, j += inc) {
        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j) {
            for (std::size_t e = 1; e < factor; e++) {
                //out = w * x
                out[j + e * offset] = twiddle[(e - 1) * q + k - 1] * in[i + e * m];
            }
        }
    }
}

/*!
 * \brief Factorize the FFT size into factors
 * \param n The size of the transform
 * \param factors The output factors
 * \param n_factors The number of factors
 */
inline void fft_factorize(std::size_t n, std::size_t* factors, std::size_t& n_factors) {
    //0. Favour the factors with implemented transform modules

    while (n > 1) {
        if (n % 7 == 0) {
            n /= 7;
            factors[n_factors++] = 7;
        } else if (n % 5 == 0) {
            n /= 5;
            factors[n_factors++] = 5;
        } else if (n % 4 == 0) {
            n /= 4;
            factors[n_factors++] = 4;
        } else if (n % 3 == 0) {
            n /= 3;
            factors[n_factors++] = 3;
        } else if (n % 2 == 0) {
            n /= 2;
            factors[n_factors++] = 2;
        } else {
            //At this point, there are no transform module
            break;
        }
    }

    //1. Search for prime factors

    std::size_t prime_factor = 11;

    while (n > 1) {
        //Search for the next prime factor
        while (n % prime_factor != 0) {
            prime_factor += 2;
        }

        n /= prime_factor;
        factors[n_factors++] = prime_factor;
    }
}

/*!
 * \brief Compute the twiddle factors
 * \param n The size of the transform
 * \param factors The factors
 * \param n_factors The number of factors
 * \param twiddle The output twiddle factors (pointers inside the main twiddle factors array)
 * \return an array containing all the twiddle factors
 */
template <typename T>
std::unique_ptr<etl::complex<T>[]> twiddle_compute(const std::size_t n, std::size_t* factors, std::size_t n_factors, etl::complex<T>** twiddle) {
    std::unique_ptr<etl::complex<T>[]> trig = etl::allocate<etl::complex<T>>(n);

    const T d_theta = -2.0 * M_PI / (static_cast<T>(n));

    std::size_t t       = 0;
    std::size_t product = 1;

    for (std::size_t i = 0; i < n_factors; i++) {
        std::size_t factor = factors[i];
        twiddle[i]         = &trig[0] + t;

        std::size_t prev_product = product;
        product *= factor;

        for (std::size_t j = 1; j < factor; j++) {
            std::size_t m = 0;
            for (std::size_t k = 1; k <= n / product; k++) {
                m = (m + j * prev_product) % n;

                T theta = d_theta * m;

                trig[t] = etl::complex<T>{std::cos(theta), std::sin(theta)};

                t++;
            }
        }
    }

    return trig;
}

/*!
 * \brief Perform the FFT
 * \param r_in The input
 * \param r_out The output
 * \param n The size of the transform
 * \param factors The factors
 * \param n_factors The number of factors
 * \param twiddle The output twiddle factors (pointers inside the main twiddle factors array)
 */
template <typename In, typename T>
void fft_perform(const In* r_in, etl::complex<T>* r_out, const std::size_t n, std::size_t* factors, std::size_t n_factors, etl::complex<T>** twiddle) {
    auto tmp = etl::allocate<etl::complex<T>>(n);

    std::copy_n(r_in, n, tmp.get());

    auto* in  = tmp.get();
    auto* out = r_out;

    std::size_t product = 1;

    for (std::size_t i = 0; i < n_factors; i++) {
        std::size_t factor = factors[i];

        if (i > 0) {
            std::swap(in, out);
        }

        product *= factor;

        std::size_t offset = n / product;

        if (factor == 2) {
            fft_2_point(in, out, product, n, twiddle[i]);
        } else if (factor == 3) {
            fft_3_point(in, out, product, n, twiddle[i], twiddle[i] + offset);
        } else if (factor == 4) {
            fft_4_point(in, out, product, n, twiddle[i], twiddle[i] + offset, twiddle[i] + 2 * offset);
        } else if (factor == 5) {
            fft_5_point(in, out, product, n, twiddle[i], twiddle[i] + offset, twiddle[i] + 2 * offset, twiddle[i] + 3 * offset);
        } else if (factor == 7) {
            fft_7_point(in, out, product, n, twiddle[i], twiddle[i] + offset, twiddle[i] + 2 * offset, twiddle[i] + 3 * offset, twiddle[i] + 4 * offset, twiddle[i] + 5 * offset);
        } else {
            fft_n_point(in, out, factor, product, n, twiddle[i]);
        }
    }

    if (out != r_out) {
        std::copy_n(out, n, r_out);
    }
}

/*!
 * \brief Compute the general FFT of r_in
 * \param r_in The input signal
 * \param r_out The output signal
 * \param n The size of the tranform
 */
template <typename In, typename T>
void fft_n(const In* r_in, etl::complex<T>* r_out, const std::size_t n) {
    //0. Factorize

    std::size_t factors[MAX_FACTORS];
    std::size_t n_factors = 0;

    fft_factorize(n, factors, n_factors);

    //1. Precompute twiddle factors (stored in trig)

    etl::complex<T>* twiddle[MAX_FACTORS];

    auto trig = twiddle_compute(n, factors, n_factors, twiddle);

    //2. Perform the FFT itself

    fft_perform(r_in, r_out, n, factors, n_factors, twiddle);
}

/*!
 * \brief Compute many general FFT of all the signals in r_in
 * \param r_in The input signal
 * \param r_out The output signal
 * \param batch The number of signals
 * \param n The size of the tranform
 */
template <typename In, typename T>
void fft_n_many(const In* r_in, etl::complex<T>* r_out, const std::size_t batch, const std::size_t n) {
    const std::size_t distance = n; //in/out distance between samples

    //0. Factorize

    std::size_t factors[MAX_FACTORS];
    std::size_t n_factors = 0;

    fft_factorize(n, factors, n_factors);

    //1. Precompute twiddle factors (stored in trig)

    etl::complex<T>* twiddle[MAX_FACTORS];

    auto trig = twiddle_compute(n, factors, n_factors, twiddle);

    //2. Perform all the FFT itself

    auto batch_fun_b = [&](const size_t first, const size_t last) {
        for (std::size_t b = first; b < last; ++b) {
            fft_perform(r_in + b * distance, r_out + b * distance, n, factors, n_factors, twiddle);
        }
    };

    dispatch_1d_any(select_parallel(batch, 8), batch_fun_b, 0, batch);
}

template <typename T>
void inplace_radix2_fft1(etl::complex<T>* x, std::size_t N) {
    using complex_t = etl::complex<T>;

    //Decimate
    for (std::size_t a = 0, b = 0; a < N; ++a) {
        if (b > a) {
            std::swap(x[a], x[b]);
        }

        std::size_t bit = N;
        do {
            bit >>= 1;
            b ^= bit;
        } while ((b & bit) == 0 && bit != 1);
    }

    constexpr T pi = M_PIl;

    const auto NN = size_t(1) << size_t(std::log2(N));

    for (std::size_t s = 1; s <= std::log2(N); ++s) {
        auto m = size_t(1) << s;
        complex_t w(1.0, 0.0);
        complex_t wm(cos(2 * -pi / m), sin(2 * -pi / m));
        for (std::size_t j = 0; j < m / 2; ++j) {
            for (std::size_t k = j; k < NN; k += m) {
                auto t = w * x[k + m / 2];

                complex_t u  = x[k];
                x[k]         = u + t;
                x[k + m / 2] = u - t;
            }

            w *= wm;
        }
    }
}

/*!
 * \brief Kernel for 1D FFT. This kernel selects the best
 * implementation between general FFT and radix 2 FFT
 * \param a The input signal
 * \param n The size of the tranform
 * \param c The output signal
 */
template <typename T1, typename T>
void fft1_kernel(const T1* a, std::size_t n, std::complex<T>* c) {
    if (n <= 131072 && math::is_power_of_two(n)) {
        std::copy_n(a, n, c);

        detail::inplace_radix2_fft1(reinterpret_cast<etl::complex<T>*>(c), n);
    } else {
        detail::fft_n(a, reinterpret_cast<etl::complex<T>*>(c), n);
    }
}

/*!
 * \brief Kernel for Inverse 1D FFT. This kernel selects the best
 * implementation between general FFT and radix 2 FFT
 * \param a The input signal
 * \param n The size of the tranform
 * \param c The output signal
 */
template <typename T>
void ifft1_kernel(const std::complex<T>* a, std::size_t n, std::complex<T>* c) {
    using complex_t = std::complex<T>;

    if (n <= 131072 && math::is_power_of_two(n)) {
        //Conjugate the complex numbers
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = std::conj(a[i]);
        }

        // Forward FFT
        detail::inplace_radix2_fft1(reinterpret_cast<etl::complex<T>*>(c), n);
    } else {
        auto a_complex = allocate<complex_t>(n);
        auto x         = a_complex.get();

        //Conjugate the complex numbers
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::conj(a[i]);
        }

        //Foward FFT
        detail::fft_n(a_complex.get(), reinterpret_cast<etl::complex<T>*>(c), n);
    }

    //Conjugate the complex numbers again
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = std::conj(c[i]);
    }

    //Scale the numbers
    for (std::size_t i = 0; i < n; ++i) {
        c[i] /= double(n);
    }
}

/*!
 * \brief Performs a 1D full convolution using FFT
 * \param a The input
 * \param m The size of the input
 * \param b The kernel
 * \param n The size of the kernel
 * \param c The output
 */
template<typename T>
void conv1_full_kernel(const T* a, std::size_t m, const T* b, std::size_t n, T* c){
    const std::size_t size = m + n - 1;

    dyn_matrix<etl::complex<T>, 1> a_padded(size);
    dyn_matrix<etl::complex<T>, 1> b_padded(size);

    direct_copy(a, a + m, a_padded.memory_start());
    direct_copy(b, b + n, b_padded.memory_start());

    detail::fft1_kernel(reinterpret_cast<std::complex<T>*>(a_padded.memory_start()), size, reinterpret_cast<std::complex<T>*>(a_padded.memory_start()));
    detail::fft1_kernel(reinterpret_cast<std::complex<T>*>(b_padded.memory_start()), size, reinterpret_cast<std::complex<T>*>(b_padded.memory_start()));

    a_padded *= b_padded;

    detail::ifft1_kernel(reinterpret_cast<std::complex<T>*>(a_padded.memory_start()), size, reinterpret_cast<std::complex<T>*>(a_padded.memory_start()));

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real;
    }
}

/*!
 * \brief Performs a 2D full convolution using FFT
 * \param a The input
 * \param m1 The first dimension of the input
 * \param m2 The second dimension of the input
 * \param b The kernel
 * \param n1 The first dimension of the kernel
 * \param n2 The second dimension of the kernel
 * \param c The output
 * \param beta Indicates how the output is modified c = beta * c + o
 */
template<typename T>
void conv2_full_kernel(const T* a, std::size_t m1, std::size_t m2, const T* b, std::size_t n1, std::size_t n2, T* c, T beta){
    const std::size_t s1 = m1 + n1 - 1;
    const std::size_t s2 = m2 + n2 - 1;
    const std::size_t n = s1 * s2;

    // 0. Pad a and b to the size of c

    dyn_matrix<etl::complex<T>, 2> a_padded(s1, s2);
    dyn_matrix<etl::complex<T>, 2> b_padded(s1, s2);

    for (std::size_t i = 0; i < m1; ++i) {
        direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
    }

    for (std::size_t i = 0; i < n1; ++i) {
        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
    }

    // 1. FFT of a and b

    // a = fft2(a)
    detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s1, s2);
    a_padded.transpose_inplace();
    detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s2, s1);
    a_padded.transpose_inplace();

    // b = fft2(b)
    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s1, s2);
    b_padded.transpose_inplace();
    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s2, s1);
    b_padded.transpose_inplace();

    // 2. Elementwise multiplication of and b

    a_padded >>= b_padded;

    // 3. Inverse FFT of a

    // a = conj(a)
    a_padded = conj(a_padded);

    // a = fft2(a)
    detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s1, s2);
    a_padded.transpose_inplace();
    detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s2, s1);
    a_padded.transpose_inplace();

    // 4. Keep only the real part of the inverse FFT

    // c = real(conj(a) / n)
    // Note: Since the conjugate does not change the real part, it is not necessary
    if(beta == T(0.0)){
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a_padded[i].real / T(n);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = beta * c[i] + a_padded[i].real / T(n);
        }
    }
}

} //end of namespace detail

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    detail::fft1_kernel(a.memory_start(), etl::size(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    detail::ifft1_kernel(a.memory_start(), etl::size(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft1_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    for (std::size_t k = 0; k < etl::dim<0>(a); ++k) {
        detail::ifft1_kernel(a(k).memory_start(), etl::dim<1>(a), c(k).memory_start());
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1_real(A&& a, C&& c) {
    using complex_t = value_t<A>;

    a.ensure_cpu_up_to_date();

    std::size_t n = etl::size(a);

    auto c_complex = allocate<complex_t>(n);
    auto cc        = c_complex.get();

    detail::ifft1_kernel(a.memory_start(), n, cc);

    for(size_t i = 0; i < n; ++i){
        c[i] = real(cc[i]);
    }

    c.invalidate_gpu();
}

template<typename A, typename C>
void fft1_many_kernel(const A* a, C* c, size_t batch, size_t n) {
    std::size_t distance = n; //Distance between samples

    if (n <= 65536 && math::is_power_of_two(n)) {
        //Copy a -> c (if not aliasing)
        if (reinterpret_cast<const void*>(a) != reinterpret_cast<const void*>(c)) {
            direct_copy(a, a + batch * n, c);
        }

        auto batch_fun_b = [&](const size_t first, const size_t last) {
            for (std::size_t i = first; i < last; ++i) {
                detail::inplace_radix2_fft1(reinterpret_cast<etl::complex<typename C::value_type>*>(c + i * distance), n);
            }
        };

        dispatch_1d_any(select_parallel(batch, 8), batch_fun_b, 0, batch);
    } else {
        detail::fft_n_many(a, reinterpret_cast<etl::complex<typename C::value_type>*>(c), batch, n);
    }
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft1_many(A&& a, C&& c) {
    static constexpr size_t N = etl::dimensions(a);

    a.ensure_cpu_up_to_date();

    std::size_t n        = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch    = etl::size(a) / n;   //Number of batch

    fft1_many_kernel(a.memory_start(), c.memory_start(), batch, n);

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft2(A&& a, C&& c) {
    //Note: We need dyn here because of transposition inplace
    auto w = etl::force_temporary_dyn(c);

    //Perform FFT on each rows
    fft1_many(a, w);

    w.transpose_inplace();

    //Perform FFT on each columns
    fft1_many(w, w);

    w.transpose_inplace();

    c = w;
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2(A&& a, C&& c) {
    using T = typename value_t<C>::value_type;

    std::size_t n = etl::size(a);

    //Conjugate the complex numbers
    c = conj(a);

    fft2(c, c);

    //Conjugate the complex numbers again
    // and scale the numbers
    c = conj(c) / T(n);
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft2_many(A&& a, C&& c);

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft2_many(A&& a, C&& c) {
    using T = typename value_t<C>::value_type;

    constexpr std::size_t D = etl::decay_traits<A>::dimensions();

    std::size_t n = etl::dim<D - 2>(a) * etl::dim<D -1>(a);

    //Conjugate the complex numbers
    c = conj(a);

    fft2_many(c, c);

    //Conjugate the complex numbers again
    // and scale the numbers
    c = conj(c) / T(n);
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real(A&& a, C&& c) {
    auto w = etl::force_temporary(a);

    ifft2(a, w);

    c = real(w);
}

/*!
 * \copydoc fft2_many
 */
template <typename A, typename C>
void fft2_many(A&& a, C&& c) {
    //Note: we need dyn matrix for inplace rectangular transpose
    auto w = etl::force_temporary_dyn(c);

    fft1_many(a, w);

    w.deep_transpose_inplace();

    fft1_many(w, w);

    w.deep_transpose_inplace();

    c = w;
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv1_full_fft(A&& a, B&& b, C&& c) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    detail::conv1_full_kernel(a.memory_start(), etl::size(a), b.memory_start(), etl::size(b), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename II, typename KK, typename CC>
void conv2_full_fft(II&& a, KK&& b, CC&& c) {
    using T = value_t<II>;

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(), T(0.0));

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the already flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename II, typename KK, typename CC>
void conv2_full_fft_flipped(II&& a, KK&& b, CC&& c) {
    using T = value_t<II>;

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 2> prepared_b(etl::dim<0>(b), etl::dim<1>(b));

    std::copy(b.memory_start(), b.memory_end(), prepared_b.memory_start());

    prepared_b.fflip_inplace();

    detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), prepared_b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(), T(0.0));

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename II, typename KK, typename CC>
void conv2_full_multi_fft(II&& input, KK&& kernel, CC&& conv) {
    using T = value_t<II>;

    const auto K = kernel.dim(0);

    if(K){
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        const auto k_s = kernel.dim(1) * kernel.dim(2);
        const auto c_s = conv.dim(1) * conv.dim(2);

        const auto m1 = input.dim(0);
        const auto m2 = input.dim(1);

        const auto n1 = kernel.dim(1);
        const auto n2 = kernel.dim(2);

        const auto s1  = m1 + n1 - 1;
        const auto s2  = m2 + n2 - 1;
        const auto n   = s1 * s2;

        dyn_matrix<etl::complex<T>, 2> a_padded(s1, s2);

        for (std::size_t i = 0; i < m1; ++i) {
            direct_copy_n(input.memory_start() + i * m2, a_padded.memory_start() + i * s2, m2);
        }

        // a = fft2(a)
        detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s1, s2);
        a_padded.transpose_inplace();
        detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s2, s1);
        a_padded.transpose_inplace();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            SERIAL_SECTION {
                for (std::size_t k = first; k < last; ++k) {
                    const T* b = kernel.memory_start() + k * k_s;
                    T* c       = conv.memory_start() + k * c_s;

                    // 0. Pad a and b to the size of c

                    dyn_matrix<etl::complex<T>, 2> b_padded(s1, s2);

                    for (std::size_t i = 0; i < n1; ++i) {
                        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
                    }

                    // 1. FFT of a and b

                    // b = fft2(b)
                    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s1, s2);
                    b_padded.transpose_inplace();
                    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s2, s1);
                    b_padded.transpose_inplace();

                    // 2. Elementwise multiplication of and b

                    b_padded >>= a_padded;

                    // 3. Inverse FFT of a

                    // a = conj(a)
                    b_padded = conj(b_padded);

                    // a = fft2(a)
                    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s1, s2);
                    b_padded.transpose_inplace();
                    detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s2, s1);
                    b_padded.transpose_inplace();

                    // 4. Keep only the real part of the inverse FFT

                    // c = real(conj(a) / n)
                    // Note: Since the conjugate does not change the real part, it is not necessary

                    for (std::size_t i = 0; i < etl::size(b_padded); ++i){
                        c[i] = b_padded[i].real / T(n);
                    }
                }
            }
        };

        if (etl::is_parallel) {
            dispatch_1d_any(select_parallel(K, 2), batch_fun_k, 0, K);
        } else {
            batch_fun_k(0, K);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename II, typename KK, typename CC>
void conv2_full_multi_flipped_fft(II&& input, KK&& kernel, CC&& conv) {
    using T = value_t<II>;

    kernel.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 3> prepared_k(kernel.dim(0), kernel.dim(1), kernel.dim(2));

    std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

    prepared_k.deep_fflip_inplace();

    conv2_full_multi_fft(input, prepared_k, conv);
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename II, typename KK, typename CC>
void conv4_full_fft(II&& input, KK&& kernel, CC&& conv) {
    using T = value_t<II>;

    if (kernel.dim(1) > 0) {
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_k_inc = input.dim(2) * input.dim(3);

        std::size_t m1 = input.dim(2);
        std::size_t m2 = input.dim(3);

        std::size_t n1 = kernel.dim(2);
        std::size_t n2 = kernel.dim(3);

        const std::size_t s1 = m1 + n1 - 1;
        const std::size_t s2 = m2 + n2 - 1;
        const std::size_t n  = s1 * s2;

        const std::size_t N = input.dim(0);

        std::fill(conv.memory_start(), conv.memory_end(), 0);

        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                SERIAL_SECTION {
                    for (std::size_t i = first; i < last; ++i) {
                        for (std::size_t k = 0; k < kernel.dim(0); ++k) {
                            const T* a = input.memory_start() + i * input_i_inc + k * input_k_inc; //input(i)(k)

                            dyn_matrix<etl::complex<T>, 2> a_padded(s1, s2);
                            dyn_matrix<etl::complex<T>, 2> b_padded(s1, s2);
                            dyn_matrix<etl::complex<T>, 2> tmp(s1, s2);

                            a_padded = 0;

                            for (std::size_t i = 0; i < m1; ++i) {
                                direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
                            }

                            // a = fft2(a)
                            detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s1, s2);
                            a_padded.transpose_inplace();
                            detail::fft_n_many(a_padded.memory_start(), a_padded.memory_start(), s2, s1);
                            a_padded.transpose_inplace();

                            for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                                const T* b = kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc; //kernel(k)(c)
                                T* cc      = conv.memory_start() + i * conv_i_inc + c * conv_c_inc;       //conv(i)(c)

                                // 0. Pad a and b to the size of cc

                                b_padded = 0;
                                for (std::size_t i = 0; i < n1; ++i) {
                                    direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
                                }

                                // 1. FFT of a and b

                                // b = fft2(b)
                                detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s1, s2);
                                b_padded.transpose_inplace();
                                detail::fft_n_many(b_padded.memory_start(), b_padded.memory_start(), s2, s1);
                                b_padded.transpose_inplace();

                                // 2. Elementwise multiplication of and b

                                tmp = a_padded >> b_padded;

                                // 3. Inverse FFT of a

                                // a = conj(a)
                                for (std::size_t i = 0; i < n; ++i) {
                                    tmp[i] = etl::conj(tmp[i]);
                                }

                                // a = fft2(a)
                                detail::fft_n_many(tmp.memory_start(), tmp.memory_start(), s1, s2);
                                tmp.transpose_inplace();
                                detail::fft_n_many(tmp.memory_start(), tmp.memory_start(), s2, s1);
                                tmp.transpose_inplace();

                                // 4. Keep only the real part of the inverse FFT

                                for (std::size_t i = 0; i < n; ++i) {
                                    cc[i] += tmp[i].real / T(n);
                                }
                            }
                        }
                    }
                }
            }
        };

        if (etl::is_parallel) {
            dispatch_1d_any(select_parallel(N, 2), batch_fun_n, 0, N);
        } else {
            batch_fun_n(0, N);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename II, typename KK, typename CC>
void conv4_full_fft_flipped(II&& input, KK&& kernel, CC&& conv) {
    using T = value_t<II>;

    if (kernel.dim(1) > 0) {
        kernel.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 4> prepared_k(kernel.dim(0), kernel.dim(1), kernel.dim(2), kernel.dim(3));

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        conv4_full_fft(input, prepared_k, conv);
    }
}

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
