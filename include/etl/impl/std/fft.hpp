//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/config.hpp"
#include "etl/allocator.hpp"

namespace etl {

namespace impl {

namespace standard {

namespace detail {

//The efficient sub transforms modules have been taken from the "FFT Algorithms"
//paper by Brian Gough, 1997

constexpr const std::size_t MAX_FACTORS = 32;

//inverse of the conj
template<typename T>
std::complex<T> inverse_conj(std::complex<T> x){
    return {-x.imag(), x.real()};
}

//conj of the inverse
template<typename T>
std::complex<T> conj_inverse(std::complex<T> x){
    return {x.imag(), -x.real()};
}

template<typename T>
void fft_2_point(const std::complex<T>* in, std::complex<T>* out, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle){
    static constexpr const std::size_t factor = 2;

    const std::size_t m = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc = (factor - 1) * offset;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc){
        std::complex<T> w(1.0, 0.0);

        if(k > 0){
            w = twiddle[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j){
            auto z0 = in[i];
            auto z1 = in[i+m];

            out[j] = z0 + z1;
            out[j+offset] = w * (z0 - z1);
        }
    }
}

template<typename T>
void fft_3_point(const std::complex<T>* in, std::complex<T>* out, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle1, const std::complex<T>* twiddle2){
    static constexpr const std::size_t factor = 3;

    const std::size_t m = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc = (factor - 1) * offset;

    static constexpr const T tau = std::sqrt(3.0) / 2.0;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; k++, j += inc){
        std::complex<T> w1(1.0, 0.0);
        std::complex<T> w2(1.0, 0.0);

        if (k > 0){
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j){
            auto z0 = in[i];
            auto z1 = in[i+m];
            auto z2 = in[i+2*m];

            auto t1 = z1 + z2;
            auto t2 = z0 - t1 / T(2.0);
            auto t3 = -tau * (z1 - z2);

            out[j] = z0 + t1;
            out[j + offset] = w1 * (t2 + inverse_conj(t3));
            out[j + 2 * offset] = w2 * (t2 + conj_inverse(t3));
        }
    }
}

template<typename T>
void fft_5_point(const std::complex<T>* in, std::complex<T>* out, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle1, const std::complex<T>* twiddle2, const std::complex<T>* twiddle3, const std::complex<T>* twiddle4){
    static constexpr const std::size_t factor = 5;

    const std::size_t m = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc = (factor - 1) * offset;

    static constexpr const T theta_1 = -1.0 * std::sin(2.0 * M_PI / 5.0);
    static constexpr const T theta_2 = -1.0 * std::sin(2.0 * M_PI / 10.0);

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc){
        std::complex<T> w1(1.0, 0.0);
        std::complex<T> w2(1.0, 0.0);
        std::complex<T> w3(1.0, 0.0);
        std::complex<T> w4(1.0, 0.0);

        if (k > 0){
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
            w3 = twiddle3[k - 1];
            w4 = twiddle4[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j){
            auto z0 = in[i];
            auto z1 = in[i + m];
            auto z2 = in[i + 2 * m];
            auto z3 = in[i + 3 * m];
            auto z4 = in[i + 4 * m];

            auto t1 = z1 + z4;
            auto t2 = z2 + z3;
            auto t3 = z1 - z4;
            auto t4 = z2 - z3;
            auto t5 = t1 + t2;
            auto t6 = T(std::sqrt(5.0) / 4.0) * (t1 - t2);
            auto t7 = z0 - (t5 / T(4));
            auto t8 = t7 + t6;
            auto t9 = t7 - t6;
            auto t10 = theta_1 * t3 + theta_2 * t4;
            auto t11 = theta_2 * t3 - theta_1 * t4;

            out[j] = z0 + t5;
            out[j + offset] = w1 * (t8 + inverse_conj(t10));
            out[j + 2 * offset] = w2 * (t9 + inverse_conj(t11));
            out[j + 3 * offset] = w3 * (t9 - inverse_conj(t11));
            out[j + 4 * offset] = w4 * (t8 - inverse_conj(t10));
        }
    }
}

template<typename T>
void fft_7_point(const std::complex<T>* in, std::complex<T>* out, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle1, const std::complex<T>* twiddle2, const std::complex<T>* twiddle3, const std::complex<T>* twiddle4, const std::complex<T>* twiddle5, const std::complex<T>* twiddle6){
    static constexpr const std::size_t factor = 7;

    const std::size_t m = n / factor;
    const std::size_t offset = product / factor;
    const std::size_t inc = (factor - 1) * offset;

    static constexpr const T theta_0 = 2.0 * M_PI / 7.0;

    static constexpr const T theta_1 = (std::cos(theta_0) + std::cos(2.0 * theta_0) + std::cos(3.0 * theta_0)) / 3.0 - 1.0;
    static constexpr const T theta_2 = (2.0 * std::cos(theta_0) - std::cos(2.0 * theta_0) - std::cos(3.0 * theta_0)) / 3.0;
    static constexpr const T theta_3 = (std::cos(theta_0) - 2.0 * std::cos(2.0 * theta_0) + std::cos(3.0 * theta_0)) / 3.0;
    static constexpr const T theta_4 = (std::cos(theta_0) + std::cos(2.0 * theta_0) - 2.0 * std::cos(3.0 * theta_0)) / 3.0;

    static constexpr const T theta_5 = (std::sin(theta_0) + std::sin(2.0 * theta_0) - std::sin(3.0 * theta_0)) / 3.0;
    static constexpr const T theta_6 = (2.0 * std::sin(theta_0) - std::sin(2.0 * theta_0) + std::sin(3.0 * theta_0)) / 3.0;
    static constexpr const T theta_7 = (std::sin(theta_0) - 2.0 * std::sin(2.0 * theta_0) - std::sin(3.0 * theta_0)) / 3.0;
    static constexpr const T theta_8 = (std::sin(theta_0) + std::sin(2.0 * theta_0) + 2.0 * std::sin(3.0 * theta_0)) / 3.0;

    for (std::size_t k = 0, i = 0, j = 0; k < n / product; ++k, j += inc){
        std::complex<T> w1(1.0, 0.0);
        std::complex<T> w2(1.0, 0.0);
        std::complex<T> w3(1.0, 0.0);
        std::complex<T> w4(1.0, 0.0);
        std::complex<T> w5(1.0, 0.0);
        std::complex<T> w6(1.0, 0.0);

        if (k > 0){
            w1 = twiddle1[k - 1];
            w2 = twiddle2[k - 1];
            w3 = twiddle3[k - 1];
            w4 = twiddle4[k - 1];
            w5 = twiddle5[k - 1];
            w6 = twiddle6[k - 1];
        }

        for (std::size_t k1 = 0; k1 < offset; k1++, ++i, ++j){
            auto z0 = in[i];
            auto z1 = in[i + m];
            auto z2 = in[i + 2 * m];
            auto z3 = in[i + 3 * m];
            auto z4 = in[i + 4 * m];
            auto z5 = in[i + 5 * m];
            auto z6 = in[i + 6 * m];

            auto t0 = z1 + z6;
            auto t1 = z1 - z6;
            auto t2 = z2 + z5;
            auto t3 = z2 - z5;
            auto t4 = z4 + z3;
            auto t5 = z4 - z3;
            auto t6 = t2 + t0;
            auto t8 = z0 + t6 + t4;
            auto t10 = theta_2 * (t0 - t4);
            auto t11 = theta_3 * (t4 - t2);
            auto t12 = theta_4 * (t2 - t0);
            auto t13 = theta_5 * (t5 + t3 + t1);
            auto t14 = theta_6 * (t1 - t5);
            auto t15 = theta_7 * (t5 - t3);
            auto t16 = theta_8 * (t3 - t1);
            auto t17 = t8 + theta_1 * (t6 + t4);
            auto t18 = t17 + t10 + t11;
            auto t19 = t17 + t12 - t11;
            auto t20 = t17 - t10 - t12;
            auto t21 = t14 + t15 + t13;
            auto t22 = t16 - t15 + t13;
            auto t23 = -t16 - t14 + t13;

            out[j] = t8;
            out[j + 1 * offset] = w1 * (t18 + conj_inverse(t21));
            out[j + 2 * offset] = w2 * (t20 + conj_inverse(t23));
            out[j + 3 * offset] = w3 * (t19 + inverse_conj(t22));
            out[j + 4 * offset] = w4 * (t19 + conj_inverse(t22));
            out[j + 5 * offset] = w5 * (t20 + inverse_conj(t23));
            out[j + 6 * offset] = w6 * (t18 + inverse_conj(t21));
        }
    }
}

template<typename T>
void fft_n_point(std::complex<T>* in, std::complex<T>* out, const std::size_t factor, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle){
    const std::size_t m = n / factor;
    const std::size_t q = n / product;
    const std::size_t offset = product / factor;
    const std::size_t inc = (factor - 1) * offset;
    const std::size_t factor_limit = (factor - 1) / 2 + 1;

    std::copy_n(in, m, out);

    for (std::size_t i = 1; i < (factor - 1) / 2 + 1; i++){
        std::transform(in + i * m, in + i * m + m, in + (factor - i) * m, out + i * m, std::plus<std::complex<T>>());
        std::transform(in + i * m, in + i * m + m, in + (factor - i) * m, out + (factor - i) * m, std::minus<std::complex<T>>());
    }

    std::copy_n(out, m, in);

    for (std::size_t i = 1; i < factor_limit; i++){
        std::transform(in, in + m, out + i * m, in, std::plus<std::complex<T>>());
    }

    for (std::size_t e = 1; e < factor_limit; e++){
        std::copy_n(out, m, in + e * m);
        std::copy_n(out, m, in + (factor - e) * m);

        for (std::size_t k = 1, j = e * q; k < (factor - 1) / 2 + 1; k++){
            std::complex<T> w(1.0, 0.0);

            if (j > 0) {
                w = twiddle[j - 1];
            }

            for (std::size_t i = 0; i < m; i++){
                auto xp = out[i + k * m];
                auto xm = out[i + (factor - k) * m];

                in[i + e * m] += w.real() * xp - w.imag() * conj_inverse(xm);
                in[i + (factor - e) * m] += w.real() * xp + w.imag() * conj_inverse(xm);
            }

            j = (j + (e * q)) % (factor * q);
        }
    }

    std::copy_n(in, offset, out);

    for (std::size_t i = 1; i < factor; i++){
        std::copy_n(in + i * m, offset, out + i * offset);
    }

    for (std::size_t i = offset, j = product; i < offset + (q - 1) * offset; i += offset, j += offset + inc){
        std::copy_n(in + i, offset, out + j);
    }

    for (std::size_t k = 1, i = offset, j = product; k < q; ++k, j += inc){
        for (std::size_t k1 = 0; k1 < offset; ++k1, ++i, ++j){
            for (std::size_t e = 1; e < factor; e++){
                //out = w * x
                out[j + e * offset] = twiddle[(e-1)*q + k-1] * in[i + e * m];
            }
        }
    }
}

inline void fft_factorize(std::size_t n, std::size_t* factors, std::size_t& n_factors){
    //0. Favour the factors with implemented transform modules

    while(n > 1){
        if(n % 7 == 0){
            n /= 7;
            factors[n_factors++] = 7;
        } else if(n % 5 == 0){
            n /= 5;
            factors[n_factors++] = 5;
        } else if(n % 3 == 0){
            n /= 3;
            factors[n_factors++] = 3;
        } else if(n % 2 == 0){
            n /= 2;
            factors[n_factors++] = 2;
        } else {
            //At this point, there are no transform module
            break;
        }
    }

    //1. Search for prime factors

    std::size_t prime_factor = 11;

    while (n > 1){
        //Search for the next prime factor
        while (n % prime_factor != 0){
            prime_factor += 2;
        }

        n /= prime_factor;
        factors[n_factors++] = prime_factor;
    }
}

template<typename T>
void fft_n(const std::complex<T>* r_in, std::complex<T>* r_out, const std::size_t n){
    std::complex<T>* twiddle[MAX_FACTORS];

    //0. Factorize

    std::size_t factors[MAX_FACTORS];
    std::size_t n_factors = 0;

    fft_factorize(n, factors, n_factors);

    //1. Precompute twiddle factors

    auto trig = etl::allocate<std::complex<T>>(n);

    {
        const T d_theta = -2.0 * M_PI / ((T) n);

        std::size_t t = 0;
        std::size_t product = 1;

        for (std::size_t i = 0; i < n_factors; i++){
            std::size_t factor = factors[i];
            twiddle[i] = &trig[0] + t;

            std::size_t prev_product = product;
            product *= factor;

            for (std::size_t j = 1; j < factor; j++){
                std::size_t m = 0;
                for (std::size_t k = 1; k <= n / product; k++){
                    m = (m + j * prev_product) % n;

                    T theta = d_theta * m;

                    trig[t] = {std::cos(theta), std::sin(theta)};

                    t++;
                }
            }
        }
    }

    auto tmp = etl::allocate<std::complex<T>>(n);

    std::copy(r_in, r_in + n, tmp.get());

    auto* in = tmp.get();
    auto* out = r_out;

    std::size_t product = 1;

    for (std::size_t i = 0; i < n_factors; i++){
        std::size_t factor = factors[i];

        if (i > 0) {
            std::swap(in, out);
        }

        product *= factor;

        std::size_t offset = n / product;

        if(factor == 2){
            fft_2_point(in, out, product, n, twiddle[i]);
        } else if(factor == 3){
            fft_3_point(in, out, product, n, twiddle[i], twiddle[i] + offset);
        } else if(factor == 5){
            fft_5_point(in, out, product, n, twiddle[i], twiddle[i] + offset, twiddle[i] + 2 * offset, twiddle[i] + 3 * offset);
        } else if(factor == 7){
            fft_7_point(in, out, product, n, twiddle[i], twiddle[i] + offset, twiddle[i] + 2 * offset, twiddle[i] + 3 * offset, twiddle[i] + 4 * offset, twiddle[i] + 5 * offset);
        } else {
            fft_n_point(in, out, factor, product, n, twiddle[i]);
        }
    }

    if (out != r_out){
        std::copy(out, out + n, r_out);
    }
}

inline std::size_t next_power_of_two(std::size_t n) {
    return std::pow(2, static_cast<std::size_t>(std::ceil(std::log2(n))));
}

template<typename T>
inline std::complex<T> inline_mul(std::complex<T> a, std::complex<T> b){
    auto ac = a.real() * b.real();
    auto bd = a.imag() * b.imag();

    auto abcd = (a.real() + a.imag()) * (b.real() + b.imag());

    return {ac - bd, abcd - ac - bd};
}

template<typename T>
void inplace_radix2_fft1(std::complex<T>* x, std::size_t N){
    using complex_t = std::complex<T>;

    //Decimate
    for (std::size_t a = 0, b = 0; a < N; ++a){
        if (b > a){
            std::swap(x[a], x[b]);
        }

        auto bit = N;
        do {
            bit >>= 1;
            b ^= bit;
        } while( (b & bit) == 0 && bit != 1 );
    }

    constexpr const T pi = M_PIl;

    const std::size_t NN = 1 << std::size_t(std::log2(N));

    for (std::size_t s = 1; s <= std::log2(N); ++s) {
        std::size_t m = 1 << s;
        complex_t w(1, 0);
        complex_t wm(cos(2 * -pi / m), sin(2 * -pi / m));
        for (std::size_t j=0; j < m/2; ++j) {
            for (std::size_t k=j; k < NN; k += m) {
                auto t = inline_mul(w, x[k + m/2]);

                complex_t u = x[k];
                x[k] = u + t;
                x[k + m/2] = u - t;
            }

            w = inline_mul(w, wm);
        }
    }
}

template<typename T>
void inplace_radix2_ifft1(std::complex<T>* x, std::size_t N){
    //Conjugate the complex numbers
    for(std::size_t i = 0; i < N; ++i){
        x[i] = std::conj(x[i]);
    }

    // Forward FFT
    detail::inplace_radix2_fft1(x, N);

    //Conjugate the complex numbers again
    for(std::size_t i = 0; i < N; ++i){
        x[i] = std::conj(x[i]);
    }

    //Scale the numbers
    for(std::size_t i = 0; i < N; ++i){
        x[i] /= double(N);
    }
}

template<typename T>
void czt1(std::complex<T>* a, std::size_t n, std::complex<T>* c){
    using complex_t = std::complex<T>;

    std::size_t M = detail::next_power_of_two(2 * n - 1);

    auto a_complex = allocate<complex_t>(M);
    auto x = a_complex.get();

    auto v_complex = allocate<complex_t>(M);
    auto y = v_complex.get();

    constexpr const T pi = M_PIl;

    //Prepare x

    std::copy(a, a + n, x);
    std::fill(x + n, x + M, complex_t(0, 0));

    for(std::size_t i = 0; i < n; ++i){
        x[i] = x[i] * std::exp(complex_t(0, -(pi / n * i * i)));
    }

    //Prepare y

    for(std::size_t i = 0; i < n; ++i){
        y[i] = std::exp(complex_t(0,  pi / n * i * i));
    }

    std::fill(y + n, y + M - n, complex_t(0, 0));

    for(int i = 1 - n; i <= -1; ++i){
        y[M + i] = std::exp(complex_t(0, pi / n * i * i));
    }

    //Do the convolution

    detail::inplace_radix2_fft1(x, M);
    detail::inplace_radix2_fft1(y, M);

    for(std::size_t i = 0; i < M; ++i){
        y[i] = y[i] * x[i];
    }

    detail::inplace_radix2_ifft1(y, M);

    //Scale back

    for(std::size_t i = 0; i < M; ++i){
        y[i] = y[i] * std::exp(complex_t(0, -(pi / n * i * i)));
    }

    std::copy(y, y + n, c);
}

template<typename T>
void ifft1_kernel(const std::complex<T>* a, std::size_t n, std::complex<T>* c){
    using complex_t = std::complex<T>;

    std::size_t N = detail::next_power_of_two(n);

    if(N == n){
        std::copy(a, a + n, c);

        detail::inplace_radix2_ifft1(c, n);
    } else {
        auto a_complex = allocate<complex_t>(n);
        auto x = a_complex.get();

        //Conjugate the complex numbers
        for(std::size_t i = 0; i < n; ++i){
            x[i] = std::conj(a[i]);
        }

        // Forward FFT
        detail::czt1(x, n, c);

        //Conjugate the complex numbers again
        for(std::size_t i = 0; i < n; ++i){
            c[i] = std::conj(c[i]);
        }

        //Scale the numbers
        for(std::size_t i = 0; i < n; ++i){
            c[i] /= double(n);
        }
    }
}

template<typename T1, typename T>
void fft1_kernel(const T1* a, std::size_t n, std::complex<T>* c){
    using complex_t = std::complex<T>;

    std::size_t N = detail::next_power_of_two(n);

    if(N == n){
        std::copy(a, a + n, c);

        detail::inplace_radix2_fft1(c, n);
    } else {
        auto a_complex = allocate<complex_t>(n);
        auto x = a_complex.get();

        std::copy(a, a + n, x);

        detail::czt1(x, n, c);
    }
}

} //end of namespace detail

//(T or complex<T>) -> complex<T>
template<typename A, typename C>
void fft1(A&& a, C&& c){
    detail::fft1_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

//(T or complex<T>) -> complex<T>
template<typename A, typename C>
void fftn1(A&& a, C&& c){
    detail::fft_n(a.memory_start(), c.memory_start(), etl::size(a));
}

//complex<T> -> complex<T>
template<typename A, typename C>
void ifft1(A&& a, C&& c){
    detail::ifft1_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

//complex<T> -> T
template<typename A, typename C>
void ifft1_real(A&& a, C&& c){
    using complex_t = value_t<A>;

    std::size_t n = etl::size(a);

    auto c_complex = allocate<complex_t>(n);
    auto cc = c_complex.get();

    detail::ifft1_kernel(a.memory_start(), n, cc);

    for(std::size_t i = 0; i < n; ++i){
        c[i] = cc[i].real();
    }
}

//(T or complex<T>) -> complex<T>
template<typename A, typename C>
void fft2(A&& a, C&& c){
    auto w = etl::force_temporary_dyn(c);

    //Perform FFT on each rows
    for(std::size_t r = 0; r < etl::dim<0>(c); ++r){
        detail::fft1_kernel(a(r).memory_start(), etl::dim<1>(a), w(r).memory_start());
    }

    w.transpose_inplace();

    //Perform FFT on each columns
    for(std::size_t r = 0; r < etl::dim<1>(c); ++r){
        detail::fft1_kernel(w(r).memory_start(), etl::dim<0>(a), w(r).memory_start());
    }

    w.transpose_inplace();

    c = w;
}

//complex<T> -> complex<T>
template<typename A, typename C>
void ifft2(A&& a, C&& c){
    std::size_t n = etl::size(a);

    //Conjugate the complex numbers
    for(std::size_t i = 0; i < n; ++i){
        c[i] = std::conj(a[i]);
    }

    fft2(c, c);

    //Conjugate the complex numbers again
    for(std::size_t i = 0; i < n; ++i){
        c[i] = std::conj(c[i]);
    }

    //Scale the numbers
    for(std::size_t i = 0; i < n; ++i){
        c[i] /= double(n);
    }
}

//complex<T> -> T
template<typename A, typename C>
void ifft2_real(A&& a, C&& c){
    auto w = etl::force_temporary(a);

    ifft2(a, w);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        c[i] = w[i].real();
    }
}

template<typename A, typename B, typename C>
void fft1_convolve(A&& a, B&& b, C&& c){
    const auto m = etl::size(a);
    const auto n = etl::size(b);
    const auto size = m + n - 1;

    auto a_padded = allocate<std::complex<value_t<A>>>(size);
    auto b_padded = allocate<std::complex<value_t<A>>>(size);

    std::copy(a.begin(), a.end(), a_padded.get());
    std::copy(b.begin(), b.end(), b_padded.get());

    detail::fft1_kernel(a_padded.get(), size, a_padded.get());
    detail::fft1_kernel(b_padded.get(), size, b_padded.get());

    for(std::size_t i = 0; i < size; ++i){
        a_padded[i] *= b_padded[i];
    }

    detail::ifft1_kernel(a_padded.get(), size, a_padded.get());

    for(std::size_t i = 0; i < size; ++i){
        c[i] = a_padded[i].real();
    }
}

template<typename A, typename B, typename C>
void fft2_convolve(A&& a, B&& b, C&& c){
    const auto m1 = etl::dim<0>(a);
    const auto n1= etl::dim<0>(b);
    const auto s1 = m1 + n1 - 1;

    const auto m2 = etl::dim<1>(a);
    const auto n2= etl::dim<1>(b);
    const auto s2 = m2 + n2 - 1;

    dyn_matrix<std::complex<value_t<A>>, 2> a_padded(s1, s2);
    dyn_matrix<std::complex<value_t<A>>, 2> b_padded(s1, s2);

    for(std::size_t i = 0; i < m1; ++i){
        for(std::size_t j = 0; j < m2; ++j){
            a_padded(i, j) = a(i,j);
        }
    }

    for(std::size_t i = 0; i < n1; ++i){
        for(std::size_t j = 0; j < n2; ++j){
            b_padded(i, j) = b(i,j);
        }
    }

    fft2(a_padded, a_padded);
    fft2(b_padded, b_padded);

    a_padded *= b_padded;

    ifft2_real(a_padded, c);
}

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
