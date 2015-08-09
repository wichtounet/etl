//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"
#include <functional>

namespace {

//The efficient sub transforms modules have been taken from the "FFT Algorithms"
//Paper by Brian Gough, 1997

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

    bool first = true;

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

}

TEMPLATE_TEST_CASE_2( "experimental/1", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 2> a;
    etl::fast_matrix<std::complex<Z>, 2> c1;
    etl::fast_matrix<std::complex<Z>, 2> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/4", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 3> a;
    etl::fast_matrix<std::complex<Z>, 3> c1;
    etl::fast_matrix<std::complex<Z>, 3> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(3.0, -3.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/2", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 4> a;
    etl::fast_matrix<std::complex<Z>, 4> c1;
    etl::fast_matrix<std::complex<Z>, 4> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/3", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 8> a;
    etl::fast_matrix<std::complex<Z>, 8> c1;
    etl::fast_matrix<std::complex<Z>, 8> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);
    a[6] = std::complex<Z>(2.0, -1.0);
    a[7] = std::complex<Z>(4.0, 3.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/5", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 6> a;
    etl::fast_matrix<std::complex<Z>, 6> c1;
    etl::fast_matrix<std::complex<Z>, 6> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/6", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 11> a;
    etl::fast_matrix<std::complex<Z>, 11> c1;
    etl::fast_matrix<std::complex<Z>, 11> c2;

    a[0] = std::complex<Z>(1.0, 1.0);
    a[1] = std::complex<Z>(2.0, 3.0);
    a[2] = std::complex<Z>(2.0, -1.0);
    a[3] = std::complex<Z>(4.0, 3.0);
    a[4] = std::complex<Z>(1.0, 1.0);
    a[5] = std::complex<Z>(2.0, 3.0);
    a[6] = std::complex<Z>(1.0, 1.0);
    a[7] = std::complex<Z>(2.0, 3.0);
    a[8] = std::complex<Z>(2.0, -1.0);
    a[9] = std::complex<Z>(4.0, 3.0);
    a[10] = std::complex<Z>(1.0, 1.0);

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()).epsilon(0.01));
        CHECK(c1[i].imag() == Approx(c2[i].imag()).epsilon(0.01));
    }
}

TEMPLATE_TEST_CASE_2( "experimental/7", "[fast][fft]", Z, float, double ) {
    etl::fast_matrix<std::complex<Z>, 7 * 11> a;
    etl::fast_matrix<std::complex<Z>, 7 * 11> c1;
    etl::fast_matrix<std::complex<Z>, 7 * 11> c2;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<Z> dist(-140.0, 250.0);
    auto d = std::bind(dist, gen);

    for(std::size_t i = 0; i < etl::size(a); ++i){
        a[i].real(d());
        a[i].imag(d());
    }

    c1 = etl::fft_1d(a);

    fft_n(a.memory_start(), c2.memory_start(), etl::size(a));

    for(std::size_t i = 0; i < etl::size(a); ++i){
        CHECK(c1[i].real() == Approx(c2[i].real()).epsilon(0.01));
        CHECK(c1[i].imag() == Approx(c2[i].imag()).epsilon(0.01));
    }
}
