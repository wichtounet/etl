//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

namespace {

constexpr const std::size_t MAX_FACTORS = 32;

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
            out[j + offset] = w1 * std::complex<T>(t2.real() - t3.imag(), t2.imag() + t3.real());
            out[j + 2 * offset] = w2 * std::complex<T>(t2.real() + t3.imag(), t2.imag() - t3.real());
        }
    }
}

inline void fft_factorize(std::size_t n, std::size_t* factors, std::size_t& n_factors){
    //0. Favour the factors with implemented transform modules

    while(n > 1){
        if(n % 3 == 0){
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

    std::size_t prime_factor = 5;

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
            std::size_t product_1 = product;
            product *= factor;
            std::size_t q = n / product;

            for (std::size_t j = 1; j < factor; j++){
                std::size_t m = 0;
                for (std::size_t k = 1; k <= q; k++){
                    m = (m + j * product_1) % n;

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
        } else {
            std::cout << "unkwown factor" << std::endl;
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
        CHECK(c1[i].real() == Approx(c2[i].real()));
        CHECK(c1[i].imag() == Approx(c2[i].imag()));
    }
}
