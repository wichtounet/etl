//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "test.hpp"

namespace {

template<typename T>
void fft_pass_2(const std::complex<T>* in, std::complex<T>* out, const std::size_t product, const std::size_t n, const std::complex<T>* twiddle){
    static constexpr const std::size_t factor = 2;

    const std::size_t m = n / factor;
    const std::size_t limit = product / factor;
    const std::size_t inc = (factor - 1) * limit;

    std::size_t i = 0;
    std::size_t j = 0;

    for (std::size_t k = 0; k < n / product; ++k, j += inc){
        std::complex<T> w(1.0, 0.0);

        if(k > 0){
            w = twiddle[k - 1];
        }

        for (std::size_t k1 = 0; k1 < limit; ++k1, ++i, ++j){
            auto z0 = in[i];
            auto z1 = in[i+m];

            out[j] = z0 + z1;
            out[j+limit] = w * (z0 - z1);
        }
    }
}

template<typename T>
void fft_n(const std::complex<T>* r_in, std::complex<T>* r_out, const std::size_t n){
    std::complex<T>* twiddle[65]; //One for each possible factor

    //0. Factorize

    std::size_t factors[65];
    std::size_t n_factors = 0;

    auto nn = n;
    while(nn > 1){
        nn /= 2;
        factors[n_factors++] = 2;
    }

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
                size_t m = 0;
                for (std::size_t k = 1; k <= q; k++){
                    m = m + j * product_1;
                    m = m % n;

                    double theta = d_theta * m;

                    trig[t].real(cos(theta));
                    trig[t].imag(sin(theta));

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
    std::size_t state = 0;

    for (std::size_t i = 0; i < n_factors; i++){
        if (first) {
            in = tmp.get();
            out = r_out;
            first = false;
        } else {
            in = r_out;
            out = tmp.get();
            first = true;
        }

        std::size_t factor = factors[i];

        product *= factor;

        if(factor == 2){
            fft_pass_2(in, out, product, n, twiddle[i]);
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
