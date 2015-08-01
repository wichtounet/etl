//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "../../config.hpp"

namespace etl {

namespace impl {

namespace standard {

namespace detail {

inline std::size_t next_power_of_two(std::size_t n) {
    return std::pow(2, static_cast<std::size_t>(std::ceil(std::log2(n))));
}

template<typename T>
void inplace_radix2_fft1(std::complex<T>* x, std::size_t N){
    using complex_t = std::complex<T>;

    //Decimate
    std::size_t m = std::log2(N);
    for (std::size_t a = 0; a < N; a++){
        std::size_t b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);

        if (b > a){
            std::swap(x[a], x[b]);
        }
    }

    const T pi = M_PIl;

    const std::size_t NN = 1 << std::size_t(std::log2(N));

    for (std::size_t s = 1; s <= std::log2(N); ++s) {
        std::size_t m = 1 << s;
        complex_t w(1, 0);
        complex_t wm(cos(2 * -pi / m), sin(2 * -pi / m));
        for (std::size_t j=0; j < m/2; ++j) {
            for (std::size_t k=j; k < NN; k += m) {
                complex_t t = w * x[k + m/2];
                complex_t u = x[k];
                x[k] = u + t;
                x[k + m/2] = u - t;
            }
            w *= wm;
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

} //end of namespace detail

//(T or complex<T>) -> complex<T>
template<typename A, typename C>
void fft1(A&& a, C&& c){
    using complex_t = std::conditional_t<
        etl::is_complex<A>::value,
        value_t<A>,
        std::complex<value_t<A>>>;

    std::size_t n = etl::size(a);
    std::size_t N = detail::next_power_of_two(n);

    if(N == n){
        std::copy(a.begin(), a.end(), c.begin());

        detail::inplace_radix2_fft1(c.memory_start(), n);
    } else {
        auto a_complex = allocate<complex_t>(n);
        auto x = a_complex.get();

        std::copy(a.begin(), a.end(), x);

        detail::czt1(x, n, c.memory_start());
    }
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

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
