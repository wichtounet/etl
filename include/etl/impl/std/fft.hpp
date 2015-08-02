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
