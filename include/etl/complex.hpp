//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template<typename T>
struct complex {
    T real;
    T imag;

    constexpr complex(const T& re = T(), const T& im = T()) : real(re), imag(im) {}

    constexpr complex(const complex& rhs) : real(rhs.real), imag(rhs.imag) {}

    complex& operator=(const T& rhs) noexcept {
        real = rhs;
        imag = 0.0;

        return *this;
    }

    complex& operator=(const complex& rhs) noexcept {
        real = rhs.real;
        imag = rhs.imag;

        return *this;
    }

    complex& operator=(const std::complex<T>& rhs) noexcept {
        real = rhs.real();
        imag = rhs.imag();

        return *this;
    }

    void operator+=(const complex& rhs){
        real += rhs.real;
        imag += rhs.imag;
    }

    void operator-=(const complex& rhs){
        real -= rhs.real;
        imag -= rhs.imag;
    }

    void operator*=(const complex& rhs){
        auto ac = real * rhs.real;
        auto bd = imag * rhs.imag;

        auto bc = imag * rhs.real;
        auto ad = real * rhs.imag;

        real = ac - bd;
        imag = bc + ad;
    }

    void operator/=(const complex& rhs){
        auto ac = real * rhs.real;
        auto bd = imag * rhs.imag;

        auto bc = imag * rhs.real;
        auto ad = real * rhs.imag;

        auto frac = rhs.real * rhs.real + rhs.imag * rhs.imag;

        real = (ac + bd) / frac;
        imag = (bc - ad) / frac;
    }
};

template<typename T>
inline complex<T> operator-(const complex<T>& rhs){
    return {-rhs.real, -rhs.imag};
}

template<typename T>
inline complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs){
    return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

template<typename T>
inline complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs){
    return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

template<typename T>
inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs){
    auto ac = lhs.real * rhs.real;
    auto bd = lhs.imag * rhs.imag;

    auto bc = lhs.imag * rhs.real;
    auto ad = lhs.real * rhs.imag;

    return {ac - bd, bc + ad};
}

template<typename T>
inline complex<T> operator*(const complex<T>& lhs, T rhs){
    return {lhs.real * rhs, lhs.imag * rhs};
}

template<typename T>
inline complex<T> operator*(T lhs, const complex<T>& rhs){
    return {lhs * rhs.real, lhs * rhs.imag};
}

template<typename T>
inline complex<T> operator/(const complex<T>& lhs, T rhs){
    return {lhs.real / rhs, lhs.imag / rhs};
}

template<typename T>
inline complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs){
    auto ac = lhs.real * rhs.real;
    auto bd = lhs.imag * rhs.imag;

    auto bc = lhs.imag * rhs.real;
    auto ad = lhs.real * rhs.imag;

    auto frac = rhs.real * rhs.real + rhs.imag * rhs.imag;

    return {(ac + bd) / frac, (bc - ad) / frac};
}

//inverse
template<typename T>
inline complex<T> inverse(complex<T> x){
    return {x.imag, x.real};
}

//inverse of the conj
template<typename T>
inline complex<T> inverse_conj(complex<T> x){
    return {-x.imag, x.real};
}

//conj of the inverse
template<typename T>
inline complex<T> conj_inverse(complex<T> x){
    return {x.imag, -x.real};
}

template<typename T>
inline complex<T> conj(const complex<T>& rhs){
    return {rhs.real, -rhs.imag};
}

} //end of namespace etl
