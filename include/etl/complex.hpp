//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief binary-compatible std::complex faster implementation
 *
 * For some reason, the compilers are not inlining std::complex
 * operations, resulting in way slower code. This template is
 * binary-compatible so it can be reinterpreted as to one another
 * and its operations can be inlined easily. This results in much
 * faster code for FFT for instance.
 */

#pragma once

namespace etl {

/*!
 * \brief Complex number implementation
 *
 * This implementation is binary-compatible with std::complex. This
 * implementation is not opaque like std::complex and its operation can be
 * inlined.
 */
template <typename T>
struct complex {
    using value_type = T; ///< The value type

    value_type real; ///< The real part
    value_type imag; ///< The imaginary part

    /*!
     * \brief Construct a complex number
     * \param re The real part
     * \param im The imaginary part
     */
    constexpr complex(const T& re = T(), const T& im = T()) : real(re), imag(im) {
        // Nothing else to init
    }

    /*!
     * \brief Construct a complex number by copy
     * \param rhs The complex to copy from
     */
    constexpr complex(const complex& rhs) = default;

    /*!
     * \brief Construct a complex number by copy from another complex with
     * different inner type.
     *
     * \param rhs The complex to copy from
     */
    template <typename X>
    constexpr complex(const complex<X>& rhs) : real(rhs.real), imag(rhs.imag) {
        // Nothing else to init
    }

    /*!
     * \brief Assign a real part to the complex number
     * \param rhs The real part
     * \return a reference to this
     */
    complex& operator=(const T& rhs) noexcept {
        real = rhs;
        imag = 0.0;

        return *this;
    }

    /*!
     * \brief Assign a complex number by copy
     * \param rhs The complex number to copy
     * \return a reference to this
     */
    complex& operator=(const complex& rhs) noexcept = default;

    /*!
     * \brief Assign a complex number by copy
     * \param rhs The complex number to copy
     * \return a reference to this
     */
    complex& operator=(const std::complex<T>& rhs) noexcept {
        real = rhs.real();
        imag = rhs.imag();

        return *this;
    }

    /*!
     * \brief Adds a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    template <typename X>
    complex& operator+=(const complex<X>& rhs) {
        real += rhs.real;
        imag += rhs.imag;

        return *this;
    }

    /*!
     * \brief Subtracts a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    template <typename X>
    complex& operator-=(const complex<X>& rhs) {
        real -= rhs.real;
        imag -= rhs.imag;

        return *this;
    }

    /*!
     * \brief Multipliies a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    template <typename X>
    complex& operator*=(const complex<X>& rhs) {
        T ac = real * rhs.real;
        T bd = imag * rhs.imag;

        T bc = imag * rhs.real;
        T ad = real * rhs.imag;

        real = ac - bd;
        imag = bc + ad;

        return *this;
    }

    /*!
     * \brief Divides a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    template <typename X>
    complex& operator/=(const complex<X>& rhs) {
        T ac = real * rhs.real;
        T bd = imag * rhs.imag;

        T bc = imag * rhs.real;
        T ad = real * rhs.imag;

        T frac = rhs.real * rhs.real + rhs.imag * rhs.imag;

        real = (ac + bd) / frac;
        imag = (bc - ad) / frac;

        return *this;
    }
};

/*!
 * \brief Test two complex numbers for equality
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return true if the numbers are equals, false otherwise
 */
template <typename T>
inline bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
    return lhs.real == rhs.real && lhs.imag == rhs.imag;
}

/*!
 * \brief Test two complex numbers for inequality
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return true if the numbers are not equals, false otherwise
 */
template <typename T>
inline bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
    return !(lhs == rhs);
}

/*!
 * \brief Returns a complex number with the value of -rhs
 * \param rhs The right hand side complex
 * \return a complex number with the value of -rhs
 */
template <typename T>
inline complex<T> operator-(complex<T> rhs) {
    return {-rhs.real, -rhs.imag};
}

/*!
 * \brief Computes the addition of two complex numbers
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the addition of the two complex numbers
 */
template <typename T>
inline complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
    return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

/*!
 * \brief Computes the subtraction of two complex numbers
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the subtraction of the two complex numbers
 */
template <typename T>
inline complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
    return {lhs.real - rhs.real, lhs.imag - rhs.imag};
}

/*!
 * \brief Computes the multiplication of two complex numbers
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the multiplication of the two complex numbers
 */
template <typename T>
inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
    T ac = lhs.real * rhs.real;
    T bd = lhs.imag * rhs.imag;

    T bc = lhs.imag * rhs.real;
    T ad = lhs.real * rhs.imag;

    return {ac - bd, bc + ad};
}

/*!
 * \brief Computes the multiplication of a complex number and a scalar
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the multiplication of a complex number and a scalar
 */
template <typename T>
inline complex<T> operator*(const complex<T>& lhs, T rhs) {
    return {lhs.real * rhs, lhs.imag * rhs};
}

/*!
 * \brief Computes the multiplication of a complex number and a scalar
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the multiplication of a complex number and a scalar
 */
template <typename T>
inline complex<T> operator*(T lhs, const complex<T>& rhs) {
    return {lhs * rhs.real, lhs * rhs.imag};
}

/*!
 * \brief Computes the division of a complex number and a scalar
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the multiplication of a complex number and a scalar
 */
template <typename T>
inline complex<T> operator/(const complex<T>& lhs, T rhs) {
    return {lhs.real / rhs, lhs.imag / rhs};
}

/*!
 * \brief Computes the division of two complex numbers
 * \param lhs The left hand side complex
 * \param rhs The right hand side complex
 * \return a new complex with the value of the division of the two complex numbers
 */
template <typename T>
inline complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
    T ac = lhs.real * rhs.real;
    T bd = lhs.imag * rhs.imag;

    T bc = lhs.imag * rhs.real;
    T ad = lhs.real * rhs.imag;

    T frac = rhs.real * rhs.real + rhs.imag * rhs.imag;

    return {(ac + bd) / frac, (bc - ad) / frac};
}

/*!
 * \brief Computes the magnitude of the given complex number
 * \param z The input complex number
 * \return The magnitude of z
 */
template <typename T>
T abs(complex<T> z) {
    auto x = z.real;
    auto y = z.imag;
    auto s = std::max(std::abs(x), std::abs(y));

    if (s == T()) {
        return s;
    }

    x = x / s;
    y = y / s;

    return s * std::sqrt(x * x + y * y);
}

/*!
 * \brief Computes the phase angle of the given complex number
 * \param z The input complex number
 * \return The phase angle of z
 */
template <typename T>
T arg(complex<T> z) {
    auto x = z.real;
    auto y = z.imag;

    return atan2(y, x);
}

/*!
 * \brief Computes the complex square root of the input
 * \param z The input complex number
 * \return The square root of z
 */
template <typename T>
complex<T> sqrt(complex<T> z) {
    auto x = z.real;
    auto y = z.imag;

    if (x == T()) {
        auto t = std::sqrt(std::abs(y) / 2);
        return {t, y < T() ? -t : t};
    } else {
        auto t = std::sqrt(2 * (abs(z) + std::abs(x)));
        auto u = t / 2;

        if (x > T()) {
            return {u, y / t};
        } else {
            return {std::abs(y) / t, y < T() ? -u : u};
        }
    }
}

/*!
 * \brief Computes the inverse complex square root of the input
 * \param z The input complex number
 * \return The inverse square root of z
 */
template <typename T>
complex<T> invsqrt(complex<T> z) {
    return complex<T>(T(1)) / sqrt(z);
}

/*!
 * \brief Computes the complex cubic root of the input
 * \param z The input complex number
 * \return The cubic root of z
 */
template <typename T>
complex<T> cbrt(complex<T> z) {
    auto z_abs = etl::abs(z);
    auto z_arg = etl::arg(z);

    auto new_abs = std::cbrt(z_abs);
    auto new_arg = z_arg / 3.0f;

    return {new_abs * std::cos(new_arg), new_abs * std::sin(new_arg)};
}

/*!
 * \brief Computes the inverse complex cubic root of the input
 * \param z The input complex number
 * \return The inverse cubic root of z
 */
template <typename T>
complex<T> invcbrt(complex<T> z) {
    return complex<T>(T(1)) / cbrt(z);
}

/*!
 * \brief Computes the complex logarithm, in base e, of the input
 * \param z The input complex number
 * \return The complex logarithm, in base e, of z
 */
template <typename T>
complex<T> log(complex<T> z) {
    return {std::log(etl::abs(z)), etl::arg(z)};
}

/*!
 * \brief Computes the complex logarithm, in base 2, of the input
 * \param z The input complex number
 * \return The complex logarithm, in base 2, of z
 */
template <typename T>
complex<T> log2(complex<T> z) {
    return etl::log(z) / etl::log(etl::complex<T>{T(2)});
}

/*!
 * \brief Computes the complex logarithm, in base 10, of the input
 * \param z The input complex number
 * \return The complex logarithm, in base 10, of z
 */
template <typename T>
complex<T> log10(complex<T> z) {
    return etl::log(z) / etl::log(etl::complex<T>{T(10)});
}

/*!
 * \brief Computes the sinus of the complex input
 * \param z The input complex number
 * \return The sinus of z
 */
template <typename T>
complex<T> sin(complex<T> z) {
    return {std::sin(z.real) * std::cosh(z.imag), std::cos(z.real) * std::sinh(z.imag)};
}

/*!
 * \brief Computes the cosine of the complex input
 * \param z The input complex number
 * \return The cosine of z
 */
template <typename T>
complex<T> cos(complex<T> z) {
    return {std::cos(z.real) * std::cosh(z.imag), -std::sin(z.real) * std::sinh(z.imag)};
}

/*!
 * \brief Computes the tangent of the complex input
 * \param z The input complex number
 * \return The tangent of z
 */
template <typename T>
complex<T> tan(complex<T> z) {
    return sin(z) / cos(z);
}

/*!
 * \brief Computes the hyperbolic cosine of the complex input
 * \param z The input complex number
 * \return The hyperbolic cosine of z
 */
template <typename T>
complex<T> cosh(complex<T> z) {
    return {std::cosh(z.real) * std::cos(z.imag), std::sinh(z.real) * std::sin(z.imag)};
}

/*!
 * \brief Computes the hyperbolic sinus of the complex input
 * \param z The input complex number
 * \return The hyperbolic sinus of z
 */
template <typename T>
complex<T> sinh(complex<T> z) {
    return {std::sinh(z.real) * std::cos(z.imag), std::cosh(z.real) * std::sin(z.imag)};
}

/*!
 * \brief Computes the hyperbolic tangent of the complex input
 * \param z The input complex number
 * \return The hyperbolic tangent of z
 */
template <typename T>
complex<T> tanh(complex<T> z) {
    return sinh(z) / cosh(z);
}

/*!
 * \brief Returns the inverse of the complex number
 * \param x The complex number
 * \return The inverse of the complex number
 */
template <typename T>
inline complex<T> inverse(complex<T> x) {
    return {x.imag, x.real};
}

/*!
 * \brief Returns the inverse of the conjugate of the complex number
 * \param x The complex number
 * \return The inverse of the conjugate of the complex number
 */
template <typename T>
inline complex<T> inverse_conj(complex<T> x) {
    return {-x.imag, x.real};
}

/*!
 * \brief Returns the conjugate of the inverse of the complex number
 * \param x The complex number
 * \return The conjugate of the inverse of the complex number
 */
template <typename T>
inline complex<T> conj_inverse(complex<T> x) {
    return {x.imag, -x.real};
}

/*!
 * \brief Returns the conjugate of the complex number
 * \param c The complex number
 * \return The conjugate of the complex number
 */
template <typename T>
inline complex<T> conj(const complex<T>& c) {
    return {c.real, -c.imag};
}

/*!
 * \brief Returns the imaginary part of the given complex number
 * \param c The complex number
 * \return the imaginary part of the given complex number
 */
template <typename T>
inline T get_imag(const std::complex<T>& c) {
    return c.imag();
}

/*!
 * \brief Returns the imaginary part of the given complex number
 * \param c The complex number
 * \return the imaginary part of the given complex number
 */
template <typename T>
inline T get_imag(const etl::complex<T>& c) {
    return c.imag;
}

/*!
 * \brief Returns the real part of the given complex number
 * \param c The complex number
 * \return the real part of the given complex number
 */
template <typename T>
inline T get_real(const std::complex<T>& c) {
    return c.real();
}

/*!
 * \brief Returns the real part of the given complex number
 * \param c The complex number
 * \return the real part of the given complex number
 */
template <typename T>
inline T get_real(const etl::complex<T>& c) {
    return c.real;
}

/*!
 * \brief Returns the conjugate of the given complex number
 * \param c The complex number
 * \return the conjugate of the given complex number
 */
template <typename T>
inline std::complex<T> get_conj(const std::complex<T>& c) {
    return std::conj(c);
}

/*!
 * \brief Returns the conjugate of the given complex number
 * \param c The complex number
 * \return the conjugate of the given complex number
 */
template <typename T>
inline etl::complex<T> get_conj(const etl::complex<T>& c) {
    return {c.real, -c.imag};
}

/*!
 * \brief Outputs a textual representation of the complex number in the given stream
 * \param os The stream to output to
 * \param c The complex number to get representation from
 * \return The output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const etl::complex<T>& c) {
    return os << "C(" << c.real << "," << c.imag << ")";
}

} //end of namespace etl
