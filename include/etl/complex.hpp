//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
    constexpr complex(const T& re = T(), const T& im = T())
            : real(re), imag(im) {}

    /*!
     * \brief Construct a complex number by copy
     * \param rhs The complex to copy from
     */
    constexpr complex(const complex& rhs)
            : real(rhs.real), imag(rhs.imag) {}

    /*!
     * \brief Assign a real part to the complex number
     * \param rhs The real part
     * \return a reference to this
     */
    complex& operator=(const T& rhs) noexcept {
        real         = rhs;
        imag         = 0.0;

        return *this;
    }

    /*!
     * \brief Assign a complex number by copy
     * \param rhs The complex number to copy
     * \return a reference to this
     */
    complex& operator=(const complex& rhs) noexcept {
        real         = rhs.real;
        imag         = rhs.imag;

        return *this;
    }

    /*!
     * \brief Assign a complex number by copy
     * \param rhs The complex number to copy
     * \return a reference to this
     */
    complex& operator=(const std::complex<T>& rhs) noexcept {
        real         = rhs.real();
        imag         = rhs.imag();

        return *this;
    }

    /*!
     * \brief Adds a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    complex& operator+=(const complex& rhs) {
        real += rhs.real;
        imag += rhs.imag;

        return *this;
    }

    /*!
     * \brief Subtracts a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    complex& operator-=(const complex& rhs) {
        real -= rhs.real;
        imag -= rhs.imag;

        return *this;
    }

    /*!
     * \brief Multipliies a complex number
     * \param rhs The complex number to add
     * \return a reference to this
     */
    complex& operator*=(const complex& rhs) {
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
    complex& operator/=(const complex& rhs) {
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
 * \param return The output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const etl::complex<T>& c){
    return os << "C(" << c.real << "," << c.imag << ")";
}

} //end of namespace etl
