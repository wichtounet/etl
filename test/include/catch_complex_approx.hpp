//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the ComplexApprox class to compare complex numbers with a margin of error
 */

#pragma once

#include <cmath>

/*!
 * \brief Utility class to compare two complex numbers with a margin of error
 */
template <typename T>
struct ComplexApprox {
    /*!
     * \brief Construct a ComplexApprox for the given complex value
     * \param value the expected complex value
     */
    explicit ComplexApprox(const std::complex<T>& value)
            : eps(std::numeric_limits<float>::epsilon() * 100), value(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a ComplexApprox for the given complex value
     * \param real the expected real part
     * \param imag the expected imaginary part
     */
    ComplexApprox(T real, T imag)
            : eps(std::numeric_limits<float>::epsilon() * 100), value(real, imag) {
        //Nothing else to init
    }

    ComplexApprox(const ComplexApprox& other) = default;

    /*!
     * \brief Compare a complex number with an expected value
     * \param lhs The complex number (the number to test)
     * \param rhs The expected complex number
     * \return true if they are approximatily the same
     */
    friend bool operator==(const std::complex<T>& lhs, const ComplexApprox& rhs) {
        return fabs(lhs.real() - rhs.value.real()) < rhs.eps * (1.0 + std::max(fabs(lhs.real()), fabs(rhs.value.real()))) && fabs(lhs.imag() - rhs.value.imag()) < rhs.eps * (1.0 + std::max(fabs(lhs.imag()), fabs(rhs.value.imag())));
    }

    /*!
     * \brief Compare a complex number with an expected value
     * \param lhs The expected complex number
     * \param rhs The complex number (the number to test)
     * \return true if they are approximatily the same
     */
    friend bool operator==(const ComplexApprox& lhs, const std::complex<T>& rhs) {
        return operator==(rhs, lhs);
    }

    /*!
     * \brief Compare a complex number with an expected value for inequality
     * \param lhs The complex number (the number to test)
     * \param rhs The expected complex number
     * \return true if they are not approximatily the same
     */
    friend bool operator!=(const std::complex<T>& lhs, const ComplexApprox& rhs) {
        return !operator==(lhs, rhs);
    }

    /*!
     * \brief Compare a complex number with an expected value for inequality
     * \param lhs The expected complex number
     * \param rhs The complex number (the number to test)
     * \return true if they are not approximatily the same
     */
    friend bool operator!=(const ComplexApprox& lhs, const std::complex<T>& rhs) {
        return !operator==(rhs, lhs);
    }

    /*!
     * \brief Returns a textual representation of the operand for Catch
     * \return a std::string representing this operand
     */
    std::string toString() const {
        std::ostringstream oss;
        oss << "ComplexApprox(" << value << ")";
        return oss.str();
    }

private:
    double eps;            ///< The epsilon for comparison
    std::complex<T> value; ///< The expected value
};
