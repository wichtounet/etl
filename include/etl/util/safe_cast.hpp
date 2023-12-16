//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brie Contains the safe_cast function overloads.
 *
 * This function helps generic code in the BLAS/CUBLAS wrappers to
 * be able to convert types from etl::complex to std::complex
 * easily.
 */

#pragma once

namespace etl::impl {

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline float* safe_cast(float* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline double* safe_cast(double* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<float>* safe_cast(std::complex<float>* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<double>* safe_cast(std::complex<double>* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<float>* safe_cast(etl::complex<float>* in) {
    return reinterpret_cast<std::complex<float>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<double>* safe_cast(etl::complex<double>* in) {
    return reinterpret_cast<std::complex<double>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const float* safe_cast(const float* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const double* safe_cast(const double* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<float>* safe_cast(const std::complex<float>* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<double>* safe_cast(const std::complex<double>* in) {
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<float>* safe_cast(const etl::complex<float>* in) {
    return reinterpret_cast<const std::complex<float>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<double>* safe_cast(const etl::complex<double>* in) {
    return reinterpret_cast<const std::complex<double>*>(in);
}

} //end of namespace etl::impl
