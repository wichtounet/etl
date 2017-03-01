//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<float>* safe_cast(std::complex<float>* in){
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<double>* safe_cast(std::complex<double>* in){
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<float>* safe_cast(etl::complex<float>* in){
    return reinterpret_cast<std::complex<float>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline std::complex<double>* safe_cast(etl::complex<double>* in){
    return reinterpret_cast<std::complex<double>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<float>* safe_cast(const std::complex<float>* in){
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<double>* safe_cast(const std::complex<double>* in){
    return in;
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<float>* safe_cast(const etl::complex<float>* in){
    return reinterpret_cast<const std::complex<float>*>(in);
}

/*!
 * \brief Cast any complex pointer to etl::complex pointer
 * \param in The input pointer
 * \return The casted pointer
 */
inline const std::complex<double>* safe_cast(const etl::complex<double>* in){
    return reinterpret_cast<const std::complex<double>*>(in);
}

} //end of namespace impl

} //end of namespace etl
