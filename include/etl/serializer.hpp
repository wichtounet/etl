//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief A serializer for ETL expressions
 */
template <typename Stream>
struct serializer {
    using stream_t = Stream;                       ///< The type of stream to use
    using char_t   = typename stream_t::char_type; ///< The char type of the stream

    stream_t stream; ///< The stream

    /*!
     * \brief Construct the serializer by forwarding the arguments
     * to the stream
     * \param args The arguments to forward to the stream constructor
     */
    template <typename... Args>
    explicit serializer(Args&&... args)
            : stream(std::forward<Args>(args)...) {}

    /*!
     * \brief Outputs the given value to the stream
     * \param value The value to write to the stream
     * \return the serializer
     */
    template <typename T, cpp_enable_iff(std::is_arithmetic<T>::value)>
    serializer& operator<<(const T& value) {
        stream.write(reinterpret_cast<const char_t*>(&value), sizeof(T));
        return *this;
    }

    /*!
     * \brief Outputs the given ETL expression to the stream
     * \param value The ETL expression to write to the stream
     * \return the serializer
     */
    template <typename T, cpp_disable_iff(std::is_arithmetic<T>::value)>
    serializer& operator<<(const T& value) {
        serialize(*this, value);
        return *this;
    }
};

} //end of namespace etl
