//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief A deserializer for ETL expressions
 */
template <typename Stream>
struct deserializer {
    using stream_t = Stream;                       ///< The type of stream to use
    using char_t   = typename stream_t::char_type; ///< The char type of the stream

    stream_t stream; ///< The stream

    /*!
     * \brief Construct the deserializer by forwarding the arguments
     * to the stream
     * \param args The arguments to forward to the stream constructor
     */
    template <typename... Args>
    explicit deserializer(Args&&... args)
            : stream(std::forward<Args>(args)...) {}

    /*!
     * \brief Reads a value of the given type from the stream
     * \param value Reference to the value where to write
     * \return the deserializer
     */
    template <typename T>
    deserializer& operator>>(T& value) {
        if constexpr (std::is_arithmetic<T>::value) {
            stream.read(reinterpret_cast<char_t*>(&value), sizeof(T));
        } else {
            deserialize(*this, value);
        }

        return *this;
    }
};

} //end of namespace etl
