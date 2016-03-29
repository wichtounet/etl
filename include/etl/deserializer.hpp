//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template <typename Stream>
struct deserializer {
    using stream_t = Stream;
    using char_t   = typename stream_t::char_type;

    stream_t stream;

    template <typename... Args>
    explicit deserializer(Args&&... args)
            : stream(std::forward<Args>(args)...) {}

    template <typename T, cpp_enable_if(std::is_arithmetic<T>::value)>
    deserializer& operator>>(T& value) {
        stream.read(reinterpret_cast<char_t*>(&value), sizeof(T));
        return *this;
    }

    template <typename T, cpp_disable_if(std::is_arithmetic<T>::value)>
    deserializer& operator>>(T& value) {
        deserialize(*this, value);
        return *this;
    }
};

} //end of namespace etl
