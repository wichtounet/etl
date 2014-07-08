//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_PRINT_HPP
#define ETL_PRINT_HPP

#include<string>

#include "tmp.hpp"

namespace etl {

template<typename T, enable_if_u<etl_traits<T>::is_value> = detail::dummy>
std::ostream& operator<<(std::ostream& stream, const T& v){
    return stream << to_string(v);
}

template<typename T, enable_if_u<etl_traits<T>::is_vector> = detail::dummy>
std::string to_string(const T& vec){
    return to_octave(vec);
}

template<typename T, enable_if_u<etl_traits<T>::is_vector> = detail::dummy>
std::string to_octave(const T& vec){
    std::string v = "[";
    std::string comma = "";
    for(std::size_t i = 0; i < size(vec); ++i){
        v += comma + std::to_string(vec(i));
        comma = ",";
    }
    v += "]";
    return v;
}

template<typename T, enable_if_u<etl_traits<T>::is_matrix> = detail::dummy>
std::string to_string(const T& m){
    std::string v = "[";
    for(std::size_t i = 0; i < rows(m); ++i){
        v += "[";
        std::string comma = "";
        for(std::size_t j = 0; j  < columns(m); ++j){
            v += comma + std::to_string(m(i, j));
            comma = ",";
        }
        v += "]";
        if(i < rows(m) - 1){
            v += "\n";
        }
    }
    v += "]";
    return v;
}

template<typename T, enable_if_u<etl_traits<T>::is_matrix> = detail::dummy>
std::string to_octave(const T& m){
    std::string v = "[";
    for(std::size_t i = 0; i < rows(m); ++i){
        std::string comma = "";
        for(std::size_t j = 0; j  < columns(m); ++j){
            v += comma + std::to_string(m(i, j));
            comma = ",";
        }
        if(i < rows(m) - 1){
            v += ";";
        }
    }
    v += "]";
    return v;
}

} //end of namespace etl

#endif
