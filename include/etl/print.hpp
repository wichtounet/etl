//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_PRINT_HPP
#define ETL_PRINT_HPP

#include<string>

#include "traits.hpp"

namespace etl {

template<typename T, enable_if_u<etl_traits<T>::is_value> = detail::dummy>
std::ostream& operator<<(std::ostream& stream, const T& v){
    return stream << to_string(v);
}

template<typename T, enable_if_u<etl_traits<T>::is_vector> = detail::dummy>
std::string to_string(const T& vec){
    return to_octave(vec);
}

template<typename T, enable_if_u<and_u<etl_traits<T>::is_matrix, not_u<etl_traits<T>::is_fast>::value>::value> = detail::dummy>
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

template<typename T, enable_if_all_u<etl_traits<T>::is_matrix, not_u<etl_traits<T>::is_fast>::value> = detail::dummy>
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

//"Fast" versions

template<typename T, enable_if_u<and_u<etl_traits<T>::is_matrix, etl_traits<T>::is_fast, (etl_traits<T>::dimensions() > 2)>::value> = detail::dummy>
std::string to_string(const T& m){
    std::string v = "[";
    for(std::size_t i = 0; i < etl_traits<T>::template dim<0>(); ++i){
        v += to_string(sub(m, i));

        if(i < etl_traits<T>::template dim<0>() - 1){
            v += "\n";
        }
    }
    v += "]";
    return v;
}

template<typename T, enable_if_u<and_u<etl_traits<T>::is_matrix, etl_traits<T>::is_fast, etl_traits<T>::dimensions() == 2>::value> = detail::dummy>
std::string to_string(const T& m){
    std::string v = "[";
    for(std::size_t i = 0; i < etl_traits<T>::template dim<0>(); ++i){
        v += "[";
        std::string comma = "";
        for(std::size_t j = 0; j  < etl_traits<T>::template dim<1>(); ++j){
            v += comma + std::to_string(m(i, j));
            comma = ",";
        }
        v += "]";
        if(i < etl_traits<T>::template dim<0>() - 1){
            v += "\n";
        }
    }
    v += "]";
    return v;
}

template<bool Sub = false, typename T, enable_if_u<and_u<etl_traits<T>::is_matrix, etl_traits<T>::is_fast, (etl_traits<T>::dimensions() > 1)>::value> = detail::dummy>
std::string to_octave(const T& m){
    std::string v;
    if(!Sub){
        v = "[";
    }

    for(std::size_t i = 0; i < etl_traits<T>::template dim<0>(); ++i){
        v += to_octave<true>(sub(m, i));

        if(i < etl_traits<T>::template dim<0>() - 1){
            v += ";";
        }
    }

    if(!Sub){
        v += "]";
    }

    return v;
}

template<bool Sub = true, typename T, enable_if_u<and_u<etl_traits<T>::is_matrix, etl_traits<T>::is_fast, etl_traits<T>::dimensions() == 1>::value> = detail::dummy>
std::string to_octave(const T& m){
    std::string v;
    if(!Sub){
        v = "[";
    }

    std::string comma = "";
    for(std::size_t j = 0; j  < etl_traits<T>::template dim<0>(); ++j){
        v += comma + std::to_string(m(j));
        comma = ",";
    }

    if(!Sub){
        v += "]";
    }

    return v;
}

} //end of namespace etl

#endif
