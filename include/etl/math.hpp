//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MATH_HPP
#define ETL_MATH_HPP

#include <cmath>

#include "cpp_utils/tmp.hpp"

namespace etl {

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr W logistic_sigmoid(W x){
    return 1.0 / (1.0 + std::exp(-x));
}

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr W softplus(W x){
    return std::log(1.0 + std::exp(x));
}

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr double sign(W v){
    return v == 0 ? 0 : (v > 0 ? 1 : -1);
}


} //end of namespace etl

#endif
