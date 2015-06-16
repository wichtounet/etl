//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cmath>

#include "cpp_utils/tmp.hpp"

namespace etl {

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr W logistic_sigmoid(W x){
    return W(1.0) / (W(1.0) + std::exp(-x));
}

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr W softplus(W x){
    return std::log(W(1.0) + std::exp(x));
}

template<typename W, cpp::enable_if_u<std::is_arithmetic<W>::value> = cpp::detail::dummy>
inline constexpr double sign(W v) noexcept {
    return v == W(0) ? W(0) : (v > W(0) ? W(1) : W(-1));
}


} //end of namespace etl
