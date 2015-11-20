//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file globals.hpp
 * \brief Contains some global functions.
*/

//TODO Think of reverting this and put the implementation here and not in dim_testable CRTP class

#pragma once

namespace etl {

template<typename E>
bool is_square(E&& expr) {
    return expr.is_square();
}

template<typename E>
bool is_rectangular(E&& expr) {
    return expr.is_rectangular();
}

template<typename E>
bool is_sub_square(E&& expr) {
    return expr.is_sub_square();
}

template<typename E>
bool is_sub_rectangular(E&& expr) {
    return expr.is_sub_rectangular();
}

} //end of namespace etl
