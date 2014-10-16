//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TMP_HPP
#define ETL_TMP_HPP

namespace etl {

template<typename E>
using value_t = typename std::decay_t<E>::value_type;

template<size_t F, size_t... Dims>
struct mul_all  : std::integral_constant<std::size_t, F * mul_all<Dims...>::value> {};

template<size_t F>
struct mul_all<F> : std::integral_constant<std::size_t, F> {};

} //end of namespace etl

#endif