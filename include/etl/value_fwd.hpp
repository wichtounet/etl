//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_VALUE_FWD_HPP
#define ETL_VALUE_FWD_HPP

namespace etl {

template<typename T, typename ST, std::size_t... Dims>
struct fast_matrix_impl;

template<typename T, std::size_t... Dims>
using fast_matrix = fast_matrix_impl<T, std::array<T, mul_all<Dims...>::value>, Dims...>;

template<typename T, std::size_t Rows>
using fast_dyn_vector = fast_matrix_impl<T, std::vector<T>, Rows>;

template<typename T, std::size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, std::vector<T>, Dims...>;

template<typename T, std::size_t D = 2>
struct dyn_matrix;

template<typename T>
using dyn_vector = dyn_matrix<T, 1>;

}

#endif
