//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_DYN_MATRIX_HPP
#define ETL_FAST_DYN_MATRIX_HPP

#include<vector>

#include "fast.hpp"

namespace etl {

template<typename T, std::size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, std::vector<T>, Dims...>;

} //end of namespace etl

#endif
