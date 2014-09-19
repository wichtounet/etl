//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DYN_VECTOR_HPP
#define ETL_DYN_VECTOR_HPP

#include "dyn_matrix.hpp"

namespace etl {

template<typename T>
using dyn_vector = dyn_matrix<T, 1>;

} //end of namespace etl

#endif
