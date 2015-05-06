//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DYN_MATRIX_HPP
#define ETL_DYN_MATRIX_HPP

namespace etl {

template<typename T, std::size_t D>
using dyn_matrix = dyn_matrix_impl<T, D>;

} //end of namespace etl

#endif
