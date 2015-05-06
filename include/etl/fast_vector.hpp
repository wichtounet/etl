//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_VECTOR_HPP
#define ETL_FAST_VECTOR_HPP

namespace etl {

template<typename T, std::size_t Rows>
using fast_vector = fast_matrix_impl<T, std::array<T, Rows>, Rows>;

} //end of namespace etl

#endif
