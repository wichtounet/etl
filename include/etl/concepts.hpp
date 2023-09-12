//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <concepts>

namespace etl {

namespace concepts_detail {

} // end of namespace traits_detail

template<typename T>
concept etl_expr = etl::decay_traits<T>::is_etl;

} //end of namespace etl
