//=======================================================================
// Copyright (c) 2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <concepts>
#include <type_traits>

namespace etl {

template<typename T>
concept etl_expr = decay_traits<T>::is_etl;

template<typename T>
struct scalar;

template <typename T, typename VT>
concept expr_or_scalar = etl_expr<T> || std::same_as<T, scalar<VT>>;

} // namespace etl
