//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace vec {

template <typename F, typename M>
inline void storeu(F* /*memory*/, M /*value*/) {}

template <typename F, typename M>
inline void store(F* /*memory*/, M /*value*/) {}

template <typename F>
F load(const F* /*memory*/) {}

template <typename F>
F loadu(const F* /*memory*/) {}

template <typename F>
F set(F /*value*/) {}

template <typename M>
M add(M /*lhs*/, M /*rhs*/) {}

template <typename M>
M sub(M /*lhs*/, M /*rhs*/) {}

template <typename M>
M mul(M /*lhs*/, M /*rhs*/) {}

template <typename M>
M div(M /*lhs*/, M /*rhs*/) {}

template <typename M>
M sqrt(M /*value*/) {}

template <typename M>
M minus(M /*value*/) {}

} //end of namespace vec

} //end of namespace etl
