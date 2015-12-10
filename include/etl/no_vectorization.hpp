//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

struct no_vec {
    template <typename F, typename M>
    static inline void storeu(F* /*memory*/, M /*value*/) {}

    template <typename F, typename M>
    static inline void store(F* /*memory*/, M /*value*/) {}

    template <typename F>
    static F load(const F* /*memory*/) {}

    template <typename F>
    static F loadu(const F* /*memory*/) {}

    template <typename F>
    static F set(F /*value*/) {}

    template <typename M>
    static M add(M /*lhs*/, M /*rhs*/) {}

    template <typename M>
    static M sub(M /*lhs*/, M /*rhs*/) {}

    template <typename M>
    static M mul(M /*lhs*/, M /*rhs*/) {}

    template <typename M>
    static M div(M /*lhs*/, M /*rhs*/) {}

    template <typename M>
    static M sqrt(M /*value*/) {}

    template <typename M>
    static M minus(M /*value*/) {}
};

} //end of namespace etl
