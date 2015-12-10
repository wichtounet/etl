//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Define traits to get vectorization information for types when no vector mode is available.
 */
template <typename T>
struct no_intrinsic_traits {
    static constexpr const bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr const std::size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr const std::size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T;
};

struct no_vec {
    template<typename T>
    using traits = no_intrinsic_traits<T>;

    template<typename T>
    using vec_type = typename traits<T>::intrinsic_type;

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
