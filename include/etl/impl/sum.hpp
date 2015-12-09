//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the "sum" reduction implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available. The selection of
 * parallel versus serial is also done at this level. The implementation
 * functions should never be used directly, only functions of this header can
 * be used directly.
 */

#pragma once

#include "etl/config.hpp"
#include "etl/traits_lite.hpp"

//Include the implementations
#include "etl/impl/std/sum.hpp"

namespace etl {

namespace detail {

enum class sum_imple {
    STD,
    SSE,
    AVX
};

template <typename E>
cpp14_constexpr sum_imple select_sum_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if(!decay_traits<E>::vectorizable){
        return sum_imple::STD;
    }

    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if (avx) {
        return sum_imple::AVX;
    } else if (sse) {
        return sum_imple::SSE;
    } else {
        return sum_imple::STD;
    }
}

template <typename E, typename Enable = void>
struct sum_impl {
    static auto apply(const E& e) {
        cpp14_constexpr auto impl = select_sum_impl<E>();

        if (impl == sum_imple::AVX) {
            return impl::standard::sum(e);
        } else if (impl == sum_imple::SSE) {
            return impl::standard::sum(e);
        } else {
            return impl::standard::sum(e);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
