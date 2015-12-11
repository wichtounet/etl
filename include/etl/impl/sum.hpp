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
 *
 * Note: In a perfect world (full constexpr function and variable templates),
 * the selection should be done with a template parameter in a variable
 * template full sspecialization (alias for each real functions).
 */

#pragma once

#include "etl/config.hpp"

//Include the implementations
#include "etl/impl/std/sum.hpp"
#include "etl/impl/sse/sum.hpp"
#include "etl/impl/avx/sum.hpp"

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

    if(decay_traits<E>::vectorizable){
        constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
        constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

        if (avx) {
            return sum_imple::AVX;
        } else if (sse) {
            return sum_imple::SSE;
        }
    }

    return sum_imple::STD;
}

template <typename E>
struct sum_impl {
    static value_t<E> apply(const E& e) {
        cpp14_constexpr auto impl = select_sum_impl<E>();

        return selected_apply(e, impl);
    }

    static value_t<E> selected_apply(const E& e, sum_imple impl) {
        if (impl == sum_imple::AVX) {
            return impl::avx::sum(e);
        } else if (impl == sum_imple::SSE) {
            return impl::sse::sum(e);
        } else {
            return impl::standard::sum(e);
        }
    }
};

template <typename E>
auto sum_direct(const E& e, sum_imple impl){
    return sum_impl<E>::selected_apply(e, impl);
}

} //end of namespace detail

} //end of namespace etl
