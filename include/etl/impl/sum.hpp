//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

    if(decay_traits<E>::template vectorizable<vector_mode>::value){
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
inline bool select_parallel(const E& e) {
    if((parallel && !local_context().serial) || local_context().parallel){
        return size(e) >= sum_parallel_threshold;
    } else {
        return false;
    }
}

template <typename E>
struct sum_impl {
    static value_t<E> apply(const E& e) {
        cpp14_constexpr auto impl = select_sum_impl<E>();

        return selected_apply(e, impl);
    }

    static value_t<E> selected_apply(const E& e, sum_imple impl) {
        bool parallel_dispatch = select_parallel(e);

        value_t<E> acc(0);

        auto acc_functor = [&acc](value_t<E> value){
            acc += value;
        };

        if (impl == sum_imple::AVX) {
            dispatch_1d_acc<value_t<E>>(parallel_dispatch, [&e](std::size_t first, std::size_t last) -> value_t<E> {
                return impl::avx::sum(e, first, last);
            }, acc_functor, 0, size(e));
        } else if (impl == sum_imple::SSE) {
            dispatch_1d_acc<value_t<E>>(parallel_dispatch, [&e](std::size_t first, std::size_t last) -> value_t<E> {
                return impl::sse::sum(e, first, last);
            }, acc_functor, 0, size(e));
        } else {
            dispatch_1d_acc<value_t<E>>(parallel_dispatch, [&e](std::size_t first, std::size_t last) -> value_t<E> {
                return impl::standard::sum(e, first, last);
            }, acc_functor, 0, size(e));
        }

        return acc;
    }
};

template <typename E>
auto sum_direct(const E& e, sum_imple impl){
    return sum_impl<E>::selected_apply(e, impl);
}

} //end of namespace detail

} //end of namespace etl
