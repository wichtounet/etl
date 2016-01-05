//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <thread>

#include "cpp_utils/parallel.hpp"

namespace etl {

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename Functor>
inline void dispatch_1d(bool p, Functor&& functor, std::size_t first, std::size_t last){
    if(p){
        cpp::default_thread_pool<> pool(threads - 1);

        auto n = last - first;
        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            pool.do_task(functor, first + t * batch, first + (t+1) * batch);
        }

        functor(first + (threads - 1) * batch, last);
    } else {
        functor(first, last);
    }
}

/*!
 * \brief Dispatch the elements of a range to a functor in a parallel manner and use an accumulator functor to accumulate the results
 * \param p Boolean tag to indicate if parallel dispatching must be done
 * \param functor The functor to execute
 * \param acc_functor The functor to accumulate results
 * \param first The beginning of the range
 * \param last The end of the range
 */
template <typename T, typename Functor, typename AccFunctor>
inline void dispatch_1d_acc(bool p, Functor&& functor, AccFunctor&& acc_functor, std::size_t first, std::size_t last){
    if(p){
        std::vector<std::future<T>> futures(threads - 1);
        cpp::default_thread_pool<> pool(threads - 1);

        auto n = last - first;
        auto batch = n / threads;

        for(std::size_t t = 0; t < threads - 1; ++t){
            futures[t] = std::async(std::launch::async, functor, first + t * batch, first + (t+1) * batch);
        }

        acc_functor(functor(first + (threads - 1) * batch, last));

        for(auto& fut : futures){
            acc_functor(fut.get());
        }
    } else {
        acc_functor(functor(first, last));
    }
}

} //end of namespace etl
