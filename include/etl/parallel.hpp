//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <thread>

#include "cpp_utils/parallel.hpp"

namespace etl {

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

} //end of namespace etl
