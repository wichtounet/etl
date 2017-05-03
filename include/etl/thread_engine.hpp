//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

#ifdef ETL_PARALLEL_SUPPORT

template<typename Pool>
struct conf_thread_engine {
    //static Pool pool;

    static void acquire(){
        cpp_assert(etl::parallel_support, "thread_engine can only be used if paralle support is enabled");
        cpp_assert(!local_context().serial, "thread_engine cannot be used in serial context");
        cpp_assert(etl::threads > 1, "thread_engine cannot be used with less than 2");

        etl::local_context().serial = true;
    }

    template <class Functor, typename... Args>
    static void schedule(Functor&& fun, Args&&... args) {
        get_pool().do_task(std::forward<Functor>(fun), std::forward<Args>(args)...);
    }

    static void wait(){
        get_pool().wait();

        etl::local_context().serial = false;
    }

    static Pool& get_pool(){
        static Pool pool(etl::threads);
        return pool;
    }
};

using thread_engine = conf_thread_engine<cpp::default_thread_pool<>>;

#else

struct thread_engine {
    static void acquire(){
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }

    template <class Functor, typename... Args>
    static void schedule(Functor&& fun, Args&&... args) {
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }

    static void wait(){
        cpp_unreachable("thread_engine can only be used if paralle support is enabled");
    }
};

#endif

} //end of namespace etl
