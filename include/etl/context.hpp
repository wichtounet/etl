//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

struct context {
    bool serial = false;
    bool parallel = false;
};

inline context& local_context(){
    static thread_local context local_context;
    return local_context;
}

namespace detail {

struct serial_context {
    bool old_serial;

    serial_context(){
        old_serial = etl::local_context().serial;
        etl::local_context().serial = true;
    }

    ~serial_context(){
        etl::local_context().serial = old_serial;
    }

    operator bool(){
        return true;
    }
};

struct parallel_context {
    bool old_parallel;

    parallel_context(){
        old_parallel = etl::local_context().parallel;
        etl::local_context().parallel = true;
    }

    ~parallel_context(){
        etl::local_context().parallel = old_parallel;
    }

    operator bool(){
        return true;
    }
};

} //end of namespace detail

#define SERIAL_SECTION if(auto etl_serial_context__ = etl::detail::serial_context())
#define PARALLEL_SECTION if(auto etl_parallel_context__ = etl::detail::parallel_context())

} //end of namespace etl
