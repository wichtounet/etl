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
};

inline context& local_context(){
    static thread_local context local_context;
    return local_context;
}

} //end of namespace etl
