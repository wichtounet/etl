//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

using timer_clock = std::chrono::steady_clock;
using seconds = std::chrono::seconds;
using milliseconds = std::chrono::milliseconds;
using microseconds = std::chrono::microseconds;
using nanoseconds = std::chrono::nanoseconds;
using clock_resolution = nanoseconds;

template<typename Resolution>
std::string resolution_to_string(){
    if(std::is_same<Resolution, seconds>::value){
        return "s";
    } else if(std::is_same<Resolution, milliseconds>::value){
        return "ms";
    } else if(std::is_same<Resolution, microseconds>::value){
        return "us";
    } else if(std::is_same<Resolution, nanoseconds>::value){
        return "ns";
    } else {
        return "?";
    }
}

} //end of namespace etl
