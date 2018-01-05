//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

using seconds      = std::chrono::seconds;      ///< The seconds resolution
using milliseconds = std::chrono::milliseconds; ///< The milliseconds resolution
using microseconds = std::chrono::microseconds; ///< The microseconds resolution
using nanoseconds  = std::chrono::nanoseconds;  ///< The nanoseconds resolution

using timer_clock      = std::chrono::steady_clock; ///< The chrono clock used by ETL
using clock_resolution = nanoseconds;               ///< The clock resolution used by ETL

/*!
 * \brief return the string representation of the given resolution
 * \return the tring representation of the given resolution
 */
template <typename Resolution>
std::string resolution_to_string() {
    if (std::is_same<Resolution, seconds>::value) {
        return "s";
    } else if (std::is_same<Resolution, milliseconds>::value) {
        return "ms";
    } else if (std::is_same<Resolution, microseconds>::value) {
        return "us";
    } else if (std::is_same<Resolution, nanoseconds>::value) {
        return "ns";
    } else {
        return "?";
    }
}

} //end of namespace etl
