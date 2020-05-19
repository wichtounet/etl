//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifndef ETL_COUNTERS

namespace etl {

/*!
 * \brief Dump all counters values to the console.
 */
inline void dump_counters() {
    //No counters
}

/*!
 * \brief Increase the given counter
 * \param name The name of the counter to increase
 */
inline void inc_counter([[maybe_unused]] const char* name) {}

} //end of namespace etl

#else

#include <chrono>
#include <iosfwd>
#include <iomanip>
#include <sstream>

namespace etl {

constexpr const size_t max_counters = 64;

struct counter_t {
    const char* name;
    std::atomic<size_t> count;

    counter_t() : name(nullptr), count(0) {}

    counter_t(const counter_t& rhs) : name(rhs.name), count(rhs.count.load()) {}

    counter_t& operator=(const counter_t& rhs) {
        if (&rhs != this) {
            name  = rhs.name;
            count = rhs.count.load();
        }

        return *this;
    }

    counter_t(counter_t&& rhs) : name(std::move(rhs.name)), count(rhs.count.load()) {}

    counter_t& operator=(counter_t&& rhs) {
        if (&rhs != this) {
            name  = std::move(rhs.name);
            count = rhs.count.load();
        }

        return *this;
    }
};

struct counters_t {
    std::array<counter_t, max_counters> counters;
    std::mutex lock;

    void reset() {
        std::lock_guard<std::mutex> l(lock);

        for (auto& counter : counters) {
            counter.name  = nullptr;
            counter.count = 0;
        }
    }
};

inline counters_t& get_counters() {
    static counters_t counters;
    return counters;
}

/*!
 * \brief Reset all counters
 */
inline void reset_counters() {
    decltype(auto) counters = get_counters();
    counters.reset();
}

/*!
 * \brief Dump all counters values to the console.
 */
inline void dump_counters() {
    decltype(auto) counters = get_counters().counters;

    //Sort the counters by count (DESC)
    std::sort(counters.begin(), counters.end(), [](auto& left, auto& right) { return left.count > right.count; });

    // Print all the used counters
    for (decltype(auto) counter : counters) {
        if (counter.name) {
            std::cout << counter.name << ": " << counter.count << std::endl;
        }
    }
}

/*!
 * \brief Dump all counters values to the console.
 */
inline void dump_counters_pretty() {
    decltype(auto) counters = get_counters().counters;

    if(counters.empty()){
        std::cout << "No counters have been recorded!" << std::endl;
        return;
    }

    std::cout << std::endl;

    //Sort the counters by count (DESC)
    std::sort(counters.begin(), counters.end(), [](auto& left, auto& right) { return left.count > right.count; });

    constexpr size_t columns = 2;

    std::string column_name[columns];
    column_name[0] = "Counter";
    column_name[1] = "Count";

    size_t column_length[columns];
    column_length[0] = column_name[0].size();
    column_length[1] = column_name[1].size();

    // Compute the width of each column
    for (decltype(auto) counter : counters) {
        if (counter.name) {
            column_length[0] = std::max(column_length[0], std::string(counter.name).size());
            column_length[1] = std::max(column_length[1], std::to_string(counter.count).size());
        }
    }

    const size_t line_length = (columns + 1) * 1 + 2 + (columns - 1) * 2 + std::accumulate(column_length, column_length + columns, 0);

    std::cout << " " << std::string(line_length, '-') << '\n';

    printf(" | %-*s | %-*s |\n",
        int(column_length[0]), column_name[0].c_str(),
        int(column_length[1]), column_name[1].c_str());

    // Print all the used counters
    for (decltype(auto) counter : counters) {
        if (counter.name) {
            printf(" | %-*s | %-*s |\n",
                int(column_length[0]), counter.name,
                int(column_length[1]), std::to_string(counter.count).c_str());
        }
    }

    std::cout << " " << std::string(line_length, '-') << '\n';

    // Print all the used counters
    for (decltype(auto) counter : counters) {
        if (counter.name) {
            std::cout << counter.name << ": " << counter.count << std::endl;
        }
    }
}

/*!
 * \brief Increase the given counter
 * \param name The name of the counter to increase
 */
inline void inc_counter(const char* name) {
#ifdef ETL_COUNTERS_VERBOSE
    std::cout << "counter:inc:" << name << std::endl;
#endif

    decltype(auto) counters = get_counters();

    for (decltype(auto) counter : counters.counters) {
        if (counter.name == name) {
            ++counter.count;

            return;
        }
    }

    std::lock_guard<std::mutex> lock(counters.lock);

    for (decltype(auto) counter : counters.counters) {
        if (counter.name == name) {
            ++counter.count;

            return;
        }
    }

    for (decltype(auto) counter : counters.counters) {
        if (!counter.name) {
            counter.name  = name;
            counter.count = 1;

            return;
        }
    }

    std::cerr << "Unable to register counter " << name << std::endl;
}

} //end of namespace etl

#endif
