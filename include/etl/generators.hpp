//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_GENERATORS_HPP
#define ETL_GENERATORS_HPP

#include <chrono> //for std::time
#include <iostream> //For stream support
#include <random>

#include "cpp_utils/tmp.hpp"

namespace etl {

using random_engine = std::mt19937_64;

template<typename T>
struct scalar {
    const T value;

    explicit constexpr scalar(T v) : value(v) {}

    constexpr const T operator[](std::size_t) const {
        return value;
    }

    template<typename... S>
    T operator()(S... /*args*/) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value;
    }
};

template<typename T = double>
struct normal_generator_op {
    using value_type = T;

    random_engine rand_engine;
    std::normal_distribution<value_type> normal_distribution;

    normal_generator_op() : rand_engine(std::time(nullptr)), normal_distribution(0.0, 1.0) {}

    value_type operator()(){
        return normal_distribution(rand_engine);
    }
};

template<typename T = double>
struct sequence_generator_op {
    using value_type = T;

    const value_type start;
    value_type current;

    sequence_generator_op(value_type start = 0) : start(start), current(start) {}

    value_type operator()(){
        return current++;
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const scalar<T>& s){
    return os << s.value;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const sequence_generator_op<T>& s){
    return os << "[" << s.start << ",...]";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const normal_generator_op<T>&){
    return os << "N(0,1)";
}

} //end of namespace etl

#endif
