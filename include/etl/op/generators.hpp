//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains generators
 */

#pragma once

#include <chrono> //for std::time

namespace etl {

/*!
 * \brief Generator from a normal distribution
 */
template <typename T = double>
struct normal_generator_op {
    using value_type = T; ///< The value type

    random_engine rand_engine;                         ///< The random engine
    std::normal_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Construct a new generator with the given mean and standard deviation
     * \param mean The mean
     * \param stddev The standard deviation
     */
    normal_generator_op(T mean, T stddev)
            : rand_engine(std::time(nullptr)), distribution(mean, stddev) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return distribution(rand_engine);
    }
};

/*!
 * \brief Selector helper to get an uniform_distribution based on the type (real or int)
 * \tpara T The type of return of the distribution
 */
template<typename T>
using uniform_distribution = std::conditional_t<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    std::uniform_int_distribution<T>>;

/*!
 * \brief Generator from an uniform distribution
 */
template <typename T = double>
struct uniform_generator_op {
    using value_type = T; ///< The value type

    random_engine rand_engine;                     ///< The random engine
    uniform_distribution<value_type> distribution; ///< The used distribution

    /*!
     * \brief Construct a new generator with the given start and end of the range
     * \param start The beginning of the range
     * \param end The end of the range
     */
    uniform_generator_op(T start, T end)
            : rand_engine(std::time(nullptr)), distribution(start, end) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return distribution(rand_engine);
    }
};

/*!
 * \brief Generator from a sequence
 */
template <typename T = double>
struct sequence_generator_op {
    using value_type = T; ///< The value type

    const value_type start; ///< The beginning of the sequence
    value_type current;     ///< The current sequence element

    /*!
     * \brief Construct a new generator with the given sequence start
     * \param start The beginning of the sequence
     */
    explicit sequence_generator_op(value_type start = 0)
            : start(start), current(start) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return current++;
    }
};

/*!
 * \brief Outputs the given generator to the given stream
 * \param os The output stream
 * \param s The generator
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const sequence_generator_op<T>& s) {
    return os << "[" << s.start << ",...]";
}

/*!
 * \brief Outputs the given generator to the given stream
 * \param os The output stream
 * \param s The generator
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const normal_generator_op<T>& s) {
    cpp_unused(s);
    return os << "N(0,1)";
}

} //end of namespace etl
