//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains truncated normal generators
 */

#pragma once

#include <chrono> //for std::time

namespace etl {

/*!
 * \brief Generator from a normal distribution
 */
template <typename T = double>
struct truncated_normal_generator_op {
    using value_type = T; ///< The value type

    random_engine rand_engine;                         ///< The random engine
    std::normal_distribution<value_type> distribution; ///< The used distribution

    static constexpr bool gpu_computable = false; ///< Indicates if the operator is computable on GPU

    /*!
     * \brief Construct a new generator with the given mean and standard deviation
     * \param mean The mean
     * \param stddev The standard deviation
     */
    truncated_normal_generator_op(T mean, T stddev)
            : rand_engine(std::time(nullptr)), distribution(mean, stddev) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        auto x = distribution(rand_engine);

        while (std::abs(x - distribution.mean()) > 2.0 * distribution.stddev()) {
            x = distribution(rand_engine);
        }

        return x;
    }

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const truncated_normal_generator_op& s) {
        cpp_unused(s);
        return os << "TN(0,1)";
    }
};

/*!
 * \brief Generator from a normal distribution using a custom random engine.
 */
template <typename G, typename T = double>
struct truncated_normal_generator_g_op {
    using value_type = T; ///< The value type

    G& rand_engine;                                    ///< The random engine
    std::normal_distribution<value_type> distribution; ///< The used distribution

    static constexpr bool gpu_computable = false; ///< Indicates if the operator is computable on GPU

    /*!
     * \brief Construct a new generator with the given mean and standard deviation
     * \param mean The mean
     * \param stddev The standard deviation
     */
    truncated_normal_generator_g_op(G& g, T mean, T stddev)
            : rand_engine(g), distribution(mean, stddev) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        auto x = distribution(rand_engine);

        while (std::abs(x - distribution.mean()) > 2.0 * distribution.stddev()) {
            x = distribution(rand_engine);
        }

        return x;
    }

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const truncated_normal_generator_g_op& s) {
        cpp_unused(s);
        return os << "TN(0,1)";
    }
};

} //end of namespace etl
