//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains sequence generators
 */

#pragma once

namespace etl {

/*!
 * \brief Generator from a sequence
 */
template <typename T = double>
struct sequence_generator_op {
    using value_type = T; ///< The value type

    const value_type start; ///< The beginning of the sequence
    value_type current;     ///< The current sequence element

    static constexpr bool gpu_computable = false; ///< Indicates if the operator is computable on GPU

    /*!
     * \brief Construct a new generator with the given sequence start
     * \param start The beginning of the sequence
     */
    explicit sequence_generator_op(value_type start = 0) : start(start), current(start) {}

    /*!
     * \brief Generate a new value
     * \return the newly generated value
     */
    value_type operator()() {
        return current++;
    }

    /*!
     * \brief Outputs the given generator to the given stream
     * \param os The output stream
     * \param s The generator
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const sequence_generator_op& s) {
        return os << "[" << s.start << ",...]";
    }
};

} //end of namespace etl
