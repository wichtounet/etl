//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Indicates if an 1D evaluation should run in paralle
 * \param n The size of the evaluation
 * \param threshold The parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool select_parallel(size_t n, size_t threshold = parallel_threshold) {
    return threads > 1 && ((parallel_support && local_context().parallel)|| (is_parallel && n >= threshold && !local_context().serial));
}

/*!
 * \brief Indicates if an 2D evaluation should run in paralle
 * \param n1 The first dimension of the evaluation
 * \param t1 The first parallel threshold
 * \param n2 The second dimension of the evaluation
 * \param t2 The second parallel threshold
 * \return true if the evaluation should be done in paralle, false otherwise
 */
inline bool select_parallel_2d(size_t n1, size_t t1, size_t n2, size_t t2) {
    return threads > 1 && ((parallel_support && local_context().parallel) || (is_parallel && n1 >= t1 && n2 >= t2 && !local_context().serial));
}

} //end of namespace etl
