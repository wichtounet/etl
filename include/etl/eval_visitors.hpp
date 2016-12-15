//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file eval_visitors.hpp
 * \brief Contains the visitors used by the evaluator to process the
 * expression trees.
*/

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief Visitor to allocate temporary when needed
 */
struct temporary_allocator_visitor {
    // Simple tag
};

/*!
 * \brief Visitor to evict GPU temporaries from the Expression tree
 */
struct gpu_clean_visitor {
    // Simple tag
};

/*!
 * \brief Visitor to perform local evaluation when necessary
 */
struct evaluator_visitor {
    bool need_value = false; ///< Indicates if the value if necessary for the next visits
};

} //end of namespace detail

} //end of namespace etl
