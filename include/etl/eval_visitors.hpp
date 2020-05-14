//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
 * \brief Visitor to perform local evaluation when necessary
 */
struct evaluator_visitor {
    // Nothing to configure
};

} //end of namespace detail

} //end of namespace etl
