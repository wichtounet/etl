//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains diagonal matrix exception implementation
 */

#pragma once

#include <exception>

namespace etl {

/*!
 * \brief Exception that is thrown when an operation is made to
 * a diagonal matrix that would render it non-diagonal.
 */
struct diagonal_exception : std::exception {
    /*!
     * \brief Returns a description of the exception
     */
    const char* what() const noexcept override {
        return "Invalid assignment to a diagonal matrix";
    }
};

} //end of namespace etl
