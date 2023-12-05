//=======================================================================
// Copyright (c) 2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Traits to get information about ETL types
 *
 * For non-ETL types, is_etl is false and in that case, no other fields should be used on the traits.
 *
 * \tparam T the type to introspect
 */
template <typename T>
struct etl_traits {
    static constexpr bool is_etl         = false; ///< Indicates if T is an ETL type
    static constexpr bool is_transformer = false; ///< Indicates if T is a transformer
    static constexpr bool is_view        = false; ///< Indicates if T is a view
    static constexpr bool is_magic_view  = false; ///< Indicates if T is a magic view
    static constexpr bool is_fast        = false; ///< Indicates if T is a fast structure
    static constexpr bool is_generator   = false; ///< Indicates if T is a generator expression

    /*!
     * \brief Return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 0;
    }
};

} //end of namespace etl
