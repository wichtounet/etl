//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <array>     //To store the dimensions
#include <tuple>     //For TMP stuff
#include <algorithm> //For std::find_if
#include <iosfwd>    //For stream support

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/traits_lite.hpp" //forward declaration of the traits
#include "etl/compat.hpp"      //To make it work with g++
#include "etl/dyn_base.hpp"    //The base class and utilities

namespace etl {

template <typename T, sparse_storage SS, std::size_t D>
struct sparse_matrix_impl;

//Implementation with COO format
template <typename T, std::size_t D>
struct sparse_matrix_impl <T, sparse_storage::COO, D> final : dyn_base<T, D> {
    static constexpr const std::size_t n_dimensions      = D;
    static constexpr const sparse_storage storage_format = sparse_storage::COO;
    static constexpr const std::size_t alignment         = intrinsic_traits<T>::alignment;

    using base_type              = dyn_base<T, D>;
    using value_type             = T;
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;
    using memory_type            = value_type*;
    using const_memory_type      = const value_type*;
    using index_memory_type      = std::size_t*;
    using iterator               = memory_type;
    using const_iterator         = const_memory_type;
    using vec_type               = intrinsic_type<T>;

private:
    memory_type _memory;
    index_memory_type _row_index;
    index_memory_type _col_index;
    std::size_t nnz;

public:
    // Construction

    //Default constructor (constructs an empty matrix)
    sparse_matrix_impl() noexcept : base_type(), _memory(nullptr), _row_index(nullptr), _col_index(nullptr), nnz(0) {
        //Nothing else to init
    }
};

} //end of namespace etl
