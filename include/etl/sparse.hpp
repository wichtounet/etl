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

#include "etl/traits_lite.hpp"    //forward declaration of the traits
#include "etl/compat.hpp"         //To make it work with g++

namespace etl {

template <typename T, sparse_storage SS, std::size_t D>
struct sparse_matrix_impl;

//Implementation with COO format
template <typename T, std::size_t D>
struct sparse_matrix_impl <T, sparse_storage::COO, D> final {
    static_assert(D > 0, "A matrix must have a least 1 dimension");

public:
    static constexpr const std::size_t n_dimensions      = D;
    static constexpr const sparse_storage storage_format = sparse_storage::COO;
    static constexpr const std::size_t alignment         = intrinsic_traits<T>::alignment;

    using value_type             = T;
    using dimension_storage_impl = std::array<std::size_t, n_dimensions>;
    using memory_type            = value_type*;
    using const_memory_type      = const value_type*;
    using index_memory_type      = std::size_t*;
    using iterator               = memory_type;
    using const_iterator         = const_memory_type;
    using vec_type               = intrinsic_type<T>;

private:
    std::size_t _size;
    dimension_storage_impl _dimensions;
    memory_type _memory;
    index_memory_type _row_index;
    index_memory_type _col_index;
    std::size_t nnz;

    static memory_type allocate(std::size_t n) {
        auto* memory = aligned_allocator<void, alignment>::template allocate<T>(n);
        cpp_assert(memory, "Impossible to allocate memory for dyn_matrix");
        cpp_assert(reinterpret_cast<uintptr_t>(memory) % alignment == 0, "Failed to align memory of matrix");

        //In case of non-trivial type, we need to call the constructors
        if(!std::is_trivial<value_type>::value){
            new (memory) value_type[n]();
        }

        return memory;
    }

    static void release(memory_type ptr, std::size_t n) {
        //In case of non-trivial type, we need to call the destructors
        if(!std::is_trivial<value_type>::value){
            for(std::size_t i = 0; i < n; ++i){
                ptr[i].~value_type();
            }
        }

        aligned_allocator<void, alignment>::template release<T>(ptr);
    }

public:
    // Construction

    //Default constructor (constructs an empty matrix)
    sparse_matrix_impl() noexcept : _size(0), _memory(nullptr), _row_index(nullptr), _col_index(nullptr), nnz(0) {
        std::fill(_dimensions.begin(), _dimensions.end(), 0);

        //check_invariants();
    }


    // Accessors

    std::size_t size() const noexcept {
        return _size;
    }

    std::size_t rows() const noexcept {
        return _dimensions[0];
    }

    std::size_t columns() const noexcept {
        static_assert(n_dimensions > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }
};

} //end of namespace etl
