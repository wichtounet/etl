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
    using base_type::_size;
    using base_type::_dimensions;
    memory_type _memory;
    index_memory_type _row_index;
    index_memory_type _col_index;
    std::size_t nnz;

    using base_type::release;
    using base_type::allocate;
    using base_type::check_invariants;

public:
    using base_type::dim;
    using base_type::rows;
    using base_type::columns;

    // Construction

    //Default constructor (constructs an empty matrix)
    sparse_matrix_impl() noexcept : base_type(), _memory(nullptr), _row_index(nullptr), _col_index(nullptr), nnz(0) {
        //Nothing else to init
    }

    template<typename It>
    void build_from_iterable(const It& iterable){
        nnz =  0;
        for (auto v : iterable) {
            if(v != 0.0){
                ++nnz;
            }
        }

        //Allocate space for the three arrays
        _memory    = allocate(nnz);
        _row_index = base_type::template allocate<index_memory_type>(nnz);
        _col_index = base_type::template allocate<index_memory_type>(nnz);

        auto it = iterable.begin();
        std::size_t n = 0;

        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                if(*it != 0.0){
                    _memory[n] = *it;
                    _row_index[n] = i;
                    _col_index[n] = j;
                    ++n;
                }

                ++it;
            }
        }
    }

    //Sizes followed by an initializer list
    template <typename... S, cpp_enable_if(dyn_detail::is_initializer_list_constructor<S...>::value)>
    explicit sparse_matrix_impl(S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...),
                                                              dyn_detail::sizes(std::make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
                                                    _memory(allocate(_size)) {
        static_assert(sizeof...(S) == D + 1, "Invalid number of dimensions");

        auto list = cpp::last_value(sizes...);
        build_from_iterable(list);
    }

    //Sizes followed by a values_t
    template <typename S1, typename... S, cpp_enable_if(
                                              (sizeof...(S) == D),
                                              cpp::is_specialization_of<values_t, typename cpp::last_type<S1, S...>::type>::value)>
    explicit sparse_matrix_impl(S1 s1, S... sizes) noexcept : base_type(dyn_detail::size(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...),
                                                                     dyn_detail::sizes(std::make_index_sequence<(sizeof...(S))>(), s1, sizes...)),
                                                           _memory(allocate(_size)) {
        auto list = cpp::last_value(sizes...).template list<value_type>();
        build_from_iterable(list);
    }

    template <bool B = n_dimensions == 2, cpp_enable_if(B)>
    value_type access(std::size_t i, std::size_t j) noexcept {
        cpp_assert(i < dim(0), "Out of bounds");
        cpp_assert(j < dim(1), "Out of bounds");

        for(std::size_t n = 0; n < nnz; ++n){
            if(_row_index[n] == i && _col_index[n] == j){
                return _memory[n];
            }

            if(_row_index[n] > i){
                break;
            }
        }

        return 0.0;
    }

    template <bool B = n_dimensions == 2, cpp_enable_if(B)>
    value_type operator()(std::size_t i, std::size_t j) noexcept {
        return access(i, j);
    }

    template <bool B = n_dimensions == 2, cpp_enable_if(B)>
    const value_type operator()(std::size_t i, std::size_t j) const noexcept {
        return access(i, j);
    }

    //Destructor

    ~sparse_matrix_impl() noexcept {
        if(_memory){
            release(_memory, nnz);
        }

        if(_row_index){
            release(_row_index, nnz);
        }

        if(_col_index){
            release(_col_index, nnz);
        }
    }
};

} //end of namespace etl
