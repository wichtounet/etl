//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DYN_VECTOR_HPP
#define ETL_DYN_VECTOR_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <array>
#include <vector>
#include <numeric>
#include <initializer_list>

#include "assert.hpp"
#include "tmp.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

namespace etl {

//TODO Make it moveable

template<typename T>
struct dyn_vector {
public:
    using         value_type = T;
    using       storage_impl = std::vector<value_type>;
    using           iterator = typename storage_impl::iterator;
    using     const_iterator = typename storage_impl::const_iterator;

private:
    storage_impl _data;
    const std::size_t rows;

public:

    //{{{ Construction

    dyn_vector(std::size_t rows) : _data(rows), rows(rows) {
        //Nothing else to init
    }

    dyn_vector(std::size_t rows, const value_type& value) : _data(rows, value), rows(rows) {
        //Nothing else to init
    }

    //TODO Probably a better way in order to move elements
    dyn_vector(std::initializer_list<value_type> l) : _data(l), rows(l.size()){
        //Nothing else to init
    }

    template<typename LE, typename Op, typename RE>
    dyn_vector(const binary_expr<value_type, LE, Op, RE>& e) : _data(::size(e)), rows(::size(e)) {
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    dyn_vector(const unary_expr<value_type, E, Op>& e) : _data(::size(e)), rows(::size(e)) {
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    //Prohibit copy 
    dyn_vector(const dyn_vector& rhs) = delete;

    //Move is possible
    dyn_vector(dyn_vector&& rhs) = default;

    //}}}

    //{{{Assignment

    //Set every element to the same scalar
    void operator=(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    //Copy
    dyn_vector& operator=(const dyn_vector& rhs){
        etl_assert(size() == rhs.size(), "Can only copy from vector of same size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Move is possible
    dyn_vector& operator=(dyn_vector&& rhs) = default;

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    dyn_vector& operator=(const binary_expr<value_type, LE, Op, RE>&& e){
        etl_assert(size() == ::size(e), "Can only copy from expr of same size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    dyn_vector& operator=(const unary_expr<value_type, E, Op>&& e){
        etl_assert(size() == ::size(e), "Can only copy from expr of same size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    dyn_vector& operator=(const Container& vec){
        etl_assert(vec.size() == size(), "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //}}}

    //{{{ Accessors

    size_t size() const {
        return rows;
    }

    value_type& operator()(size_t i){
        etl_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator()(size_t i) const {
        etl_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    value_type& operator[](size_t i){
        etl_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator[](size_t i) const {
        etl_assert(i < rows, "Out of bounds");

        return _data[i];
    }

    const_iterator begin() const {
        return _data.begin();
    }

    iterator begin(){
        return _data.begin();
    }

    const_iterator end() const {
        return _data.end();
    }

    iterator end(){
        return _data.end();
    }

    //}}}
};

} //end of namespace etl

#endif
