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

template<typename T>
struct dyn_vector {
public:
    static constexpr const std::size_t n_dimensions = 1;

    using         value_type = T;
    using       storage_impl = std::vector<value_type>;
    using           iterator = typename storage_impl::iterator;
    using     const_iterator = typename storage_impl::const_iterator;

private:
    storage_impl _data;
    const std::size_t _rows;

public:
    //{{{ Construction

    explicit dyn_vector(std::size_t rows) : _data(rows), _rows(rows) {
        //Nothing else to init
    }

    dyn_vector(std::size_t rows, const value_type& value) : _data(rows, value), _rows(rows) {
        //Nothing else to init
    }

    //TODO Probably a better way in order to move elements
    dyn_vector(std::initializer_list<value_type> l) : _data(l), _rows(l.size()){
        //Nothing else to init
    }

    explicit dyn_vector(const dyn_vector& rhs) : _data(rhs._data), _rows(rhs._rows){
        //Nothing else to init
    }

    template<typename LE, typename Op, typename RE>
    explicit dyn_vector(const binary_expr<value_type, LE, Op, RE>& e) : _data(etl::size(e)), _rows(etl::size(e)) {
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    explicit dyn_vector(const unary_expr<value_type, E, Op>& e) : _data(etl::size(e)), _rows(etl::size(e)) {
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E>
    explicit dyn_vector(const transform_expr<value_type, E>& e) : _data(etl::size(e)), _rows(etl::size(e)) {
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    explicit dyn_vector(const Container& vec) : _data(vec.size()), _rows(vec.size()) {
        std::copy(vec.begin(), vec.end(), begin());
    }

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
        ensure_same_size(*this, rhs);

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
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    dyn_vector& operator=(const unary_expr<value_type, E, Op>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E>
    dyn_vector& operator=(const transform_expr<value_type, E>&& e){
        ensure_same_size(*this, e);

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
        return _rows;
    }

    size_t rows() const {
        return _rows;
    }

    size_t dimensions() const {
        return 1;
    }

    size_t dim(std::size_t d) const {
        etl_assert(d == 0, "Invalid dimension");
        etl_unused(d);

        return _rows;
    }

    value_type& operator()(size_t i){
        etl_assert(i < _rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator()(size_t i) const {
        etl_assert(i < _rows, "Out of bounds");

        return _data[i];
    }

    value_type& operator[](size_t i){
        etl_assert(i < _rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator[](size_t i) const {
        etl_assert(i < _rows, "Out of bounds");

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

    storage_impl&& release(){
        return std::move(_data);
    }

    //}}}
};

} //end of namespace etl

#endif
