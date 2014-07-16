//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_VECTOR_HPP
#define ETL_FAST_VECTOR_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <array>
#include <vector>
#include <numeric>
#include <initializer_list>

#include "assert.hpp"
#include "traits.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

namespace etl {

template<typename T, std::size_t Rows>
struct fast_vector {
public:
    using         value_type = T;
    using       storage_impl = std::array<value_type, Rows>;
    using           iterator = typename storage_impl::iterator;
    using     const_iterator = typename storage_impl::const_iterator;

    static constexpr const std::size_t etl_size = Rows;

private:
    storage_impl _data;

public:

    //{{{ Construction

    fast_vector(){
        //Nothing else to init
    }

    fast_vector(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_vector(std::initializer_list<value_type> l){
        etl_assert(l.size() == Rows, "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    fast_vector(const fast_vector& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    fast_vector(fast_vector&& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    template<typename LE, typename Op, typename RE>
    fast_vector(const binary_expr<value_type, LE, Op, RE>& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    fast_vector(const unary_expr<value_type, E, Op>& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }
    }

    //}}}

    //{{{Assignment

    //Set every element to the same scalar
    void operator=(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    //Copy
    fast_vector& operator=(const fast_vector& rhs){
        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Prohibit move
    fast_vector& operator=(fast_vector&& rhs) = delete;

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    fast_vector& operator=(const binary_expr<value_type, LE, Op, RE>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    fast_vector& operator=(const unary_expr<value_type, E, Op>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    fast_vector& operator=(const Container& vec){
        for(std::size_t i = 0; i < Rows; ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //}}}

    //{{{ Accessors

    static constexpr size_t size(){
        return Rows;
    }

    static constexpr size_t rows(){
        return Rows;
    }

    value_type& operator()(size_t i){
        etl_assert(i < Rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator()(size_t i) const {
        etl_assert(i < Rows, "Out of bounds");

        return _data[i];
    }

    value_type& operator[](size_t i){
        etl_assert(i < Rows, "Out of bounds");

        return _data[i];
    }

    const value_type& operator[](size_t i) const {
        etl_assert(i < Rows, "Out of bounds");

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
