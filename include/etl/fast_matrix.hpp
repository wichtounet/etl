//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_MATRIX_HPP
#define ETL_FAST_MATRIX_HPP

#include<array>

#include "assert.hpp"
#include "tmp.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

//TODO Ensure that the binary_expr that is taken comes from a matrix
//or least from a vector of Rows * Columns size

namespace etl {

template<typename T, size_t Rows, size_t Columns>
class fast_matrix {
public:
    typedef std::array<T, Rows * Columns> array_impl;
    typedef typename array_impl::iterator iterator;
    typedef typename array_impl::const_iterator const_iterator;

private:
    array_impl _data;

public:
    static constexpr const bool etl_marker = true;

    using value_type = T;

    static constexpr const std::size_t rows = Rows;
    static constexpr const std::size_t columns = Columns;

    fast_matrix(){
        //Nothing to init
    }

    fast_matrix(const T& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix(std::initializer_list<T> l){
        etl_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    fast_matrix& operator=(const fast_matrix& rhs){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    fast_matrix(binary_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename LE, typename Op, typename RE>
    fast_matrix& operator=(binary_expr<T, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    fast_matrix(unary_expr<T, E, Op>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    fast_matrix& operator=(unary_expr<T, E, Op>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    //Prohibit copy
    fast_matrix(const fast_matrix& rhs) = delete;

    //Make sure
    fast_matrix(fast_matrix&& rhs) = default;
    fast_matrix& operator=(fast_matrix&& rhs) = default;

    //Modifiers

    //Set the same value to each element of the matrix
    fast_matrix& operator=(const T& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Multiply each element by a scalar
    fast_matrix& operator*=(const T& value){
        for(size_t i = 0; i < size(); ++i){
            _data[i] *= value;
        }

        return *this;
    }

    //Divide each element by a scalar
    fast_matrix& operator/=(const T& value){
        for(size_t i = 0; i < size(); ++i){
            _data[i] /= value;
        }

        return *this;
    }

    template<typename RE>
    fast_matrix& operator+=(RE&& rhs){
        for(size_t i = 0; i < size(); ++i){
            _data[i] += rhs[i];
        }

        return *this;
    }

    template<typename RE>
    fast_matrix& operator-=(RE&& rhs){
        for(size_t i = 0; i < size(); ++i){
            _data[i] -= rhs[i];
        }

        return *this;
    }

    //Accessors

    constexpr size_t size() const {
        return Rows * Columns;
    }

    T& operator()(size_t i, size_t j){
        etl_assert(i < Rows, "Out of bounds");
        etl_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const T& operator()(size_t i, size_t j) const {
        etl_assert(i < Rows, "Out of bounds");
        etl_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const T& operator[](size_t i) const {
        etl_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    T& operator[](size_t i){
        etl_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    const_iterator begin() const {
        return _data.begin();
    }

    const_iterator end() const {
        return _data.end();
    }

    iterator begin(){
        return _data.begin();
    }

    iterator end(){
        return _data.end();
    }
};

} //end of namespace etl

#endif
