//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_MATRIX_HPP
#define ETL_FAST_MATRIX_HPP

#include<array>
#include<string>

#include "assert.hpp"
#include "traits.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

//TODO Ensure that the binary_expr that is taken comes from a matrix
//or least from a vector of Rows * Columns size

namespace etl {

template<typename T, size_t Rows, size_t Columns>
struct fast_matrix {
public:
    using       value_type = T;
    using     storage_impl = std::array<value_type, Rows * Columns>;
    using         iterator = typename storage_impl::iterator;
    using   const_iterator = typename storage_impl::const_iterator;

    static constexpr const std::size_t etl_size = Rows * Columns;

private:
    storage_impl _data;

public:

    ///{{{ Construction

    fast_matrix(){
        //Nothing to init
    }

    fast_matrix(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix(std::initializer_list<value_type> l){
        etl_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    template<typename LE, typename Op, typename RE>
    fast_matrix(const binary_expr<value_type, LE, Op, RE>& e){
        for(std::size_t i = 0; i < Rows; ++i){
            for(std::size_t j = 0; j < Columns; ++j){
                _data[i * Columns + j] = e(i,j);
            }
        }
    }

    template<typename E, typename Op>
    fast_matrix(const unary_expr<value_type, E, Op>& e){
        for(std::size_t i = 0; i < Rows; ++i){
            for(std::size_t j = 0; j < Columns; ++j){
                _data[i * Columns + j] = e(i,j);
            }
        }
    }

    //Prohibit copy and move
    fast_matrix(const fast_matrix& rhs) = delete;
    fast_matrix(fast_matrix&& rhs) = delete;

    //}}}

    //{{{ Assignment

    //Copy assignment operator

    fast_matrix& operator=(const fast_matrix& rhs){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    fast_matrix& operator=(const Container& vec){
        etl_assert(vec.size() == Rows * Columns, "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < Rows * Columns; ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    fast_matrix& operator=(binary_expr<value_type, LE, Op, RE>&& e){
        for(std::size_t i = 0; i < Rows; ++i){
            for(std::size_t j = 0; j < Columns; ++j){
                _data[i * Columns + j] = e(i,j);
            }
        }

        return *this;
    }

    template<typename E, typename Op>
    fast_matrix& operator=(unary_expr<value_type, E, Op>&& e){
        for(std::size_t i = 0; i < Rows; ++i){
            for(std::size_t j = 0; j < Columns; ++j){
                _data[i * Columns + j] = e(i,j);
            }
        }

        return *this;
    }

    //Set the same value to each element of the matrix
    fast_matrix& operator=(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Prohibit move
    fast_matrix& operator=(fast_matrix&& rhs) = delete;

    //}}}

    //{{{ Accessors

    static constexpr size_t size(){
        return Rows * Columns;
    }

    static constexpr size_t rows(){
        return Rows;
    }

    static constexpr size_t columns(){
        return Columns;
    }

    value_type& operator()(size_t i, size_t j){
        etl_assert(i < Rows, "Out of bounds");
        etl_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const value_type& operator()(size_t i, size_t j) const {
        etl_assert(i < Rows, "Out of bounds");
        etl_assert(j < Columns, "Out of bounds");

        return _data[i * Columns + j];
    }

    const value_type& operator[](size_t i) const {
        etl_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    value_type& operator[](size_t i){
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

    //}}}
};

} //end of namespace etl

#endif
