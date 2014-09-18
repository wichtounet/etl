//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_DYN_MATRIX_HPP
#define ETL_DYN_MATRIX_HPP

#include<vector>
#include<tuple>

#include "assert.hpp"
#include "tmp.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"
#include "integer_sequence.hpp"

namespace etl {

namespace dyn_detail {

template<std::size_t N, typename... T>
struct nth_type {
    using type = typename std::tuple_element<N, std::tuple<T...>>::type;
};

template<typename... T>
struct last_type {
    using type = typename nth_type<sizeof...(T)-1, T...>::type;
};

template<int I, typename T1, typename... T>
auto nth_value(T1&& t, T&&... /*args*/) -> typename std::enable_if<(I == 0), decltype(std::forward<T1>(t))>::type{
    return std::forward<T1>(t);
}

// Induction step
template<int I, typename T1, typename... T>
auto nth_value(T1&& /*t*/, T&&... args) ->
    typename std::enable_if<(I > 0), decltype(
        std::forward<typename nth_type<I, T1, T...>::type>(
            std::declval<typename nth_type<I, T1, T...>::type>()
            )
        )>::type
{
    using return_type = typename nth_type<I, T1, T...>::type;
    return std::forward<return_type>(nth_value<I - 1>((std::forward<T>(args))...));
}

template<typename... T>
auto last_value(T&&... args){
    return nth_value<sizeof...(T) - 1>(args...);
}

inline std::size_t size(std::size_t first){
    return first;
}

template<typename... T>
inline std::size_t size(std::size_t first, T... args){
    return first * size(args...);
}

template<std::size_t... I, typename... T>
inline std::size_t size(const index_sequence<I...>& /*i*/, const T&... args){
    return size((nth_value<I>(args...))...);
}

template<std::size_t... I, typename... T>
inline std::vector<std::size_t> sizes(const index_sequence<I...>& /*i*/, const T&... args){
    return {static_cast<std::size_t>(nth_value<I>(args...))...};
}

} // end of namespace dyn_detail

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

template<typename T>
struct big_dyn_matrix {
public:
    using                value_type = T;
    using              storage_impl = std::vector<value_type>;
    using    dimension_storage_impl = std::vector<std::size_t>;
    using                  iterator = typename storage_impl::iterator;
    using            const_iterator = typename storage_impl::const_iterator;

private:
    const std::size_t _size;
    storage_impl _data;
    dimension_storage_impl _dimensions;

public:
    ///{{{ Construction

    explicit big_dyn_matrix(const big_dyn_matrix& rhs) : _size(rhs._size), _data(rhs._data), _dimensions(rhs._dimensions) {
        //Nothing to init
    }

    //Normal constructor with only sizes
    template<typename... S, enable_if_u<and_u<(sizeof...(S) > 0), all_convertible_to<std::size_t, S...>::value>::value> = detail::dummy>
    big_dyn_matrix(S... sizes) : 
            _size(dyn_detail::size(sizes...)), 
            _data(_size), 
            _dimensions({static_cast<std::size_t>(sizes)...}) {
        //Nothing to init
    }

    //Sizes followed by an initializer list
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 1),
            is_specialization_of<std::initializer_list, typename dyn_detail::last_type<S...>::type>::value
        >::value> = detail::dummy>
    big_dyn_matrix(S... sizes) : 
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-1)>(), sizes...)), 
            _data(_size), 
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        //Nothing to init
    }

    //Sizes followed by an init flag followed by the value
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 2),
            std::is_same<init_flag_t, typename dyn_detail::nth_type<sizeof...(S) - 2, S...>::type>::value
        >::value> = detail::dummy>
    big_dyn_matrix(S... sizes) : 
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-2)>(), sizes...)), 
            _data(_size, dyn_detail::last_value(sizes...)), 
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-2)>(), sizes...)) {
        //Nothing to init
    }

    //template<typename... S>
    //big_dyn_matrix(S... sizes) : _size(dyn_detail::size(sizes...)), _data(_size), _dimensions({static_cast<std::size_t>(sizes)...}) {
        //static_assert(all_convertible_to<std::size_t, S...>::value, "Invalid sizes");
    //}

    //dyn_matrix(std::size_t rows, std::size_t columns, const value_type& value) : _data(rows * columns), _rows(rows), _columns(columns) {
        //std::fill(_data.begin(), _data.end(), value);
    //}

    //dyn_matrix(std::size_t rows, std::size_t columns, std::initializer_list<value_type> l) : _data(rows * columns), _rows(rows), _columns(columns) {
        //etl_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        //std::copy(l.begin(), l.end(), begin());
    //}

    //template<typename LE, typename Op, typename RE>
    //explicit dyn_matrix(const binary_expr<value_type, LE, Op, RE>& e) :
            //_data(etl::rows(e) * etl::columns(e)),
            //_rows(etl::rows(e)),
            //_columns(etl::columns(e)) {
        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}
    //}

    //template<typename E, typename Op>
    //explicit dyn_matrix(const unary_expr<value_type, E, Op>& e) :
            //_data(etl::rows(e) * etl::columns(e)),
            //_rows(etl::rows(e)),
            //_columns(etl::columns(e)){
        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}
    //}

    //template<typename E>
    //explicit dyn_matrix(const transform_expr<value_type, E>& e) :
            //_data(etl::rows(e) * etl::columns(e)),
            //_rows(etl::rows(e)),
            //_columns(etl::columns(e)){
        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}
    //}

    ////Default move
    //dyn_matrix(dyn_matrix&& rhs) = default;

    ////}}}

    ////{{{ Assignment

    ////Copy assignment operator

    //dyn_matrix& operator=(const dyn_matrix& rhs){
        //ensure_same_size(*this, rhs);

        //for(std::size_t i = 0; i < size(); ++i){
            //_data[i] = rhs[i];
        //}

        //return *this;
    //}

    ////Allow copy from other containers

    //template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    //dyn_matrix& operator=(const Container& vec){
        //etl_assert(vec.size() == _rows * _columns, "Cannot copy from a vector of different size");

        //for(std::size_t i = 0; i < _rows * _columns; ++i){
            //_data[i] = vec[i];
        //}

        //return *this;
    //}

    ////Construct from expression

    //template<typename LE, typename Op, typename RE>
    //dyn_matrix& operator=(binary_expr<value_type, LE, Op, RE>&& e){
        //ensure_same_size(*this, e);

        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}

        //return *this;
    //}

    //template<typename E, typename Op>
    //dyn_matrix& operator=(unary_expr<value_type, E, Op>&& e){
        //ensure_same_size(*this, e);

        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}

        //return *this;
    //}

    //template<typename E>
    //dyn_matrix& operator=(transform_expr<value_type, E>&& e){
        //ensure_same_size(*this, e);

        //for(std::size_t i = 0; i < rows(); ++i){
            //for(std::size_t j = 0; j < columns(); ++j){
                //_data[i * columns() + j] = e(i,j);
            //}
        //}

        //return *this;
    //}

    ////Set the same value to each element of the matrix
    //dyn_matrix& operator=(const value_type& value){
        //std::fill(_data.begin(), _data.end(), value);

        //return *this;
    //}

    ////Default move
    //dyn_matrix& operator=(dyn_matrix&& rhs) = default;

    //}}}

    //{{{ Accessors

    size_t size() const {
        return _size;
    }

    size_t rows() const {
        return _dimensions[0];
    }

    size_t columns() const {
        etl_assert(_dimensions.size() > 1, "columns() only valid for 2D+ matrices");
        return _dimensions[1];
    }

    size_t dimensions() const {
        return _dimensions.size();
    }

    size_t dim(std::size_t d) const {
        etl_assert(d < _dimensions.size(), "Invalid dimension");

        return _dimensions[d];
    }

    //value_type& operator()(size_t i, size_t j){
        //etl_assert(i < _rows, "Out of bounds");
        //etl_assert(j < _columns, "Out of bounds");

        //return _data[i * _columns + j];
    //}

    //const value_type& operator()(size_t i, size_t j) const {
        //etl_assert(i < _rows, "Out of bounds");
        //etl_assert(j < _columns, "Out of bounds");

        //return _data[i * _columns + j];
    //}

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

template<typename T>
struct dyn_matrix {
public:
    using       value_type = T;
    using     storage_impl = std::vector<value_type>;
    using         iterator = typename storage_impl::iterator;
    using   const_iterator = typename storage_impl::const_iterator;

private:
    storage_impl _data;
    const std::size_t _rows;
    const std::size_t _columns;

public:
    ///{{{ Construction

    explicit dyn_matrix(const dyn_matrix& rhs) : _data(rhs._data), _rows(rhs._rows), _columns(rhs._columns) {
        //Nothing to init
    }

    dyn_matrix(std::size_t rows, std::size_t columns) : _data(rows * columns), _rows(rows), _columns(columns) {
        //Nothing to init
    }

    dyn_matrix(std::size_t rows, std::size_t columns, const value_type& value) : _data(rows * columns), _rows(rows), _columns(columns) {
        std::fill(_data.begin(), _data.end(), value);
    }

    dyn_matrix(std::size_t rows, std::size_t columns, std::initializer_list<value_type> l) : _data(rows * columns), _rows(rows), _columns(columns) {
        etl_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    template<typename LE, typename Op, typename RE>
    explicit dyn_matrix(const binary_expr<value_type, LE, Op, RE>& e) :
            _data(etl::rows(e) * etl::columns(e)),
            _rows(etl::rows(e)),
            _columns(etl::columns(e)) {
        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }
    }

    template<typename E, typename Op>
    explicit dyn_matrix(const unary_expr<value_type, E, Op>& e) :
            _data(etl::rows(e) * etl::columns(e)),
            _rows(etl::rows(e)),
            _columns(etl::columns(e)){
        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }
    }

    template<typename E>
    explicit dyn_matrix(const transform_expr<value_type, E>& e) :
            _data(etl::rows(e) * etl::columns(e)),
            _rows(etl::rows(e)),
            _columns(etl::columns(e)){
        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }
    }

    //Default move
    dyn_matrix(dyn_matrix&& rhs) = default;

    //}}}

    //{{{ Assignment

    //Copy assignment operator

    dyn_matrix& operator=(const dyn_matrix& rhs){
        ensure_same_size(*this, rhs);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = rhs[i];
        }

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    dyn_matrix& operator=(const Container& vec){
        etl_assert(vec.size() == _rows * _columns, "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < _rows * _columns; ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    dyn_matrix& operator=(binary_expr<value_type, LE, Op, RE>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }

        return *this;
    }

    template<typename E, typename Op>
    dyn_matrix& operator=(unary_expr<value_type, E, Op>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }

        return *this;
    }

    template<typename E>
    dyn_matrix& operator=(transform_expr<value_type, E>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < rows(); ++i){
            for(std::size_t j = 0; j < columns(); ++j){
                _data[i * columns() + j] = e(i,j);
            }
        }

        return *this;
    }

    //Set the same value to each element of the matrix
    dyn_matrix& operator=(const value_type& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Default move
    dyn_matrix& operator=(dyn_matrix&& rhs) = default;

    //}}}

    //{{{ Accessors

    size_t size() const {
        return _rows * _columns;
    }

    size_t rows() const {
        return _rows;
    }

    size_t columns() const {
        return _columns;
    }

    size_t dimensions() const {
        return 2;
    }

    size_t dim(std::size_t d) const {
        etl_assert(d == 0 || d == 1, "Invalid dimension");

        return d == 0 ? _rows : _columns;
    }

    value_type& operator()(size_t i, size_t j){
        etl_assert(i < _rows, "Out of bounds");
        etl_assert(j < _columns, "Out of bounds");

        return _data[i * _columns + j];
    }

    const value_type& operator()(size_t i, size_t j) const {
        etl_assert(i < _rows, "Out of bounds");
        etl_assert(j < _columns, "Out of bounds");

        return _data[i * _columns + j];
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
