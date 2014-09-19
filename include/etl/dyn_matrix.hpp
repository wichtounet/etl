//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_old_matrix_HPP
#define ETL_old_matrix_HPP

#include<vector>
#include<tuple>

#include "assert.hpp"
#include "tmp.hpp"
#include "fast_op.hpp"
#include "fast_expr.hpp"

namespace etl {

enum class init_flag_t { DUMMY };
constexpr const init_flag_t init_flag = init_flag_t::DUMMY;

template<typename... V>
struct values_t {
    const std::tuple<V...> values;
    values_t(V... v) : values(v...) {};

    template<typename T, std::size_t... I>
    std::vector<T> list_sub(const index_sequence<I...>& /*i*/) const {
        return {static_cast<T>(std::get<I>(values))...};
    }

    template<typename T>
    std::vector<T> list() const {
        return list_sub<T>(make_index_sequence<sizeof...(V)>());
    }
};

template<typename... V>
values_t<V...> values(V... v){
    return {v...};
}

namespace dyn_detail {

template<typename... S>
struct is_init_constructor : std::integral_constant<bool, false> {};

template<typename S1, typename S2, typename S3, typename... S>
struct is_init_constructor<S1, S2, S3, S...> :
    std::integral_constant<bool, std::is_same<init_flag_t, typename nth_type<1+sizeof...(S), S1, S2, S3, S...>::type>::value> {};

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

template<typename T>
struct dyn_matrix {
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
    //{{{ Construction

    explicit dyn_matrix(const dyn_matrix& rhs) : _size(rhs._size), _data(rhs._data), _dimensions(rhs._dimensions) {
        //Nothing to init
    }

    //Normal constructor with only sizes
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 0),
            all_convertible_to<std::size_t, S...>::value,
            is_homogeneous<typename first_type<S...>::type, S...>::value
        >::value> = detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(sizes...)),
            _data(_size),
            _dimensions({static_cast<std::size_t>(sizes)...}) {
        //Nothing to init
    }

    //Sizes followed by an initializer list
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 1),
            is_specialization_of<std::initializer_list, typename last_type<S...>::type>::value
        >::value> = detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
            _data(last_value(sizes...)),
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        //Nothing to init
    }

    //Sizes followed by a values_t
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 1),
            is_specialization_of<values_t, typename last_type<S...>::type>::value
        >::value> = detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
            _data(last_value(sizes...).template list<value_type>()),
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        //Nothing to init
    }

    //Sizes followed by a value
    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 1),
            std::is_convertible<std::size_t, typename first_type<S...>::type>::value,                           //The first type must be convertible to size_t
            is_sub_homogeneous<typename first_type<S...>::type, S...>::value,                                   //The first N-1 types must homegeneous
            std::is_same<value_type, typename last_type<S...>::type>::value,                                    //The last type must be exactly value_type
            not_u<std::is_same<typename first_type<S...>::type, typename last_type<S...>::type>::value>::value  //The first and last types must be different
        >::value> = detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-1)>(), sizes...)),
            _data(_size, last_value(sizes...)),
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-1)>(), sizes...)) {
        //Nothing to init
    }

    //Sizes followed by an init flag followed by the value
    template<typename... S, enable_if_u<dyn_detail::is_init_constructor<S...>::value> = detail::dummy>
    dyn_matrix(S... sizes) :
            _size(dyn_detail::size(make_index_sequence<(sizeof...(S)-2)>(), sizes...)),
            _data(_size, last_value(sizes...)),
            _dimensions(dyn_detail::sizes(make_index_sequence<(sizeof...(S)-2)>(), sizes...)) {
        //Nothing to init
    }

    template<typename LE, typename Op, typename RE>
    explicit dyn_matrix(const binary_expr<value_type, LE, Op, RE>& e) :
            _size(etl::size(e)),
            _data(_size),
            _dimensions(etl::dimensions(e)) {

        for(std::size_t d = 0; d < etl::dimensions(e); ++d){
            _dimensions[d] = etl::dim(e, d);
        }

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    explicit dyn_matrix(const unary_expr<value_type, E, Op>& e) :
            _size(etl::size(e)),
            _data(_size),
            _dimensions(etl::dimensions(e)) {

        for(std::size_t d = 0; d < etl::dimensions(e); ++d){
            _dimensions[d] = etl::dim(e, d);
        }

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E>
    explicit dyn_matrix(const transform_expr<value_type, E>& e) :
            _size(etl::size(e)),
            _data(_size),
            _dimensions(etl::dimensions(e)) {

        for(std::size_t d = 0; d < etl::dimensions(e); ++d){
            _dimensions[d] = etl::dim(e, d);
        }

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
        etl_assert(vec.size() == size(), "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = vec[i];
        }

        return *this;
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    dyn_matrix& operator=(binary_expr<value_type, LE, Op, RE>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    dyn_matrix& operator=(unary_expr<value_type, E, Op>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
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

    value_type& operator()(size_t i, size_t j){
        etl_assert(i < dim(0), "Out of bounds");
        etl_assert(j < dim(1), "Out of bounds");
        etl_assert(_dimensions.size() == 2, "Invalid number of parameters");

        return _data[i * dim(1) + j];
    }

    const value_type& operator()(size_t i, size_t j) const {
        etl_assert(i < dim(0), "Out of bounds");
        etl_assert(j < dim(1), "Out of bounds");
        etl_assert(dimensions() == 2, "Invalid number of parameters");

        return _data[i * dim(1) + j];
    }

    template<typename... S, enable_if_u<(sizeof...(S) > 0)> = detail::dummy>
    std::size_t index(S... sizes) const {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        auto subsize = size() / dim(0);
        std::size_t index = 0;
        std::size_t i = 0;

        for_each_in(
            [&subsize, &index, &i, this](std::size_t s){
                index += subsize * s;
                subsize /= dim(++i);
            }, sizes...);

        return _data[index];
    }

    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 2),
            all_convertible_to<std::size_t, S...>::value
        >::value> = detail::dummy>
    const value_type& operator()(S... sizes) const {
        etl_assert(sizeof...(S) == dimensions(), "Invalid number of parameters");

        return _data[index(sizes...)];
    }

    template<typename... S, enable_if_u<
        and_u<
            (sizeof...(S) > 2),
            all_convertible_to<std::size_t, S...>::value
        >::value> = detail::dummy>
    value_type& operator()(S... sizes){
        etl_assert(sizeof...(S) == dimensions(), "Invalid number of parameters");

        return _data[index(sizes...)];
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
