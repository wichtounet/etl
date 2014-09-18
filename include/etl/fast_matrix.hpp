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

namespace etl {

namespace matrix_detail {

template<size_t F, size_t... Dims>
struct matrix_size  : std::integral_constant<std::size_t, F * matrix_size<Dims...>::value> {};

template<size_t F>
struct matrix_size<F> : std::integral_constant<std::size_t, F> {};

template<typename M, size_t I, typename Enable = void>
struct matrix_subsize  : std::integral_constant<std::size_t, M::template dim<I+1>() * matrix_subsize<M, I+1>::value> {};

template<typename M, size_t I>
struct matrix_subsize<M, I, enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template<typename F, typename... S>
struct valid_sizes  : std::integral_constant<bool, and_u<valid_sizes<F>::value, valid_sizes<S...>::value>::value> {};

template<typename F>
struct valid_sizes<F> : std::integral_constant<bool, std::is_convertible<F, std::size_t>::value> {};

template<size_t S, size_t I, size_t F, size_t... Dims>
struct matrix_dimension {
    template<size_t S2, size_t I2, typename Enable = void>
    struct matrix_dimension_int : std::integral_constant<std::size_t, matrix_dimension<S, I+1, Dims...>::value> {};

    template<size_t S2, size_t I2>
    struct matrix_dimension_int<S2, I2, enable_if_t<S2 == I2>> : std::integral_constant<std::size_t, F> {};

    static constexpr const std::size_t value = matrix_dimension_int<S, I>::value;
};

template<typename M, size_t I, size_t Stop, typename S1, typename... S>
struct matrix_index {
    template<size_t I2, typename Enable = void>
    struct matrix_index_int {
        static size_t compute(S1 first, S... args){
            etl_assert(first < M::template dim<I2>(), "Out of bounds");

            return matrix_subsize<M, I>::value * first + matrix_index<M, I+1, Stop, S...>::compute(args...);
        }
    };

    template<size_t I2>
    struct matrix_index_int<I2, enable_if_t<I2 == Stop>> {
        static size_t compute(S1 first){
            etl_assert(first < M::template dim<I2>(), "Out of bounds");

            return first;
        }
    };

    static size_t compute(S1 first, S... args){
        return matrix_index_int<I>::compute(first, args...);
    }
};

} //end of namespace detail

template<typename T, size_t... Dims>
struct fast_matrix {
    static_assert(sizeof...(Dims) > 1, "At least two dimension must be specified");

public:
    static constexpr const std::size_t n_dimensions = sizeof...(Dims);
    static constexpr const std::size_t etl_size = matrix_detail::matrix_size<Dims...>::value;

    using       value_type = T;
    using     storage_impl = std::array<value_type, etl_size>;
    using         iterator = typename storage_impl::iterator;
    using   const_iterator = typename storage_impl::const_iterator;
    using        this_type = fast_matrix<T, Dims...>;

private:
    storage_impl _data;

    template<typename... S>
    static constexpr std::size_t index(S... args){
        return matrix_detail::matrix_index<this_type, 0, n_dimensions - 1, S...>::compute(args...);
    }

    template<typename... S>
    value_type& access(S... args){
        return _data[index(args...)];
    }

    template<typename... S>
    const value_type& access(S... args) const {
        return _data[index(args...)];
    }

public:

    ///{{{ Construction

    fast_matrix(){
        //Nothing to init
    }

    template<typename VT, enable_if_u<or_u<std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value>::value> = detail::dummy>
    explicit fast_matrix(const VT& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix(std::initializer_list<value_type> l){
        etl_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    explicit fast_matrix(const fast_matrix& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    fast_matrix(fast_matrix&& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    template<typename LE, typename Op, typename RE>
    explicit fast_matrix(const binary_expr<value_type, LE, Op, RE>& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, typename Op>
    explicit fast_matrix(const unary_expr<value_type, E, Op>& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E>
    explicit fast_matrix(const transform_expr<value_type, E>& e){
        //TODO This will only support 2D Expressions

        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                _data[index(i,j)] = e(i,j);
            }
        }
    }

    //}}}

    //{{{ Assignment

    //Copy assignment operator

    fast_matrix& operator=(const fast_matrix& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());

        return *this;
    }

    //Allow copy from other containers

    template<typename Container, enable_if_u<std::is_same<typename Container::value_type, value_type>::value> = detail::dummy>
    fast_matrix& operator=(const Container& vec){
        std::copy(vec.begin(), vec.end(), begin());

        return *this;
    }

    //Construct from expression

    template<typename LE, typename Op, typename RE>
    fast_matrix& operator=(binary_expr<value_type, LE, Op, RE>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, typename Op>
    fast_matrix& operator=(unary_expr<value_type, E, Op>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E>
    fast_matrix& operator=(transform_expr<value_type, E>&& e){
        //TODO This will only support 2D Expressions

        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                _data[index(i,j)] = e(i,j);
            }
        }

        return *this;
    }

    //Set the same value to each element of the matrix
    template<typename VT, enable_if_u<or_u<std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value>::value> = detail::dummy>
    fast_matrix& operator=(const VT& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Prohibit move
    fast_matrix& operator=(fast_matrix&& rhs) = delete;

    //}}}

    //{{{ Accessors

    static constexpr size_t size(){
        return etl_size;
    }

    static constexpr size_t rows(){
        return dim<0>();
    }

    static constexpr enable_if_t<(sizeof...(Dims) > 1), std::size_t> columns(){
        return dim<1>();
    }

    static constexpr size_t dimensions(){
        return n_dimensions;
    }

    template<size_t D>
    static constexpr size_t dim(){
        return matrix_detail::matrix_dimension<D, 0, Dims...>::value;
    }

    //TODO Would probably be useful to have dim(size_t i)

    template<typename... S>
    value_type& operator()(S... args){
        static_assert(sizeof...(S) == sizeof...(Dims), "Invalid number of parameters");
        static_assert(matrix_detail::valid_sizes<S...>::value, "Invalid size types");

        return access(static_cast<size_t>(args)...);
    }

    template<typename... S>
    const value_type& operator()(S... args) const {
        static_assert(sizeof...(S) == sizeof...(Dims), "Invalid number of parameters");
        static_assert(matrix_detail::valid_sizes<S...>::value, "Invalid size types");

        return access(static_cast<size_t>(args)...);
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
