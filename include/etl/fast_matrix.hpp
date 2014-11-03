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

#include "cpp_utils/assert.hpp"

#include "tmp.hpp"
#include "traits_fwd.hpp"

namespace etl {

namespace matrix_detail {

template<typename M, std::size_t I, typename Enable = void>
struct matrix_subsize  : std::integral_constant<std::size_t, M::template dim<I+1>() * matrix_subsize<M, I+1>::value> {};

template<typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template<typename M, std::size_t I, std::size_t Stop, typename S1, typename... S>
struct matrix_index {
    template<std::size_t I2, typename Enable = void>
    struct matrix_index_int {
        static std::size_t compute(S1 first, S... args){
            cpp_assert(first < M::template dim<I2>(), "Out of bounds");

            return matrix_subsize<M, I>::value * first + matrix_index<M, I+1, Stop, S...>::compute(args...);
        }
    };

    template<std::size_t I2>
    struct matrix_index_int<I2, std::enable_if_t<I2 == Stop>> {
        static std::size_t compute(S1 first){
            cpp_assert(first < M::template dim<I2>(), "Out of bounds");

            return first;
        }
    };

    static std::size_t compute(S1 first, S... args){
        return matrix_index_int<I>::compute(first, args...);
    }
};

} //end of namespace detail

template<typename T, std::size_t... Dims>
struct fast_matrix {
    static_assert(sizeof...(Dims) > 0, "At least one dimension must be specified");

public:
    static constexpr const std::size_t n_dimensions = sizeof...(Dims);
    static constexpr const std::size_t etl_size = mul_all<Dims...>::value;

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

    template<typename VT, cpp::enable_if_u<cpp::or_u<std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value>::value> = cpp::detail::dummy>
    explicit fast_matrix(const VT& value){
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix(std::initializer_list<value_type> l){
        cpp_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    explicit fast_matrix(const fast_matrix& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    fast_matrix(fast_matrix&& rhs){
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    template<typename E, cpp::enable_if_all_u<
        std::is_convertible<typename E::value_type, value_type>::value,
        is_copy_expr<E>::value
    > = cpp::detail::dummy>
    explicit fast_matrix(const E& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename Generator>
    explicit fast_matrix(generator_expr<Generator>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 1> = cpp::detail::dummy>
    explicit fast_matrix(const unstable_transform_expr<value_type, E>& e){
        static_assert(n_dimensions == 1, "Transform expressions are only 1D-valid for now");

        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            _data[index(i)] = e(i);
        }
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 2> = cpp::detail::dummy>
    explicit fast_matrix(const unstable_transform_expr<value_type, E>& e){
        static_assert(n_dimensions == 2, "Transform expressions are only 2D-valid for now");

        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                _data[index(i,j)] = e(i,j);
            }
        }
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 3> = cpp::detail::dummy>
    explicit fast_matrix(const unstable_transform_expr<value_type, E>& e){
        static_assert(n_dimensions == 3, "Transform expressions are only 3D-valid for now");

        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                for(std::size_t k = 0; k < dim<2>(); ++k){
                    _data[index(i,j,k)] = e(i,j,k);
                }
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

    template<typename Container, cpp::enable_if_u<std::is_convertible<typename Container::value_type, value_type>::value> = cpp::detail::dummy>
    fast_matrix& operator=(const Container& vec){
        std::copy(vec.begin(), vec.end(), begin());

        return *this;
    }

    //Construct from expression

    template<typename E, cpp::enable_if_all_u<
        std::is_convertible<typename E::value_type, value_type>::value,
        is_copy_expr<E>::value
    > = cpp::detail::dummy>
    fast_matrix& operator=(E&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename Generator>
    fast_matrix& operator=(generator_expr<Generator>&& e){
        for(std::size_t i = 0; i < size(); ++i){
            _data[i] = e[i];
        }

        return *this;
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 1> = cpp::detail::dummy>
    fast_matrix& operator=(unstable_transform_expr<value_type, E>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            _data[index(i)] = e(i);
        }

        return *this;
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 2> = cpp::detail::dummy>
    fast_matrix& operator=(unstable_transform_expr<value_type, E>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                _data[index(i,j)] = e(i,j);
            }
        }

        return *this;
    }

    template<typename E, cpp::enable_if_u<etl_traits<unstable_transform_expr<value_type, E>>::dimensions() == 3> = cpp::detail::dummy>
    fast_matrix& operator=(unstable_transform_expr<value_type, E>&& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < dim<0>(); ++i){
            for(std::size_t j = 0; j < dim<1>(); ++j){
                for(std::size_t k = 0; k < dim<2>(); ++k){
                    _data[index(i,j,k)] = e(i,j,k);
                }
            }
        }

        return *this;
    }

    //Set the same value to each element of the matrix
    template<typename VT, cpp::enable_if_u<cpp::or_u<std::is_convertible<VT, value_type>::value, std::is_assignable<T&, VT>::value>::value> = cpp::detail::dummy>
    fast_matrix& operator=(const VT& value){
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Prohibit move
    fast_matrix& operator=(fast_matrix&& rhs) = delete;

    //}}}

    //{{{ Accessors

    static constexpr std::size_t size(){
        return etl_size;
    }

    static constexpr std::size_t rows(){
        return dim<0>();
    }

    static constexpr std::size_t columns(){
        static_assert(n_dimensions > 1, "columns() can only be used on 2D+ matrices");

        return dim<1>();
    }

    static constexpr std::size_t dimensions(){
        return n_dimensions;
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return nth_size<D, 0, Dims...>::value;
    }

    //TODO Would probably be useful to have dim(std::size_t i)

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), value_type&> operator()(S... args){
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), const value_type&> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    const value_type& operator[](std::size_t i) const {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    value_type& operator[](std::size_t i){
        cpp_assert(i < size(), "Out of bounds");

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
