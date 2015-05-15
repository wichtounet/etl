//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_HPP
#define ETL_FAST_HPP

#include <algorithm>    //For std::find_if
#include <iostream>     //For stream support

#include "cpp_utils/assert.hpp"

#include "tmp.hpp"
#include "evaluator.hpp"
#include "traits_lite.hpp"          //forward declaration of the traits
#include "compat.hpp"               //To make it work with g++

// CRTP classes
#include "crtp/inplace_assignable.hpp"
#include "crtp/comparable.hpp"
#include "crtp/iterable.hpp"
#include "crtp/expression_able.hpp"

namespace etl {

namespace matrix_detail {

template<typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I+1>() * matrix_subsize<M, I+1>::value> {};

template<typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template<typename M, std::size_t I, typename Enable = void>
struct matrix_leadingsize : std::integral_constant<std::size_t, M::template dim<I-1>() * matrix_leadingsize<M, I-1>::value> {};

template<typename M>
struct matrix_leadingsize<M, 0> : std::integral_constant<std::size_t, 1> {};

template<typename M, std::size_t I, typename S1>
inline constexpr std::size_t rm_compute_index(S1 first) noexcept {
    return first;
}

template<typename M, std::size_t I, typename S1, typename... S, cpp_enable_if((sizeof...(S) > 0))>
inline constexpr std::size_t rm_compute_index(S1 first, S... args) noexcept {
    return matrix_subsize<M, I>::value * first + rm_compute_index<M, I+1>(args...);
}

template<typename M, std::size_t I, typename S1>
inline constexpr std::size_t cm_compute_index(S1 first) noexcept {
    return matrix_leadingsize<M, I>::value * first;
}

template<typename M, std::size_t I, typename S1, typename... S, cpp_enable_if((sizeof...(S) > 0))>
inline constexpr std::size_t cm_compute_index(S1 first, S... args) noexcept {
    return matrix_leadingsize<M, I>::value * first + cm_compute_index<M, I+1>(args...);
}

template<typename M, std::size_t I, typename... S, cpp_enable_if(M::storage_order == order::RowMajor)>
inline constexpr std::size_t compute_index(S... args) noexcept {
    return rm_compute_index<M, I>(args...);
}

template<typename M, std::size_t I, typename... S, cpp_enable_if(M::storage_order == order::ColumnMajor)>
inline constexpr std::size_t compute_index(S... args) noexcept {
    return cm_compute_index<M, I>(args...);
}

template <typename N>
struct is_vector : std::false_type { };

template <typename N, typename A>
struct is_vector<std::vector<N, A>> : std::true_type { };

template <typename N>
struct is_vector<std::vector<N>> : std::true_type { };

} //end of namespace matrix_detail

template<typename T, typename ST, order SO, std::size_t... Dims>
struct fast_matrix_impl final :
        inplace_assignable<fast_matrix_impl<T, ST, SO, Dims...>>,
        comparable<fast_matrix_impl<T, ST, SO, Dims...>>,
        expression_able<fast_matrix_impl<T, ST, SO, Dims...>>,
        iterable<fast_matrix_impl<T, ST, SO, Dims...>> {
    static_assert(sizeof...(Dims) > 0, "At least one dimension must be specified");

public:
    static constexpr const std::size_t n_dimensions = sizeof...(Dims);
    static constexpr const std::size_t etl_size = mul_all<Dims...>::value;
    static constexpr const order storage_order = SO;
    static constexpr const bool array_impl = !matrix_detail::is_vector<ST>::value;

    using        value_type = T;
    using      storage_impl = ST;
    using          iterator = typename storage_impl::iterator;
    using    const_iterator = typename storage_impl::const_iterator;
    using         this_type = fast_matrix_impl<T, ST, SO, Dims...>;
    using       memory_type = value_type*;
    using const_memory_type = const value_type*;
    using          vec_type = intrinsic_type<T>;

private:
    storage_impl _data;

    template<typename... S>
    static constexpr std::size_t index(S... args){
        return matrix_detail::compute_index<this_type, 0>(args...);
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

    template<typename S = ST, cpp::enable_if_c<matrix_detail::is_vector<S>> = cpp::detail::dummy>
    void init(){
        _data.resize(etl_size);
    }

    template<typename S = ST, cpp::disable_if_c<matrix_detail::is_vector<S>> = cpp::detail::dummy>
    void init() noexcept {
        //Nothing to init
    }

    fast_matrix_impl() noexcept(array_impl) {
        init();
    }

    template<typename VT, cpp::enable_if_one_c<std::is_convertible<VT, value_type>, std::is_assignable<T&, VT>> = cpp::detail::dummy>
    explicit fast_matrix_impl(const VT& value){
        init();
        std::fill(_data.begin(), _data.end(), value);
    }

    fast_matrix_impl(std::initializer_list<value_type> l){
        init();

        cpp_assert(l.size() == size(), "Cannot copy from an initializer of different size");

        std::copy(l.begin(), l.end(), begin());
    }

    fast_matrix_impl(const fast_matrix_impl& rhs) noexcept(array_impl) {
        init();
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    template<typename SST = ST, cpp_enable_if(matrix_detail::is_vector<SST>::value)>
    fast_matrix_impl(fast_matrix_impl&& rhs) noexcept {
        _data = std::move(rhs._data);
    }

    template<typename SST = ST, cpp_disable_if(matrix_detail::is_vector<SST>::value)>
    explicit fast_matrix_impl(fast_matrix_impl&& rhs) noexcept {
        std::copy(rhs.begin(), rhs.end(), begin());
    }

    template<typename E, cpp_enable_if(std::is_convertible<value_t<E>, value_type>::value, is_copy_expr<E>::value)>
    explicit fast_matrix_impl(E&& e){
        init();
        ensure_same_size(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
    }

    template<typename Container, cpp_enable_if(
            std::is_convertible<typename Container::value_type, value_type>::value,
            cpp::not_c<is_copy_expr<Container>>::value
        )>
    explicit fast_matrix_impl(const Container& vec){
        init();
        cpp_assert(vec.size() == size(), "Cannnot copy from a container of another size");

        std::copy(vec.begin(), vec.end(), begin());
    }

    template<typename Generator>
    explicit fast_matrix_impl(generator_expr<Generator>&& e){
        init();
        assign_evaluate(e, *this);
    }

    //}}}

    //{{{ Assignment

    //Copy assignment operator

    template<typename SST = ST, cpp_enable_if(matrix_detail::is_vector<SST>::value)>
    fast_matrix_impl& operator=(const fast_matrix_impl& rhs) noexcept {
        if(this != &rhs){
            _data = rhs._data;
        }
        return *this;
    }

    template<typename SST = ST, cpp_disable_if(matrix_detail::is_vector<SST>::value)>
    fast_matrix_impl& operator=(const fast_matrix_impl& rhs) noexcept {
        if(this != &rhs){
            std::copy(rhs.begin(), rhs.end(), begin());
        }
        return *this;
    }

    template<std::size_t... SDims>
    fast_matrix_impl& operator=(const fast_matrix_impl<T, ST, SO, SDims...>& rhs) noexcept {
        ensure_same_size(*this, rhs);
        _data = rhs._data;
        return *this;
    }

    //Allow copy from other containers

    template<typename Container, cpp::enable_if_c<std::is_convertible<typename Container::value_type, value_type>> = cpp::detail::dummy>
    fast_matrix_impl& operator=(const Container& vec) noexcept {
        std::copy(vec.begin(), vec.end(), begin());

        return *this;
    }

    //Construct from expression

    template<typename E, cpp_enable_if(std::is_convertible<typename E::value_type, value_type>::value && is_copy_expr<E>::value)>
    fast_matrix_impl& operator=(E&& e){
        ensure_same_size(*this, e);

        assign_evaluate(std::forward<E>(e), *this);

        return *this;
    }

    template<typename Generator>
    fast_matrix_impl& operator=(generator_expr<Generator>&& e){
        assign_evaluate(e, *this);

        return *this;
    }

    //Set the same value to each element of the matrix
    template<typename VT, cpp::enable_if_one_c<std::is_convertible<VT, value_type>, std::is_assignable<T&, VT>> = cpp::detail::dummy>
    fast_matrix_impl& operator=(const VT& value) noexcept {
        std::fill(_data.begin(), _data.end(), value);

        return *this;
    }

    //Prohibit move

    template<typename SST = ST, cpp_enable_if(matrix_detail::is_vector<SST>::value)>
    fast_matrix_impl& operator=(fast_matrix_impl&& rhs){
        if(this != &rhs){
            _data = std::move(rhs._data);
        }
        return *this;
    }

    template<typename SST = ST, cpp_disable_if(matrix_detail::is_vector<SST>::value)>
    fast_matrix_impl& operator=(fast_matrix_impl&& rhs) = delete;

    //}}}

    //{{{ Swap operations

    void swap(fast_matrix_impl& other){
        //TODO Ensure dimensions...
        using std::swap;
        swap(_data, other._data);
    }

    //}}}

    //{{{ Accessors

    static constexpr std::size_t size() noexcept {
        return etl_size;
    }

    static constexpr std::size_t rows() noexcept {
        return dim<0>();
    }

    static constexpr std::size_t columns() noexcept {
        static_assert(n_dimensions > 1, "columns() can only be used on 2D+ matrices");

        return dim<1>();
    }

    static constexpr std::size_t dimensions() noexcept {
        return n_dimensions;
    }

    template<std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return nth_size<D, 0, Dims...>::value;
    }

    std::size_t dim(std::size_t d) noexcept {
        return dyn_nth_size<Dims...>(d);
    }

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) noexcept {
        return sub(*this, i);
    }

    template<bool B = (n_dimensions > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const noexcept {
        return sub(*this, i);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), value_type&> operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), const value_type&> operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return access(static_cast<std::size_t>(args)...);
    }

    const value_type& operator[](std::size_t i) const noexcept {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    value_type& operator[](std::size_t i) noexcept {
        cpp_assert(i < size(), "Out of bounds");

        return _data[i];
    }

    vec_type load(std::size_t i) const noexcept {
        return vec::loadu(memory_start() + i);
    }

    iterator begin() noexcept(noexcept(_data.begin())) {
        return _data.begin();
    }

    iterator end() noexcept(noexcept(_data.end())) {
        return _data.end();
    }

    const_iterator begin() const noexcept(noexcept(_data.cbegin())) {
        return _data.cbegin();
    }

    const_iterator end() const noexcept(noexcept(_data.end())) {
        return _data.cend();
    }

    const_iterator cbegin() const noexcept(noexcept(_data.cbegin())) {
        return _data.cbegin();
    }

    const_iterator cend() const noexcept(noexcept(_data.end())) {
        return _data.cend();
    }

    //}}}

    //{{{ Direct memory access

    memory_type memory_start() noexcept {
        return &_data[0];
    }

    const_memory_type memory_start() const noexcept {
        return &_data[0];
    }

    memory_type memory_end() noexcept {
        return &_data[size()];
    }

    const_memory_type memory_end() const noexcept {
        return &_data[size()];
    }

    //}}}
};

template<typename T, typename ST, order SO, std::size_t... Dims>
void swap(fast_matrix_impl<T, ST, SO, Dims...>& lhs, fast_matrix_impl<T, ST, SO, Dims...>& rhs){
    lhs.swap(rhs);
}

template<typename T, typename ST, order SO, std::size_t... Dims>
std::ostream& operator<<(std::ostream& os, const fast_matrix_impl<T, ST, SO, Dims...>& /*matrix*/){
    if(sizeof...(Dims) == 1){
        return os << "V[" << concat_sizes(Dims...) << "]";
    } else {
        return os << "M[" << concat_sizes(Dims...) << "]";
    }
}

} //end of namespace etl

#endif
