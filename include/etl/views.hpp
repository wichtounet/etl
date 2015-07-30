//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd>     //For stream support

#include "cpp_utils/tmp.hpp"
#include "tmp.hpp"
#include "traits_lite.hpp"

namespace etl {

template<typename T, typename S>
using return_helper =
    std::conditional_t<
        std::is_const<std::remove_reference_t<S>>::value,
        const value_t<T>&,
        std::conditional_t<
            cpp::and_u<
                std::is_lvalue_reference<S>::value,
                cpp::not_u<std::is_const<T>::value>::value
            >::value,
            value_t<T>&,
            value_t<T>>>;

template<typename T, typename S>
using const_return_helper = std::conditional_t<
        std::is_lvalue_reference<S>::value,
        const value_t<T>&,
        value_t<T>>;

template<typename T, std::size_t D>
struct dim_view {
    T sub;
    const std::size_t i;

    static_assert(D == 1 || D == 2, "Invalid dimension");

    using          sub_type = T;
    using        value_type = value_t<sub_type>;
    using       memory_type = memory_t<sub_type>;
    using const_memory_type = std::add_const_t<memory_t<sub_type>>;
    using       return_type = return_helper<sub_type, decltype(sub(0,0))>;
    using const_return_type = const_return_helper<sub_type, decltype(sub(0,0))>;

    dim_view(sub_type sub, std::size_t i) : sub(sub), i(i) {}

    const_return_type operator[](std::size_t j) const {
        if(D == 1){
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    return_type operator[](std::size_t j){
        if(D == 1){
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    const_return_type operator()(std::size_t j) const {
        if(D == 1){
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    return_type operator()(std::size_t j){
        if(D == 1){
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    sub_type& value(){
        return sub;
    }

    //{{{ Direct memory access

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    memory_type memory_start() noexcept {
        return sub.memory_start() + i * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    const_memory_type memory_start() const noexcept {
        return sub.memory_start() + i * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    memory_type memory_end() noexcept {
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    const_memory_type memory_end() const noexcept {
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    //}}}
};

template<typename T>
struct sub_view {
    T parent;
    const std::size_t i;

    using       parent_type = T;
    using        value_type = value_t<parent_type>;
    using       memory_type = memory_t<parent_type>;
    using const_memory_type = std::add_const_t<memory_t<parent_type>>;
    using       return_type = return_helper<parent_type, decltype(parent[0])>;
    using const_return_type = const_return_helper<parent_type, decltype(parent[0])>;

    sub_view(parent_type parent, std::size_t i) : parent(parent), i(i) {}

    const_return_type operator[](std::size_t j) const {
        if(decay_traits<parent_type>::storage_order == order::RowMajor){
            return parent[i * subsize(parent) + j];
        } else {
            return parent[i + dim<0>(parent) * j];
        }
    }

    return_type operator[](std::size_t j){
        if(decay_traits<parent_type>::storage_order == order::RowMajor){
            return parent[i * subsize(parent) + j];
        } else {
            return parent[i + dim<0>(parent) * j];
        }
    }

    template<typename... S>
    const_return_type operator()(S... args) const {
        return parent(i, static_cast<std::size_t>(args)...);
    }

    template<typename... S>
    return_type operator()(S... args){
        return parent(i, static_cast<std::size_t>(args)...);
    }

    parent_type& value(){
        return parent;
    }

    //{{{ Direct memory access

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value && decay_traits<parent_type>::storage_order == order::RowMajor)>
    memory_type memory_start() noexcept {
        return parent.memory_start() + i * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value && decay_traits<parent_type>::storage_order == order::RowMajor)>
    const_memory_type memory_start() const noexcept {
        return parent.memory_start() + i * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value && decay_traits<parent_type>::storage_order == order::RowMajor)>
    memory_type memory_end() noexcept {
        return parent.memory_start() + (i + 1) * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value && decay_traits<parent_type>::storage_order == order::RowMajor)>
    const_memory_type memory_end() const noexcept {
        return parent.memory_start() + (i + 1) * subsize(parent);
    }

    //}}}
};

namespace fast_matrix_view_detail {

template<typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I+1>() * matrix_subsize<M, I+1>::value> {};

template<typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template<typename M, std::size_t I, typename S1>
inline constexpr std::size_t compute_index(S1 first) noexcept {
    return first;
}

template<typename M, std::size_t I, typename S1, typename... S, cpp_enable_if((sizeof...(S) > 0))>
inline constexpr std::size_t compute_index(S1 first, S... args) noexcept {
    return matrix_subsize<M, I>::value * first + compute_index<M, I+1>(args...);
}

} //end of namespace fast_matrix_view_detail

template<typename T, std::size_t... Dims>
struct fast_matrix_view {
    T sub;

    using          sub_type = T;
    using        value_type = value_t<sub_type>;
    using       memory_type = memory_t<sub_type>;
    using const_memory_type = std::add_const_t<memory_t<sub_type>>;
    using       return_type = return_helper<sub_type, decltype(sub[0])>;
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>;

    static constexpr std::size_t n_dimensions = sizeof...(Dims);

    explicit fast_matrix_view(sub_type sub) : sub(sub) {}

    template<typename... S>
    static constexpr std::size_t index(S... args){
        return fast_matrix_view_detail::compute_index<fast_matrix_view<T, Dims...>, 0>(args...);
    }

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    return_type operator[](std::size_t j){
        return sub[j];
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), return_type&> operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sizeof...(Dims), const_return_type&> operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    sub_type& value(){
        return sub;
    }

    template<std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return nth_size<D, 0, Dims...>::value;
    }

    //{{{ Direct memory access

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_start() noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_start() const noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_end() noexcept {
        return sub.memory_end();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_end() const noexcept {
        return sub.memory_end();
    }

    //}}}
};

template<typename T>
struct dyn_matrix_view {
    T sub;
    std::size_t rows;
    std::size_t columns;

    using          sub_type = T;
    using        value_type = value_t<sub_type>;
    using       memory_type = memory_t<sub_type>;
    using const_memory_type = std::add_const_t<memory_t<sub_type>>;
    using       return_type = return_helper<sub_type, decltype(sub(0))>;
    using const_return_type = const_return_helper<sub_type, decltype(sub(0))>;

    dyn_matrix_view(sub_type sub, std::size_t rows, std::size_t columns) : sub(sub), rows(rows), columns(columns) {}

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t i, std::size_t j) const {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[i * columns + j];
        } else {
            return sub[i + rows * j];
        }
    }

    return_type operator[](std::size_t j){
        return sub[j];
    }

    return_type operator()(std::size_t j){
        return sub[j];
    }

    return_type operator()(std::size_t i, std::size_t j){
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[i * columns + j];
        } else {
            return sub[i + rows * j];
        }
    }

    sub_type& value(){
        return sub;
    }

    //{{{ Direct memory access

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_start() noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_start() const noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_end() noexcept {
        return sub.memory_end();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_end() const noexcept {
        return sub.memory_end();
    }

    //}}}
};

template<typename T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dim_view<T, D>& v){
    return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const sub_view<T>& v){
    return os << "sub(" << v.parent << ", " << v.i << ")";
}

template<typename T, std::size_t Rows, std::size_t Columns>
std::ostream& operator<<(std::ostream& os, const fast_matrix_view<T, Rows, Columns>& v){
    return os << "reshape[" << Rows << "," << Columns << "](" << v.sub << ")";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_view<T>& v){
    return os << "reshape[" << v.rows << "," << v.columns << "](" << v.sub << ")";
}

} //end of namespace etl
