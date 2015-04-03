//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_VIEWS_HPP
#define ETL_VIEWS_HPP

#include <iostream>     //For stream support

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
    static_assert(D == 1 || D == 2, "Invalid dimension");

    using sub_type = T;
    using value_type = value_t<sub_type>;

    sub_type sub;
    const std::size_t i;

    using return_type = return_helper<sub_type, decltype(sub(0,0))>;
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
    value_type* memory_start() noexcept {
        return sub.memory_start() + i * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    const value_type* memory_start() const noexcept {
        return sub.memory_start() + i * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    value_type* memory_end() noexcept {
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    template<typename ST = T, std::size_t SD = D, cpp_enable_if(has_direct_access<ST>::value && SD == 1)>
    const value_type* memory_end() const noexcept {
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    //}}}
};

template<typename T>
struct sub_view {
    using parent_type = T;
    using value_type = value_t<parent_type>;

    parent_type parent;
    const std::size_t i;

    using return_type = return_helper<parent_type, decltype(parent[0])>;
    using const_return_type = const_return_helper<parent_type, decltype(parent[0])>;

    sub_view(parent_type parent, std::size_t i) : parent(parent), i(i) {}

    const_return_type operator[](std::size_t j) const {
        return parent[i * subsize(parent) + j];
    }

    template<typename... S>
    const_return_type operator()(S... args) const {
        return parent(i, static_cast<std::size_t>(args)...);
    }

    return_type operator[](std::size_t j){
        return parent[i * subsize(parent) + j];
    }

    template<typename... S>
    return_type operator()(S... args){
        return parent(i, static_cast<std::size_t>(args)...);
    }

    parent_type& value(){
        return parent;
    }

    //{{{ Direct memory access

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value)>
    value_type* memory_start() noexcept {
        return parent.memory_start() + i * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value)>
    const value_type* memory_start() const noexcept {
        return parent.memory_start() + i * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value)>
    value_type* memory_end() noexcept {
        return parent.memory_start() + (i + 1) * subsize(parent);
    }

    template<typename ST = T, cpp_enable_if(has_direct_access<ST>::value)>
    const value_type* memory_end() const noexcept {
        return parent.memory_start() + (i + 1) * subsize(parent);
    }

    //}}}
};

template<typename T, std::size_t Rows, std::size_t Columns>
struct fast_matrix_view {
    static_assert(Rows > 0 && Columns > 0 , "Invalid dimensions");

    using sub_type = T;
    using value_type = value_t<sub_type>;

    sub_type sub;

    using return_type = return_helper<sub_type, decltype(sub(0))>;
    using const_return_type = const_return_helper<sub_type, decltype(sub(0))>;

    explicit fast_matrix_view(sub_type sub) : sub(sub) {}

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t j) const {
        return sub(j);
    }

    const_return_type operator()(std::size_t i, std::size_t j) const {
        return sub[i * Columns + j];
    }

    return_type operator[](std::size_t j){
        return sub[j];
    }

    return_type operator()(std::size_t j){
        return sub(j);
    }

    return_type operator()(std::size_t i, std::size_t j){
        return sub[i * Columns + j];
    }
    
    sub_type& value(){
        return sub;
    }

    //{{{ Direct memory access

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    value_type* memory_start() noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const value_type* memory_start() const noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    value_type* memory_end() noexcept {
        return sub.memory_end();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const value_type* memory_end() const noexcept {
        return sub.memory_end();
    }

    //}}}
};

template<typename T>
struct dyn_matrix_view {
    using sub_type = T;
    using value_type = value_t<sub_type>;

    sub_type sub;
    std::size_t rows;
    std::size_t columns;

    using return_type = return_helper<sub_type, decltype(sub(0))>;
    using const_return_type = const_return_helper<sub_type, decltype(sub(0))>;

    dyn_matrix_view(sub_type sub, std::size_t rows, std::size_t columns) : sub(sub), rows(rows), columns(columns) {}

    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t j) const {
        return sub[j];
    }

    const_return_type operator()(std::size_t i, std::size_t j) const {
        return sub[i * columns + j];
    }

    return_type operator[](std::size_t j){
        return sub[j];
    }

    return_type operator()(std::size_t j){
        return sub[j];
    }

    return_type operator()(std::size_t i, std::size_t j){
        return sub[i * columns + j];
    }
    
    sub_type& value(){
        return sub;
    }

    //{{{ Direct memory access

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    value_type* memory_start() noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const value_type* memory_start() const noexcept {
        return sub.memory_start();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    value_type* memory_end() noexcept {
        return sub.memory_end();
    }

    template<typename SS = T, cpp_enable_if(has_direct_access<SS>::value)>
    const value_type* memory_end() const noexcept {
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

#endif
