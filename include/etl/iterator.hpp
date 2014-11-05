//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_ITERATOR_HPP
#define ETL_ITERATOR_HPP

#include <iterator>

#include "traits_fwd.hpp"

namespace etl {

template<typename Expr, bool Ref = false, bool Const = true>
struct iterator : public std::iterator<std::random_access_iterator_tag, value_t<Expr>> {
private:
    Expr* expr;
    std::size_t i;

public:
    using base_iterator_t = std::iterator<std::random_access_iterator_tag, value_t<Expr>>;
    using value_type = value_t<Expr>;
    using reference_t = std::conditional_t<Ref,
          std::conditional_t<Const, const value_type&, value_type&>,
          value_type>;
    using pointer_t = std::conditional_t<Const, const value_type*, value_type*>;
    using difference_t = typename base_iterator_t::difference_type;

    iterator(Expr& expr, std::size_t i) : expr(&expr), i(i) {}

    reference_t operator*(){
        return (*expr)[i];
    }

    reference_t operator[](difference_t n){
        return (*expr)[n];
    }

    pointer_t operator->(){
        return &(*expr)[i];
    }

    iterator& operator--(){
        --i;
        return *this;
    }

    iterator operator--(int){
        iterator prev(*this);
        --i;
        return prev;
    }

    iterator& operator++(){
        ++i;
        return *this;
    }

    iterator operator++(int){
        iterator prev(*this);
        ++i;
        return prev;
    }

    iterator& operator+=(difference_t n){
        i += n;
        return *this;
    }

    iterator& operator-=(difference_t n){
        i -= n;
        return *this;
    }

    iterator operator+(difference_t n){
        iterator it(*this);
        it += n;
        return it;
    }

    iterator operator-(difference_t n){
        iterator it(*this);
        it -= n;
        return it;
    }

    difference_t operator-(const iterator& it){
        return i - it.i;
    }

    bool operator==(const iterator& other) const {
        return expr == other.expr && i == other.i;
    }

    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }
};

} //end of namespace etl

#endif
