//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CHECKS_HPP
#define ETL_CHECKS_HPP

namespace etl {

template<typename... E>
struct all_etl_expr : cpp::and_c<is_etl_expr<E>...> {};

template<typename LE, typename RE, cpp_enable_if(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator)>
void ensure_same_size(const LE&, const RE&) noexcept {
    //Nothing to test, generators are of infinite size
}

template<typename LE, typename RE, cpp_enable_if(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE>::value && !(etl_traits<LE>::is_fast && etl_traits<RE>::is_fast))>
void ensure_same_size(const LE& lhs, const RE& rhs){
    cpp_assert(size(lhs) == size(rhs), "Cannot perform element-wise operations on collections of different size");
    cpp_unused(lhs);
    cpp_unused(rhs);
}

template<typename LE, typename RE, cpp_enable_if(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE>::value && etl_traits<LE>::is_fast && etl_traits<RE>::is_fast)>
void ensure_same_size(const LE&, const RE&){
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

} //end of namespace etl

#endif
