//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CHECKS_HPP
#define ETL_CHECKS_HPP

namespace etl {

template<typename LE, typename RE, cpp_enable_if(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator)>
void ensure_same_size(const LE& /*unused*/, const RE& /*unused*/) noexcept {
    //Nothing to test, generators are of infinite size
}

template<typename LE, typename RE, cpp_enable_if(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE>::value && !all_fast<LE,RE>::value)>
void ensure_same_size(const LE& lhs, const RE& rhs){
    cpp_assert(size(lhs) == size(rhs), "Cannot perform element-wise operations on collections of different size");
    cpp_unused(lhs);
    cpp_unused(rhs);
}

template<typename LE, typename RE, cpp_enable_if(!(etl_traits<LE>::is_generator || etl_traits<RE>::is_generator) && all_etl_expr<LE, RE>::value && all_fast<LE, RE>::value)>
void ensure_same_size(const LE& /*unused*/, const RE& /*unused*/){
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

template<std::size_t C1, std::size_t C2, typename E, cpp_enable_if(etl_traits<E>::dimensions() == 2 && !etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& e){
    cpp_assert(etl::template dim<0>(e) % C1 == 0 && etl::template dim<1>(e) % C2 == 0, "Dimensions not divisible by the pooling ratio");
    cpp_unused(e);
}

template<std::size_t C1, std::size_t C2, typename E, cpp_enable_if(etl_traits<E>::dimensions() == 3 && !etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& e){
    cpp_assert(etl::template dim<1>(e) % C1 == 0 && etl::template dim<2>(e) % C2 == 0, "Dimensions not divisible by the pooling ratio");
    cpp_unused(e);
}

template<std::size_t C1, std::size_t C2, typename E, cpp_enable_if(etl_traits<E>::dimensions() == 2 && etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& /*unused*/){
    static_assert(etl_traits<E>::template dim<0>() % C1 == 0 && etl_traits<E>::template dim<1>() % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

template<std::size_t C1, std::size_t C2, typename E, cpp_enable_if(etl_traits<E>::dimensions() == 3 && etl_traits<E>::is_fast)>
void validate_pmax_pooling_impl(const E& /*unused*/){
    static_assert(etl_traits<E>::template dim<1>() % C1 == 0 && etl_traits<E>::template dim<2>() % C2 == 0, "Dimensions not divisible by the pooling ratio");
}

template<std::size_t C1, std::size_t C2, typename E>
void validate_pmax_pooling(const E& e){
    static_assert(is_etl_expr<E>::value, "Prob. Max Pooling only defined for ETL expressions");
    static_assert(etl_traits<E>::dimensions() == 2 || etl_traits<E>::dimensions() == 3, "Prob. Max Pooling only defined for 2D and 3D");

    validate_pmax_pooling_impl<C1, C2>(e);
}

} //end of namespace etl

#endif
