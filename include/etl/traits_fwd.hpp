//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRAITS_FWD_HPP
#define ETL_TRAITS_FWD_HPP

namespace etl {

template<typename T, typename Enable = void>
struct is_etl_expr;

template<typename E, typename Enable = void>
struct sub_size_compare;

template<typename T, typename Enable = void>
struct etl_traits;

template<typename E, cpp::enable_if_u<cpp::not_u<etl_traits<E>::is_fast>::value> = cpp::detail::dummy>
std::size_t size(const E& v);

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t size(const E&);

template<typename E>
constexpr std::size_t dimensions(const E&);

template<std::size_t D, typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t dim(const E& e);

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t dim(const E& e, std::size_t d);

template<std::size_t D, typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t dim(const E&);

} //end of namespace etl

#endif
