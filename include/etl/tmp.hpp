//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TMP_HPP
#define ETL_TMP_HPP

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

template< bool B, class T = void >
using disable_if_t = typename std::enable_if<!B, T>::type;

namespace detail {

//Note: Unfortunately, CLang is bugged (Bug 11723), therefore, it is not
//possible to use universal enable_if/disable_if directly, it is necessary to
//use the dummy :( FU Clang!

enum class enabler_t { DUMMY };
constexpr const enabler_t dummy = enabler_t::DUMMY;

} //end of detail

template<bool B>
using enable_if_u = typename std::enable_if<B, detail::enabler_t>::type;

template<bool B>
using disable_if_u = typename std::enable_if<!B, detail::enabler_t>::type;

template <bool b1, bool b2>
struct and_u : std::false_type {};

template <>
struct and_u<true, true> : std::true_type {};

template <bool b1>
struct not_u : std::true_type {};

template <>
struct not_u<true> : std::false_type {};

template <bool b1, bool b2>
struct or_u : std::true_type {};

template <>
struct or_u<false, false> : std::false_type {};

template<typename T, typename Enable = void> 
struct is_etl_expr : std::false_type {};

template<typename T> 
struct is_etl_expr<T, enable_if_t<std::remove_reference<T>::type::etl_marker>> : std::true_type {};

#endif