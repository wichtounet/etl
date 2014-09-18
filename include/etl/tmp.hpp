//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TMP_HPP
#define ETL_TMP_HPP

template<bool B, class T = void>
using enable_if_t = typename std::enable_if<B,T>::type;

template<bool B, class T = void>
using disable_if_t = typename std::enable_if<!B, T>::type;

template<typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template<typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

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

template<bool b1>
struct not_u : std::true_type {};

template<>
struct not_u<true> : std::false_type {};

template<bool b1, bool b2, bool b3 = true, bool b4 = true>
struct and_u : std::false_type {};

template<>
struct and_u<true, true, true, true> : std::true_type {};

template<bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false, bool b7 = false, bool b8 = false, bool b9 = false>
struct or_u : std::true_type {};

template<>
struct or_u<false, false, false, false, false, false, false, false, false> : std::false_type {};

template<template<typename...> class TT, typename T>
struct is_specialization_of : std::false_type {};

template<template<typename...> class TT, typename... Args>
struct is_specialization_of<TT, TT<Args...>> : std::true_type {};

template<typename V, typename F, typename... S>
struct all_convertible_to  : std::integral_constant<bool, and_u<all_convertible_to<V, F>::value, all_convertible_to<V, S...>::value>::value> {};

template<typename V, typename F>
struct all_convertible_to<V, F> : std::integral_constant<bool, std::is_convertible<F, V>::value> {};

#endif