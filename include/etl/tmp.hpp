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

template<bool b1, bool b2, bool b3 = false, bool b4 = false>
struct or_u : std::true_type {};

template<>
struct or_u<false, false, false, false> : std::false_type {};

template<template<typename...> class TT, typename T>
struct is_specialization_of : std::false_type {};

template<template<typename...> class TT, typename... Args>
struct is_specialization_of<TT, TT<Args...>> : std::true_type {};

template<template<typename, std::size_t> class TT, typename T>
struct is_2 : std::false_type { };

template<template<typename, std::size_t> class TT, typename V1, std::size_t R>
struct is_2<TT, TT<V1, R>> : std::true_type { };

template<template<typename, std::size_t, std::size_t> class TT, typename T>
struct is_3 : std::false_type { };

template<template<typename, std::size_t, std::size_t> class TT, typename V1, std::size_t R1, std::size_t R2>
struct is_3<TT, TT<V1, R1, R2>> : std::true_type { };

namespace etl {

template<typename T, std::size_t Rows>
struct fast_vector;

template<typename T, size_t Rows, size_t Columns>
struct fast_matrix;

template <typename T, typename Expr, typename UnaryOp>
class unary_expr;

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr;

};

template<typename T>
struct is_fast_vector : std::integral_constant<bool, is_2<etl::fast_vector, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_fast_matrix : std::integral_constant<bool, is_3<etl::fast_matrix, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_unary_expr : std::integral_constant<bool, is_specialization_of<etl::unary_expr, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_binary_expr : std::integral_constant<bool, is_specialization_of<etl::binary_expr, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T, typename Enable = void> 
struct is_etl_expr : std::integral_constant<bool, or_u<
       is_fast_vector<T>::value, is_fast_matrix<T>::value,
       is_unary_expr<T>::value, is_binary_expr<T>::value
    >::value> {};

template<typename T, typename Enable = void> 
struct is_etl_fast : 
    std::integral_constant<bool, or_u<is_fast_vector<T>::value, is_fast_matrix<T>::value>::value> {};

template<typename LE, typename RE, typename Enable = void>
struct get_etl_size ;

template<typename LE, typename RE>
struct get_etl_size<LE, RE, enable_if_t<is_etl_expr<LE>::value>> 
    : std::integral_constant<std::size_t, std::remove_reference<LE>::type::etl_size> {} ;

template<typename LE, typename RE>
struct get_etl_size<LE, RE, enable_if_t<and_u<is_etl_expr<RE>::value, not_u<is_etl_expr<LE>::value>::value>::value>> 
    : std::integral_constant<std::size_t, std::remove_reference<RE>::type::etl_size> {};

#endif