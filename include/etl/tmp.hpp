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

template<bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false>
struct or_u : std::true_type {};

template<>
struct or_u<false, false, false, false, false, false> : std::false_type {};

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

template<typename T>
struct dyn_vector;

template<typename T>
struct dyn_matrix;

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
struct is_dyn_vector : std::integral_constant<bool, is_specialization_of<etl::dyn_vector, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_dyn_matrix : std::integral_constant<bool, is_specialization_of<etl::dyn_matrix, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_unary_expr : std::integral_constant<bool, is_specialization_of<etl::unary_expr, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T>
struct is_binary_expr : std::integral_constant<bool, is_specialization_of<etl::binary_expr, remove_cv_t<remove_reference_t<T>>>::value> {};

template<typename T, typename Enable = void> 
struct is_etl_expr : std::integral_constant<bool, or_u<
       is_fast_vector<T>::value, is_fast_matrix<T>::value,
       is_dyn_vector<T>::value, is_dyn_matrix<T>::value,
       is_unary_expr<T>::value, is_binary_expr<T>::value
    >::value> {};

template<typename T, typename Enable = void>
struct is_etl_value :
    std::integral_constant<bool, or_u<is_fast_vector<T>::value, is_fast_matrix<T>::value, is_dyn_vector<T>::value, is_dyn_matrix<T>::value>::value> {};

template<typename T, typename Enable = void>
struct etl_traits;

template<typename T>
struct etl_traits<T, enable_if_t<or_u<is_fast_vector<T>::value, is_dyn_vector<T>::value>::value>> {
    static constexpr const bool is_vector = true;
    static constexpr const bool is_matrix = false;
    static constexpr const bool is_fast = is_fast_vector<T>::value;
    static constexpr const bool is_value = true;

    static std::size_t size(const T& v){
        return v.size(); 
    }
};

template<typename T>
struct etl_traits<T, enable_if_t<or_u<is_fast_matrix<T>::value, is_dyn_matrix<T>::value>::value>> {
    static constexpr const bool is_vector = false;
    static constexpr const bool is_matrix = true;
    static constexpr const bool is_fast = is_fast_matrix<T>::value;
    static constexpr const bool is_value = true;

    static std::size_t size(const T& v){
        return v.size(); 
    }

    static std::size_t rows(const T& v){
        return v.rows(); 
    }

    static std::size_t columns(const T& v){
        return v.columns(); 
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_vector>> {
    static constexpr const bool is_vector = true;
    static constexpr const bool is_matrix = false;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::size(v.value()); 
    }
};

template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>, enable_if_t<etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_matrix>> {
    static constexpr const bool is_vector = false;
    static constexpr const bool is_matrix = true;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::size(v.value()); 
    }

    static std::size_t rows(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::rows(v.value()); 
    }

    static std::size_t columns(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::columns(v.value()); 
    }
};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<and_u<is_etl_expr<LeftExpr>::value, etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_vector>::value>> {
    static constexpr const bool is_vector = true;
    static constexpr const bool is_matrix = false;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::size(v.lhs()); 
    }
};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<and_u<not_u<is_etl_expr<LeftExpr>::value>::value, is_etl_expr<RightExpr>::value, etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_vector>::value>> {
    static constexpr const bool is_vector = true;
    static constexpr const bool is_matrix = false;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::size(v.rhs()); 
    }
};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<and_u<is_etl_expr<LeftExpr>::value, etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_matrix>::value>> {
    static constexpr const bool is_vector = false;
    static constexpr const bool is_matrix = true;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::size(v.lhs()); 
    }

    static std::size_t rows(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::rows(v.lhs()); 
    }

    static std::size_t columns(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::columns(v.lhs()); 
    }
};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<and_u<not_u<is_etl_expr<LeftExpr>::value>::value, is_etl_expr<RightExpr>::value, etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_matrix>::value>> {
    static constexpr const bool is_vector = false;
    static constexpr const bool is_matrix = true;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::size(v.rhs()); 
    }

    static std::size_t rows(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::rows(v.rhs()); 
    }

    static std::size_t columns(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::columns(v.rhs()); 
    }
};

template<typename E>
std::size_t size(const E& v){
    return etl_traits<E>::size(v);
}

template<typename E>
std::size_t columns(const E& v){
    return etl_traits<E>::columns(v);
}

template<typename E>
std::size_t rows(const E& v){
    return etl_traits<E>::rows(v);
}

#endif