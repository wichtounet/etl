//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRAITS_HPP
#define ETL_TRAITS_HPP

#include "tmp.hpp"

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

template<typename T>
struct hflip_transformer;

template<typename T>
struct vflip_transformer;

template<typename T>
struct fflip_transformer;

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

template<typename T>
struct is_transformer_expr : std::integral_constant<bool, or_u<
            is_specialization_of<etl::hflip_transformer, remove_cv_t<remove_reference_t<T>>>::value,
            is_specialization_of<etl::vflip_transformer, remove_cv_t<remove_reference_t<T>>>::value,
            is_specialization_of<etl::fflip_transformer, remove_cv_t<remove_reference_t<T>>>::value>::value> {};

template<typename T, typename Enable = void>
struct is_etl_expr : std::integral_constant<bool, or_u<
       is_fast_vector<T>::value, is_fast_matrix<T>::value,
       is_dyn_vector<T>::value, is_dyn_matrix<T>::value,
       is_unary_expr<T>::value, is_binary_expr<T>::value,
       is_transformer_expr<T>::value
    >::value> {};

template<typename T, typename Enable = void>
struct is_etl_value :
    std::integral_constant<bool, or_u<is_fast_vector<T>::value, is_fast_matrix<T>::value, is_dyn_vector<T>::value, is_dyn_matrix<T>::value>::value> {};

template<typename T, typename Enable = void>
struct etl_traits;

/*!
 * \brief Specialization for value structures (fast_vector, dyn_vector,
 * fast_matrix, dyn_matrix).
 */
template<typename T>
struct etl_traits<T, enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_vector = or_u<is_dyn_vector<T>::value, is_fast_vector<T>::value>::value;
    static constexpr const bool is_matrix = or_u<is_dyn_matrix<T>::value, is_fast_matrix<T>::value>::value;
    static constexpr const bool is_fast = or_u<is_fast_vector<T>::value, is_fast_matrix<T>::value>::value;
    static constexpr const bool is_value = true;

    static std::size_t size(const T& v){
        return v.size();
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t rows(const T& v){
        return v.rows();
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t columns(const T& v){
        return v.columns();
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return T::size();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t rows(){
        return T::rows();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t columns(){
        return T::columns();
    }
};

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    static constexpr const bool is_vector = etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_vector;
    static constexpr const bool is_matrix = etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_matrix;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<Expr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::size(v.value());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t rows(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::rows(v.value());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t columns(const etl::unary_expr<T, Expr, UnaryOp>& v){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::columns(v.value());
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::size();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t rows(){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::rows();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t columns(){
        return etl_traits<remove_cv_t<remove_reference_t<Expr>>>::columns();
    }
};

/*!
 * \brief Specialization for binary_expr when the type is decided by the left
 * expression.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<is_etl_expr<LeftExpr>::value>> {
    static constexpr const bool is_vector = etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_vector;
    static constexpr const bool is_matrix = etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_matrix;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::size(v.lhs());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t rows(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::rows(v.lhs());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t columns(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::columns(v.lhs());
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::size();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t rows(){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::rows();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t columns(){
        return etl_traits<remove_cv_t<remove_reference_t<LeftExpr>>>::columns();
    }
};

/*!
 * \brief Specialization for binary_expr when the type is decided by the right
 * expression.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, enable_if_t<and_u<not_u<is_etl_expr<LeftExpr>::value>::value, is_etl_expr<RightExpr>::value>::value>> {
    static constexpr const bool is_vector = etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_vector;
    static constexpr const bool is_matrix = etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_matrix;
    static constexpr const bool is_fast = etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::size(v.rhs());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t rows(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::rows(v.rhs());
    }

    template<bool B = is_matrix, enable_if_u<B> = detail::dummy>
    static std::size_t columns(const etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>& v){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::columns(v.rhs());
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::size();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t rows(){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::rows();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t columns(){
        return etl_traits<remove_cv_t<remove_reference_t<RightExpr>>>::columns();
    }
};

/*!
 * \brief Specialization for transformers
 */
template <typename T>
struct etl_traits<T, enable_if_t<is_transformer_expr<T>::value>> {
    static constexpr const bool is_vector = etl_traits<typename T::sub_type>::is_vector;
    static constexpr const bool is_matrix = etl_traits<typename T::sub_type>::is_matrix;
    static constexpr const bool is_fast = etl_traits<typename T::sub_type>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const T& v){
        return etl_traits<typename T::sub_type>::size(v.sub);
    }

    template<typename TT = typename T::sub_type, enable_if_u<etl_traits<TT>::is_matrix> = detail::dummy>
    static std::size_t rows(const T& v){
        return etl_traits<typename T::sub_type>::rows(v.sub);
    }

    template<typename TT = typename T::sub_type, enable_if_u<etl_traits<TT>::is_matrix> = detail::dummy>
    static std::size_t columns(const T& v){
        return etl_traits<typename T::sub_type>::columns(v.sub);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<remove_cv_t<remove_reference_t<typename T::sub_type>>>::size();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t rows(){
        return etl_traits<remove_cv_t<remove_reference_t<typename T::sub_type>>>::rows();
    }

    template<bool B = is_matrix, enable_if_u<and_u<B, is_fast>::value> = detail::dummy>
    static constexpr std::size_t columns(){
        return etl_traits<remove_cv_t<remove_reference_t<typename T::sub_type>>>::columns();
    }
};

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t size(const E& v){
    return etl_traits<E>::size(v);
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t size(const E&){
    return etl_traits<E>::size();
}

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t columns(const E& v){
    return etl_traits<E>::columns(v);
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t columns(const E&){
    return etl_traits<E>::columns();
}

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t rows(const E& v){
    return etl_traits<E>::rows(v);
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t rows(const E&){
    return etl_traits<E>::rows();
}

template<typename LE, typename RE, disable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value, etl_traits<LE>::is_fast, etl_traits<RE>::is_fast>::value> = detail::dummy>
void ensure_same_size(const LE& lhs, const RE& rhs){
    etl_assert(size(lhs) == size(rhs), "Cannot perform element-wise operations on collections of different size");
    etl_unused(lhs);
    etl_unused(rhs);
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value, etl_traits<LE>::is_fast, etl_traits<RE>::is_fast>::value> = detail::dummy>
void ensure_same_size(const LE&, const RE&){
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

#endif