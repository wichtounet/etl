//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRAITS_HPP
#define ETL_TRAITS_HPP

#include "tmp.hpp"
#include "assert.hpp"

namespace etl {

template<typename T, std::size_t D>
struct dyn_matrix;

template<typename T, size_t... Dims>
struct fast_matrix;

template<typename T, size_t Rows, size_t Columns>
struct fast_matrix_view;

template<typename T>
struct dyn_matrix_view;

template <typename T, typename Expr, typename UnaryOp>
class unary_expr;

template <typename T, typename Expr>
class transform_expr;

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr;

template<typename T>
struct hflip_transformer;

template<typename T>
struct vflip_transformer;

template<typename T>
struct fflip_transformer;

template<typename T>
struct transpose_transformer;

template<typename T, std::size_t D>
struct dim_view;

template<typename T>
struct sub_view;

template<template<typename, std::size_t> class TT, typename T>
struct is_2 : std::false_type { };

template<template<typename, std::size_t> class TT, typename V1, std::size_t R>
struct is_2<TT, TT<V1, R>> : std::true_type { };

template<template<typename, std::size_t, std::size_t> class TT, typename T>
struct is_3 : std::false_type { };

template<template<typename, std::size_t, std::size_t> class TT, typename V1, std::size_t R1, std::size_t R2>
struct is_3<TT, TT<V1, R1, R2>> : std::true_type { };

template<template<typename, std::size_t...> class TT, typename T>
struct is_var : std::false_type { };

template<template<typename, std::size_t...> class TT, typename V1, std::size_t... R>
struct is_var<TT, TT<V1, R...>> : std::true_type { };

template<typename T>
struct is_fast_matrix : std::integral_constant<bool, is_var<etl::fast_matrix, std::remove_cv_t<std::remove_reference_t<T>>>::value> {};

template<typename T>
struct is_dyn_matrix : std::integral_constant<bool, is_2<etl::dyn_matrix, std::remove_cv_t<std::remove_reference_t<T>>>::value> {};

template<typename T>
struct is_unary_expr : std::integral_constant<bool, is_specialization_of<etl::unary_expr, std::remove_cv_t<std::remove_reference_t<T>>>::value> {};

template<typename T>
struct is_transform_expr : std::integral_constant<bool, is_specialization_of<etl::transform_expr, std::remove_cv_t<std::remove_reference_t<T>>>::value> {};

template<typename T>
struct is_binary_expr : std::integral_constant<bool, is_specialization_of<etl::binary_expr, std::remove_cv_t<std::remove_reference_t<T>>>::value> {};

template<typename T>
struct is_transformer_expr : std::integral_constant<bool, or_u<
            is_specialization_of<etl::transpose_transformer, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_specialization_of<etl::hflip_transformer, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_specialization_of<etl::vflip_transformer, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_specialization_of<etl::fflip_transformer, std::remove_cv_t<std::remove_reference_t<T>>>::value>::value> {};

template<typename T>
struct is_view : std::integral_constant<bool, or_u<
            is_2<etl::dim_view, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_3<etl::fast_matrix_view, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_specialization_of<etl::dyn_matrix_view, std::remove_cv_t<std::remove_reference_t<T>>>::value,
            is_specialization_of<etl::sub_view, std::remove_cv_t<std::remove_reference_t<T>>>::value
            >::value> {};

template<typename T, typename Enable = void>
struct is_etl_expr : std::integral_constant<bool, or_u<
       is_fast_matrix<T>::value,
       is_dyn_matrix<T>::value,
       is_unary_expr<T>::value, is_binary_expr<T>::value,
       is_transform_expr<T>::value,
       is_transformer_expr<T>::value, is_view<T>::value
    >::value> {};

template<typename T, typename Enable = void>
struct is_etl_value :
    std::integral_constant<bool, or_u<is_fast_matrix<T>::value, is_dyn_matrix<T>::value>::value> {};

template<typename T, typename Enable = void>
struct etl_traits;

/*!
 * \brief Specialization for value structures (fast_vector, fast_matrix)
 */
template<typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_fast = is_fast_matrix<T>::value;
    static constexpr const bool is_value = true;

    static std::size_t size(const T& v){
        return v.size();
    }

    static std::size_t dim(const T& v, std::size_t d){
        return v.dim(d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return T::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return T::n_dimensions;
    }
};

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    using expr_t = etl::unary_expr<T, Expr, UnaryOp>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<Expr>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.value());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization transform_expr
 */
template <typename T, typename Expr>
struct etl_traits<etl::transform_expr<T, Expr>> {
    using expr_t = etl::transform_expr<T, Expr>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<Expr>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.value());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for binary_expr when the type is decided by the left
 * expression.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, std::enable_if_t<is_etl_expr<LeftExpr>::value>> {
    using expr_t = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<LeftExpr>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.lhs());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.lhs(), d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for binary_expr when the type is decided by the right
 * expression.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>, std::enable_if_t<and_u<not_u<is_etl_expr<LeftExpr>::value>::value, is_etl_expr<RightExpr>::value>::value>> {
    using expr_t = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<RightExpr>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.rhs());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.rhs(), d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for tranpose_transformer
 */
template <typename T>
struct etl_traits<transpose_transformer<T>> {
    using expr_t = etl::transpose_transformer<T>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.sub, 1-d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<1-D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for transformers
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<and_u<is_transformer_expr<T>::value, not_u<is_specialization_of<etl::transpose_transformer, T>::value>::value>::value>> {
    using expr_t = T;
    using sub_expr_t = typename T::sub_type;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t = etl::dim_view<T, D>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        if(D == 1){
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else if (D == 2){
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        etl_assert(d == 0, "Invalid dimension");
        etl_unused(d);

        return size(v);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return D == 1 ? etl_traits<sub_expr_t>::template dim<1>() : etl_traits<sub_expr_t>::template dim<0>();
    }

    template<std::size_t D2>
    static constexpr std::size_t dim(){
        static_assert(D2 == 0, "Invalid dimension");

        return size();
    }

    static constexpr std::size_t dimensions(){
        return 1;
    }
};

/*!
 * \brief Specialization for sub_view
 */
template <typename T>
struct etl_traits<etl::sub_view<T>> {
    using expr_t = etl::sub_view<T>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.parent) / etl_traits<sub_expr_t>::dim(v.parent, 0);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.parent, d + 1);
    }

    template<bool B = is_fast, enable_if_u<B> = detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D+1>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

/*!
 * \brief Specialization for fast_matrix_view.
 */
template<typename T, std::size_t Rows, std::size_t Columns>
struct etl_traits<etl::fast_matrix_view<T, Rows, Columns>> {
    using expr_t = etl::fast_matrix_view<T, Rows, Columns>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    static constexpr const bool is_fast = true;
    static constexpr const bool is_value = false;

    static constexpr std::size_t size(const expr_t&){
        return Rows * Columns;
    }

    static std::size_t dim(const expr_t&, std::size_t d){
        return d == 1 ? Rows : Columns;
    }

    static constexpr std::size_t size(){
        return Rows * Columns;
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return D == 0 ? Rows : Columns;
    }

    static constexpr std::size_t dimensions(){
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_matrix_view.
 */
template<typename T>
struct etl_traits<etl::dyn_matrix_view<T>> {
    using expr_t = etl::dyn_matrix_view<T>;
    using sub_expr_t = std::remove_cv_t<std::remove_reference_t<T>>;

    static constexpr const bool is_fast = false;
    static constexpr const bool is_value = false;

    static std::size_t size(const expr_t& v){
        return v.rows * v.columns;
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return d == 0 ? v.rows : v.columns;
    }

    static constexpr std::size_t dimensions(){
        return 2;
    }
};

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t size(const E& v){
    return etl_traits<E>::size(v);
}

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t rows(const E& v){
    return etl_traits<E>::dim(v, 0);
}

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t columns(const E& v){
    etl_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(v, 1);
}

template<typename E, enable_if_u<not_u<etl_traits<E>::is_fast>::value> = detail::dummy>
std::size_t subsize(const E& v){
    //TODO Assert
    return etl_traits<E>::size(v) / etl_traits<E>::dim(v, 0);
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t size(const E&){
    return etl_traits<E>::size();
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t rows(const E&){
    return etl_traits<E>::template dim<0>();
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t columns(const E&){
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");

    return etl_traits<E>::template dim<1>();
}

template<typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t subsize(const E&){
    //TODO Assert
    return etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

template<typename E>
constexpr std::size_t dimensions(const E&){
    return etl_traits<E>::dimensions();
}

template<std::size_t D, typename E, disable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t dim(const E& e){
    return etl_traits<E>::dim(e, D);
}

template<typename E, disable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t dim(const E& e, std::size_t d){
    return etl_traits<E>::dim(e, d);
}

template<std::size_t D, typename E, enable_if_u<etl_traits<E>::is_fast> = detail::dummy>
constexpr std::size_t dim(const E&){
    return etl_traits<E>::template dim<D>();
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

} //end of namespace etl

#endif