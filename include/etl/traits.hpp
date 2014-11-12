//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRAITS_HPP
#define ETL_TRAITS_HPP

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "traits_fwd.hpp" //To avoid nasty errors

namespace etl {

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
struct is_fast_matrix : std::integral_constant<bool, is_var<etl::fast_matrix, std::decay_t<T>>::value> {};

template<typename T>
struct is_dyn_matrix : std::integral_constant<bool, is_2<etl::dyn_matrix, std::decay_t<T>>::value> {};

template<typename T>
struct is_unary_expr : std::integral_constant<bool, cpp::is_specialization_of<etl::unary_expr, std::decay_t<T>>::value> {};

template<typename T>
struct is_stable_transform_expr : std::integral_constant<bool, cpp::is_specialization_of<etl::stable_transform_expr, std::decay_t<T>>::value> {};

template<typename T>
struct is_binary_expr : std::integral_constant<bool, cpp::is_specialization_of<etl::binary_expr, std::decay_t<T>>::value> {};

template<typename T>
struct is_generator_expr : std::integral_constant<bool, cpp::is_specialization_of<etl::generator_expr, std::decay_t<T>>::value> {};

template<typename T>
struct is_transformer : std::integral_constant<bool, cpp::or_u<
            cpp::or_u<
                cpp::is_specialization_of<etl::hflip_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::vflip_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::fflip_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::transpose_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::sum_r_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::sum_l_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::mean_r_transformer, std::decay_t<T>>::value,
                cpp::is_specialization_of<etl::mean_l_transformer, std::decay_t<T>>::value
            >::value,
            is_var<etl::rep_r_transformer, std::decay_t<T>>::value,
            is_var<etl::rep_l_transformer, std::decay_t<T>>::value,
            is_3<etl::p_max_pool_h_transformer, std::decay_t<T>>::value,
            is_3<etl::p_max_pool_p_transformer, std::decay_t<T>>::value
        >::value> {};

template<typename T>
struct is_view : std::integral_constant<bool, cpp::or_u<
            is_2<etl::dim_view, std::decay_t<T>>::value,
            is_3<etl::fast_matrix_view, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::dyn_matrix_view, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::sub_view, std::decay_t<T>>::value
            >::value> {};

template<typename T, typename Enable>
struct is_etl_expr : std::integral_constant<bool, cpp::or_u<
       is_fast_matrix<T>::value,
       is_dyn_matrix<T>::value,
       is_unary_expr<T>::value, is_binary_expr<T>::value,
       is_stable_transform_expr<T>::value,
       is_generator_expr<T>::value,
       is_transformer<T>::value, is_view<T>::value
    >::value> {};

template<typename T, typename Enable>
struct is_copy_expr : std::integral_constant<bool, cpp::or_u<
       is_fast_matrix<T>::value,
       is_dyn_matrix<T>::value,
       is_unary_expr<T>::value,
       is_binary_expr<T>::value,
       is_stable_transform_expr<T>::value
    >::value> {};

template<typename T, typename Enable = void>
struct is_etl_value :
    std::integral_constant<bool, cpp::or_u<is_fast_matrix<T>::value, is_dyn_matrix<T>::value>::value> {};

template<typename T, typename Enable>
struct etl_traits;

/*!
 * \brief Specialization for value structures (fast_vector, fast_matrix)
 */
template<typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_fast = is_fast_matrix<T>::value;
    static constexpr const bool is_value = true;
    static constexpr const bool is_generator = false;

    static std::size_t size(const T& v){
        return v.size();
    }

    static std::size_t dim(const T& v, std::size_t d){
        return v.dim(d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
    using sub_expr_t = std::decay_t<Expr>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = etl_traits<sub_expr_t>::is_generator;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.value());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
 * \brief Specialization (un)stable_transform_expr
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_stable_transform_expr<T>::value>> {
    using expr_t = std::decay_t<T>;
    using sub_expr_t = std::decay_t<typename expr_t::expr_type>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = etl_traits<sub_expr_t>::is_generator;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.value());
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
 * \brief Specialization generator_expr
 */
template <typename Generator>
struct etl_traits<etl::generator_expr<Generator>> {
    static constexpr const bool is_fast = true;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = true;
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>> {
    static constexpr const bool is_fast = true;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = true;
};

/*!
 * \brief Specialization for binary_expr.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    using expr_t = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>;
    using left_expr_t = std::decay_t<LeftExpr>;
    using right_expr_t = std::decay_t<RightExpr>;

    static constexpr const bool left_directed = cpp::not_u<etl_traits<left_expr_t>::is_generator>::value;

    using sub_expr_t = std::conditional_t<left_directed, left_expr_t, right_expr_t>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = cpp::and_u<
            etl_traits<left_expr_t>::is_generator,
            etl_traits<right_expr_t>::is_generator>::value;

    template<bool B = left_directed, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr auto& get(const expr_t& v){
        return v.lhs();
    }

    template<bool B = left_directed, cpp::disable_if_u<B> = cpp::detail::dummy>
    static constexpr auto& get(const expr_t& v){
        return v.rhs();
    }

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(get(v));
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(get(v), d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.sub, 1-d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
 * \brief Specialization for rep_r_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_r_transformer<T, D...>> {
    using expr_t = etl::rep_r_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == 0 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0,0,D...>::value;
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr std::size_t size(){
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D2>
    static constexpr std::size_t dim(){
        return D2 == 0 ? etl_traits<sub_expr_t>::template dim<0>() : nth_size<D2-1,0,D...>::value;
    }

    static constexpr std::size_t dimensions(){
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for rep_l_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_l_transformer<T, D...>> {
    using expr_t = etl::rep_l_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == dimensions() - 1 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0,0,D...>::value;
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr std::size_t size(){
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template<std::size_t D2>
    static constexpr std::size_t dim(){
        return D2 == dimensions() - 1 ? etl_traits<sub_expr_t>::template dim<0>() : nth_size<D2,0,D...>::value;
    }

    static constexpr std::size_t dimensions(){
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_u<
            cpp::is_specialization_of<etl::sum_r_transformer, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::mean_r_transformer, std::decay_t<T>>::value
        >::value>> {
    using expr_t = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t){
        return etl::dim<0>(v.sub);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    static constexpr std::size_t dimensions(){
        return 1;
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_u<
            cpp::is_specialization_of<etl::sum_l_transformer, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::mean_l_transformer, std::decay_t<T>>::value
        >::value>> {
    using expr_t = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl::size(v.sub) / etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl::dim(v.sub, d + 1);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

template<typename T, std::size_t C1, std::size_t C2>
struct etl_traits<p_max_pool_p_transformer<T, C1, C2>> {
    using expr_t = p_max_pool_p_transformer<T, C1, C2>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.sub) / (C1 * C2);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        if(d == dimensions() - 1){
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C2;
        } else if(d == dimensions() - 2){
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C1;
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, d);
        }
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
    static constexpr std::size_t size(){
        return etl_traits<sub_expr_t>::size() / (C1 * C2);
    }

    template<std::size_t D>
    static constexpr std::size_t dim(){
        return
                D == dimensions() - 1 ? etl_traits<sub_expr_t>::template dim<D>() / C2
            :   D == dimensions() - 2 ? etl_traits<sub_expr_t>::template dim<D>() / C1
            :                           etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions(){
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for flipping transformers
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_u<
            cpp::is_specialization_of<etl::hflip_transformer, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::vflip_transformer, std::decay_t<T>>::value,
            cpp::is_specialization_of<etl::fflip_transformer, std::decay_t<T>>::value,
            is_3<etl::p_max_pool_h_transformer, std::decay_t<T>>::value
        >::value>> {
    using expr_t = T;
    using sub_expr_t = std::decay_t<typename T::sub_type>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        if(D == 1){
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else if (D == 2){
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        cpp_assert(d == 0, "Invalid dimension");
        cpp_unused(d);

        return size(v);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

    static std::size_t size(const expr_t& v){
        return etl_traits<sub_expr_t>::size(v.parent) / etl_traits<sub_expr_t>::dim(v.parent, 0);
    }

    static std::size_t dim(const expr_t& v, std::size_t d){
        return etl_traits<sub_expr_t>::dim(v.parent, d + 1);
    }

    template<bool B = is_fast, cpp::enable_if_u<B> = cpp::detail::dummy>
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
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = true;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

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
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast = false;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator = false;

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

template<typename E>
using decay_traits = etl_traits<std::decay_t<E>>;

template<typename E, cpp::enable_if_u<cpp::not_u<etl_traits<E>::is_fast>::value>>
std::size_t size(const E& v){
    return etl_traits<E>::size(v);
}

template<typename E, cpp::enable_if_u<cpp::not_u<etl_traits<E>::is_fast>::value> = cpp::detail::dummy>
std::size_t rows(const E& v){
    return etl_traits<E>::dim(v, 0);
}

template<typename E, cpp::enable_if_u<cpp::not_u<etl_traits<E>::is_fast>::value> = cpp::detail::dummy>
std::size_t columns(const E& v){
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(v, 1);
}

template<typename E, cpp::enable_if_u<cpp::not_u<etl_traits<E>::is_fast>::value> = cpp::detail::dummy>
std::size_t subsize(const E& v){
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size(v) / etl_traits<E>::dim(v, 0);
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast>>
constexpr std::size_t size(const E&){
    return etl_traits<E>::size();
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t rows(const E&){
    return etl_traits<E>::template dim<0>();
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t columns(const E&){
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");

    return etl_traits<E>::template dim<1>();
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t subsize(const E&){
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

template<typename E>
constexpr std::size_t dimensions(const E&){
    return etl_traits<E>::dimensions();
}

template<typename E>
constexpr std::size_t dimensions(){
    return decay_traits<E>::dimensions();
}

template<std::size_t D, typename E, cpp::disable_if_u<etl_traits<E>::is_fast>>
constexpr std::size_t dim(const E& e){
    return etl_traits<E>::dim(e, D);
}

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast>>
constexpr std::size_t dim(const E& e, std::size_t d){
    return etl_traits<E>::dim(e, d);
}

template<std::size_t D, typename E, cpp::enable_if_u<etl_traits<E>::is_fast>>
constexpr std::size_t dim(const E&){
    return etl_traits<E>::template dim<D>();
}

template<std::size_t D, typename E>
constexpr std::size_t dim(){
    return decay_traits<E>::template dim<D>();
}

template<typename LE, typename RE, cpp::enable_if_one_u<etl_traits<LE>::is_generator, etl_traits<RE>::is_generator> = cpp::detail::dummy>
void ensure_same_size(const LE&, const RE&){
    //Nothing to test, generators are of infinite size
}

template<typename LE, typename RE, cpp::disable_if_one_u<
        cpp::and_u<
            cpp::not_u<cpp::or_u<etl_traits<LE>::is_generator, etl_traits<RE>::is_generator>::value>::value,
            cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value, etl_traits<LE>::is_fast, etl_traits<RE>::is_fast>::value
        >::value,
        cpp::or_u<etl_traits<LE>::is_generator, etl_traits<RE>::is_generator>::value
    > = cpp::detail::dummy>
void ensure_same_size(const LE& lhs, const RE& rhs){
    cpp_assert(size(lhs) == size(rhs), "Cannot perform element-wise operations on collections of different size");
    cpp_unused(lhs);
    cpp_unused(rhs);
}

template<typename LE, typename RE, cpp::enable_if_all_u<cpp::not_u<cpp::or_u<etl_traits<LE>::is_generator, etl_traits<RE>::is_generator>::value>::value, is_etl_expr<LE>::value, is_etl_expr<RE>::value, etl_traits<LE>::is_fast, etl_traits<RE>::is_fast> = cpp::detail::dummy>
void ensure_same_size(const LE&, const RE&){
    static_assert(etl_traits<LE>::size() == etl_traits<RE>::size(), "Cannot perform element-wise operations on collections of different size");
}

template<typename E, typename Enable>
struct sub_size_compare;

template<typename E>
struct sub_size_compare<E, std::enable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, std::numeric_limits<std::size_t>::max()> {};

template<typename E>
struct sub_size_compare<E, cpp::disable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, etl_traits<E>::dimensions()> {};

} //end of namespace etl

#endif
