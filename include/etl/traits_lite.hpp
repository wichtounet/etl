//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TRAITS_LITE_HPP
#define ETL_TRAITS_LITE_HPP

namespace etl {

template<typename T, typename DT = std::decay_t<T>>
struct is_dyn_matrix;

template<typename T>
struct is_etl_expr;

template<typename T>
struct is_copy_expr;

template<typename T, typename DT = std::decay_t<T>>
struct is_temporary_unary_expr;

template<typename T, typename DT = std::decay_t<T>>
struct is_temporary_binary_expr;

template<typename T>
struct is_temporary_expr;

template<typename T, typename DT = std::decay_t<T>>
struct is_view;

template<typename T, typename DT = std::decay_t<T>>
struct is_magic_view;

template<typename T, typename DT = std::decay_t<T>>
struct is_transformer;

template<typename T>
struct is_etl_value;

template<typename E, typename Enable = void>
struct sub_size_compare;

template<typename T, typename Enable = void>
struct etl_traits;

template<typename T, typename DT = std::decay_t<T>>
struct has_direct_access;

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
std::size_t size(const E& v);

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t size(const E& /*unused*/) noexcept;

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
std::size_t dim(const E& e, std::size_t d);

template<std::size_t D, typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
std::size_t dim(const E& e);

template<std::size_t D, typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t dim(const E& /*unused*/) noexcept;

template<std::size_t D, typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t dim() noexcept;

template<typename T>
struct is_single_precision : std::is_same<typename std::decay_t<T>::value_type, float> {};

template<typename A, typename B>
struct is_single_precision_2 : cpp::and_c<is_single_precision<A>, is_single_precision<B>> {};

template<typename A, typename B, typename C>
struct is_single_precision_3 : cpp::and_c<is_single_precision<A>, is_single_precision<B>, is_single_precision<C>> {};

template<typename T>
struct is_double_precision : std::is_same<typename std::decay_t<T>::value_type, double> {};

template<typename A, typename B>
struct is_double_precision_2 : cpp::and_c<is_double_precision<A>, is_double_precision<B>> {};

template<typename A, typename B, typename C>
struct is_double_precision_3 : cpp::and_c<is_double_precision<A>, is_double_precision<B>, is_double_precision<C>> {};

template<typename T>
struct is_complex_single_precision : std::is_same<typename std::decay_t<T>::value_type, std::complex<float>> {};

template<typename T>
struct is_complex_double_precision : std::is_same<typename std::decay_t<T>::value_type, std::complex<double>> {};

template<typename T>
struct is_complex : cpp::or_c<is_complex_single_precision<T>, is_complex_double_precision<T>> {};

template<typename A, typename B>
struct is_dma_2 : cpp::and_c<has_direct_access<A>, has_direct_access<B>> {};

template<typename A, typename B, typename C>
struct is_dma_3 : cpp::and_c<has_direct_access<A>, has_direct_access<B>, has_direct_access<C>> {};

template<typename E>
using decay_traits = etl_traits<std::decay_t<E>>;

template<typename E>
constexpr std::size_t dimensions(const E& /*unused*/) noexcept {
    return etl_traits<E>::dimensions();
}

template<typename E>
constexpr std::size_t dimensions() noexcept {
    return decay_traits<E>::dimensions();
}

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
std::size_t rows(const E& v){
    return etl_traits<E>::dim(v, 0);
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t rows(const E& /*unused*/) noexcept {
    return etl_traits<E>::template dim<0>();
}

template<typename E, cpp::disable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
std::size_t columns(const E& v){
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(v, 1);
}

template<typename E, cpp::enable_if_u<etl_traits<E>::is_fast> = cpp::detail::dummy>
constexpr std::size_t columns(const E& /*unused*/) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::template dim<1>();
}

} //end of namespace etl

#endif
