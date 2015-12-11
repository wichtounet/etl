//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/traits_lite.hpp" //To avoid nasty errors

namespace etl {

namespace traits_detail {

template <typename T>
struct is_fast_matrix_impl : std::false_type {};

template <typename V1, typename V2, order V3, std::size_t... R>
struct is_fast_matrix_impl<fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

template <typename T>
struct is_dyn_matrix_impl : std::false_type {};

template <typename V1, order V2, std::size_t V3>
struct is_dyn_matrix_impl<dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T>
struct is_sparse_matrix_impl : std::false_type {};

template <typename V1, sparse_storage V2, std::size_t V3>
struct is_sparse_matrix_impl<sparse_matrix_impl<V1, V2, V3>> : std::true_type {};

} //end of namespace traits_detail

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_fast_matrix : traits_detail::is_fast_matrix_impl<std::decay_t<T>> {};

/*!
 * \brief Traits indicating if the given ETL type is a dyn matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_dyn_matrix : traits_detail::is_dyn_matrix_impl<std::decay_t<T>> {};

/*!
 * \brief Traits indicating if the given ETL type is a sparse matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_sparse_matrix : traits_detail::is_sparse_matrix_impl<std::decay_t<T>> {};

template <typename T, typename DT = std::decay_t<T>>
using is_unary_expr               = cpp::is_specialization_of<etl::unary_expr, DT>;

template <typename T, typename DT = std::decay_t<T>>
using is_binary_expr              = cpp::is_specialization_of<etl::binary_expr, DT>;

template <typename T, typename DT = std::decay_t<T>>
using is_generator_expr           = cpp::is_specialization_of<etl::generator_expr, DT>;

template <typename T, typename DT>
struct is_optimized_expr : cpp::is_specialization_of<etl::optimized_expr, DT> {};

template <typename T, typename DT>
struct is_temporary_unary_expr : cpp::is_specialization_of<etl::temporary_unary_expr, DT> {};

template <typename T, typename DT>
struct is_temporary_binary_expr : cpp::is_specialization_of<etl::temporary_binary_expr, DT> {};

template <typename T>
struct is_temporary_expr : cpp::or_c<is_temporary_unary_expr<T>, is_temporary_binary_expr<T>> {};

template <typename T>
struct is_transformer : cpp::bool_constant<decay_traits<T>::is_transformer> {};

template <typename T>
struct is_view : cpp::bool_constant<decay_traits<T>::is_view> {};

template <typename T>
struct is_magic_view : cpp::bool_constant<decay_traits<T>::is_magic_view> {};

template <typename T>
struct is_etl_expr : cpp::bool_constant<decay_traits<T>::is_etl> {};

template <typename T>
struct is_etl_direct_value : cpp::or_c<
                          is_fast_matrix<T>,
                          is_dyn_matrix<T>> {};

template <typename T>
struct is_etl_value : cpp::or_c<
                          is_fast_matrix<T>,
                          is_dyn_matrix<T>,
                          is_sparse_matrix<T>> {};

template <typename T>
struct is_direct_sub_view : std::false_type {};

template <typename T>
struct is_direct_sub_view<sub_view<T>> : cpp::and_u<has_direct_access<T>::value, decay_traits<T>::storage_order == order::RowMajor> {};

template <typename T>
struct is_direct_dim_view : std::false_type {};

template <typename T>
struct is_direct_dim_view<dim_view<T, 1>> : has_direct_access<T> {};

template <typename T>
struct is_direct_fast_matrix_view : std::false_type {};

template <typename T, std::size_t... Dims>
struct is_direct_fast_matrix_view<fast_matrix_view<T, Dims...>> : has_direct_access<T> {};

template <typename T>
struct is_direct_dyn_matrix_view : std::false_type {};

template <typename T>
struct is_direct_dyn_matrix_view<dyn_matrix_view<T>> : has_direct_access<T> {};

template <typename T>
struct is_direct_dyn_vector_view : std::false_type {};

template <typename T>
struct is_direct_dyn_vector_view<dyn_vector_view<T>> : has_direct_access<T> {};

template <typename T>
struct is_direct_identity_view : std::false_type {};

template <typename T, typename V>
struct is_direct_identity_view<etl::unary_expr<T, V, identity_op>> : has_direct_access<V> {};

template <typename T, typename DT>
struct has_direct_access : cpp::or_c<
                               is_etl_direct_value<DT>, is_temporary_unary_expr<DT>, is_temporary_binary_expr<DT>, is_direct_identity_view<DT>, is_direct_sub_view<DT>, is_direct_dim_view<DT>, is_direct_fast_matrix_view<DT>, is_direct_dyn_matrix_view<DT>, is_direct_dyn_vector_view<DT>> {};

template <typename T, typename Enable>
struct etl_traits {
    static constexpr const bool is_etl = false;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view = false;
    static constexpr const bool is_magic_view = false;
};

/*!
 * \brief Specialization for value structures
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_etl                 = true;
    static constexpr const bool is_transformer = false;
    static constexpr const bool is_view = false;
    static constexpr const bool is_magic_view = false;
    static constexpr const bool is_fast                 = is_fast_matrix<T>::value;
    static constexpr const bool is_value                = true;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = !is_sparse_matrix<T>::value;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = T::storage_order;

    static std::size_t size(const T& v) {
        return v.size();
    }

    static std::size_t dim(const T& v, std::size_t d) {
        return v.dim(d);
    }

    static constexpr std::size_t size() {
        return T::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return T::n_dimensions;
    }
};

//Warning: default template parameters for size and dim are already defined in traits_fwd.hpp

template <typename E, cpp_disable_if_fwd(etl_traits<E>::is_fast)>
std::size_t size(const E& v) {
    return etl_traits<E>::size(v);
}

template <typename E, cpp_enable_if_fwd(etl_traits<E>::is_fast)>
constexpr std::size_t size(const E& /*unused*/) noexcept {
    return etl_traits<E>::size();
}

template <typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t subsize(const E& v) {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size(v) / etl_traits<E>::dim(v, 0);
}

template <typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t subsize(const E& /*unused*/) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

template <std::size_t D, typename E, cpp_disable_if_fwd(etl_traits<E>::is_fast)>
std::size_t dim(const E& e) {
    return etl_traits<E>::dim(e, D);
}

template <typename E>
std::size_t dim(const E& e, std::size_t d) {
    return etl_traits<E>::dim(e, d);
}

template <std::size_t D, typename E, cpp_enable_if_fwd(etl_traits<E>::is_fast)>
constexpr std::size_t dim(const E& /*unused*/) noexcept {
    return etl_traits<E>::template dim<D>();
}

template <std::size_t D, typename E, cpp_enable_if_fwd(etl_traits<E>::is_fast)>
constexpr std::size_t dim() noexcept {
    return decay_traits<E>::template dim<D>();
}

template <typename E, typename Enable>
struct sub_size_compare;

template <typename E>
struct sub_size_compare<E, std::enable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, std::numeric_limits<std::size_t>::max()> {};

template <typename E>
struct sub_size_compare<E, cpp::disable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, etl_traits<E>::dimensions()> {};

template <typename E, cpp_enable_if(decay_traits<E>::storage_order == order::RowMajor)>
constexpr std::pair<std::size_t, std::size_t> index_to_2d(E&& sub, std::size_t i) {
    return std::make_pair(i / dim<0>(sub), i % dim<0>(sub));
}

template <typename E, cpp_enable_if(decay_traits<E>::storage_order == order::ColumnMajor)>
constexpr std::pair<std::size_t, std::size_t> index_to_2d(E&& sub, std::size_t i) {
    return std::make_pair(i % dim<0>(sub), i / dim<0>(sub));
}

template <typename E>
std::size_t row_stride(E&& e) {
    return decay_traits<E>::storage_order == order::RowMajor
               ? etl::dim<1>(e)
               : 1;
}

template <typename E>
std::size_t col_stride(E&& e) {
    return decay_traits<E>::storage_order == order::RowMajor
               ? 1
               : etl::dim<0>(e);
}

template <typename E>
std::size_t major_stride(E&& e) {
    return decay_traits<E>::storage_order == order::RowMajor
               ? etl::dim<1>(e)
               : etl::dim<0>(e);
}

} //end of namespace etl
