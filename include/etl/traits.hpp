//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/tmp.hpp"         //Some TMP stuff
#include "etl/traits_lite.hpp" //To avoid nasty errors

namespace etl {

template <typename T>
struct is_fast_matrix_impl : std::false_type {};

template <typename V1, typename V2, order V3, std::size_t... R>
struct is_fast_matrix_impl<fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

template <typename T, typename DT = std::decay_t<T>>
using is_fast_matrix              = is_fast_matrix_impl<DT>;

template <typename T>
struct is_dyn_matrix_impl : std::false_type {};

template <typename V1, order V2, std::size_t V3>
struct is_dyn_matrix_impl<dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T, typename DT>
struct is_dyn_matrix : is_dyn_matrix_impl<DT> {};

template <typename T>
struct is_sparse_matrix_impl : std::false_type {};

template <typename V1, sparse_storage V2, std::size_t V3>
struct is_sparse_matrix_impl<sparse_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T, typename DT>
struct is_sparse_matrix : is_sparse_matrix_impl<DT> {};

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

template <typename T, typename DT>
struct is_transformer : cpp::or_c<
                            cpp::is_specialization_of<etl::hflip_transformer, DT>,
                            cpp::is_specialization_of<etl::vflip_transformer, DT>,
                            cpp::is_specialization_of<etl::fflip_transformer, DT>,
                            cpp::is_specialization_of<etl::transpose_transformer, DT>,
                            cpp::is_specialization_of<etl::sum_r_transformer, DT>,
                            cpp::is_specialization_of<etl::sum_l_transformer, DT>,
                            cpp::is_specialization_of<etl::mean_r_transformer, DT>,
                            cpp::is_specialization_of<etl::mean_l_transformer, DT>,
                            cpp::is_specialization_of<etl::mm_mul_transformer, DT>,
                            cpp::is_specialization_of<etl::dyn_convmtx_transformer, DT>,
                            cpp::is_specialization_of<etl::dyn_convmtx2_transformer, DT>,
                            is_var<etl::rep_r_transformer, DT>,
                            is_var<etl::rep_l_transformer, DT>,
                            is_2<etl::dyn_rep_r_transformer, DT>,
                            is_2<etl::dyn_rep_l_transformer, DT>,
                            is_3<etl::p_max_pool_h_transformer, DT>,
                            is_3<etl::p_max_pool_p_transformer, DT>> {};

template <typename T, typename DT>
struct is_view : cpp::or_c<
                     is_2<etl::dim_view, DT>,
                     is_var<etl::fast_matrix_view, DT>,
                     cpp::is_specialization_of<etl::dyn_matrix_view, DT>,
                     cpp::is_specialization_of<etl::dyn_vector_view, DT>,
                     cpp::is_specialization_of<etl::sub_view, DT>> {};

template <typename T, typename DT>
struct is_magic_view : cpp::or_c<
                           cpp::is_specialization_of<etl::magic_view, DT>,
                           is_2<etl::fast_magic_view, DT>> {};

template <typename T>
struct is_etl_expr : cpp::or_c<
                         is_fast_matrix<T>,
                         is_dyn_matrix<T>,
                         is_sparse_matrix<T>,
                         is_unary_expr<T>,
                         is_binary_expr<T>,
                         is_temporary_unary_expr<T>,
                         is_temporary_binary_expr<T>,
                         is_generator_expr<T>,
                         is_transformer<T>, is_view<T>,
                         is_transformer<T>, is_magic_view<T>,
                         is_optimized_expr<T>> {};

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
struct etl_traits;

/*!
 * \brief Specialization for value structures
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_fast                 = is_fast_matrix<T>::value;
    static constexpr const bool is_value                = true;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = true;
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

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    using expr_t     = etl::unary_expr<T, Expr, UnaryOp>;
    using sub_expr_t = std::decay_t<Expr>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = etl_traits<sub_expr_t>::is_generator;
    static constexpr const bool vectorizable            = etl_traits<sub_expr_t>::vectorizable && UnaryOp::vectorizable;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.value());
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization generator_expr
 */
template <typename Generator>
struct etl_traits<etl::generator_expr<Generator>> {
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = true;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>> {
    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = true;
    static constexpr const bool vectorizable            = true;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;
};

/*!
 * \brief Specialization for binary_expr.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    using expr_t       = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>;
    using left_expr_t  = std::decay_t<LeftExpr>;
    using right_expr_t = std::decay_t<RightExpr>;

    static constexpr const bool left_directed = cpp::not_u<etl_traits<left_expr_t>::is_generator>::value;

    using sub_expr_t = std::conditional_t<left_directed, left_expr_t, right_expr_t>;

    static constexpr const bool is_fast  = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value = false;
    static constexpr const bool is_generator =
        etl_traits<left_expr_t>::is_generator && etl_traits<right_expr_t>::is_generator;
    static constexpr const bool vectorizable = etl_traits<left_expr_t>::vectorizable && etl_traits<right_expr_t>::vectorizable && BinaryOp::vectorizable;
    static constexpr const bool needs_temporary_visitor =
        etl_traits<left_expr_t>::needs_temporary_visitor || etl_traits<right_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor =
        etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order;

    template <bool B = left_directed, cpp_enable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.lhs();
    }

    template <bool B = left_directed, cpp_disable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.rhs();
    }

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(get(v));
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(get(v), d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for tranpose_transformer
 */
template <typename T>
struct etl_traits<transpose_transformer<T>> {
    using expr_t     = etl::transpose_transformer<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, 1 - d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<1 - D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for mm_mul_transformer
 */
template <typename LE, typename RE>
struct etl_traits<mm_mul_transformer<LE, RE>> {
    using expr_t       = etl::mm_mul_transformer<LE, RE>;
    using left_expr_t  = std::decay_t<LE>;
    using right_expr_t = std::decay_t<RE>;

    static constexpr const bool is_fast      = etl_traits<left_expr_t>::is_fast && etl_traits<right_expr_t>::is_fast;
    static constexpr const bool is_value     = false;
    static constexpr const bool is_generator = false;
    static constexpr const bool vectorizable = false;
    static constexpr const bool needs_temporary_visitor =
        etl_traits<left_expr_t>::needs_temporary_visitor || etl_traits<right_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor =
        etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return dim(v, 0) * dim(v, 1);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return etl::dim(v.left, 0);
        } else {
            cpp_assert(d == 1, "Only 2D mmul are supported");

            return etl::dim(v.right, 1);
        }
    }

    static constexpr std::size_t size() {
        return etl_traits<left_expr_t>::template dim<0>() * etl_traits<right_expr_t>::template dim<1>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(D < 2, "Only 2D mmul are supported");

        return D == 0 ? etl_traits<left_expr_t>::template dim<0>() : etl_traits<right_expr_t>::template dim<1>();
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx_transformer<E>> {
    using expr_t     = etl::dyn_convmtx_transformer<E>;
    using sub_expr_t = std::decay_t<E>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.h * (etl::size(v.sub) + v.h - 1);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return v.h;
        } else {
            return etl::size(v.sub) + v.h - 1;
        }
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_convmtx2_transformer
 */
template <typename E>
struct etl_traits<dyn_convmtx2_transformer<E>> {
    using expr_t     = etl::dyn_convmtx2_transformer<E>;
    using sub_expr_t = std::decay_t<E>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        auto c_height = (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        auto c_width  = v.k1 * v.k2;
        return c_height * c_width;
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return (etl::dim<0>(v.sub) + v.k1 - 1) * (etl::dim<1>(v.sub) + v.k2 - 1);
        } else {
            return v.k1 * v.k2;
        }
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for rep_r_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_r_transformer<T, D...>> {
    using expr_t     = etl::rep_r_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr const std::size_t sub_d = etl_traits<sub_expr_t>::dimensions();

    static std::size_t size(const expr_t& v) {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == 0 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0, 0, D...>::value;
    }

    static constexpr std::size_t size() {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D2, cpp_enable_if(D2 < sub_d)>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D2>();
    }

    template <std::size_t D2, cpp_disable_if(D2 < sub_d)>
    static constexpr std::size_t dim() {
        return nth_size<D2 - sub_d, 0, D...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for rep_l_transformer
 */
template <typename T, std::size_t... D>
struct etl_traits<rep_l_transformer<T, D...>> {
    using expr_t     = etl::rep_l_transformer<T, D...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        static_assert(sizeof...(D) == 1, "dim(d) is uninmplemented for rep<T, D1, D...>");
        return d == dimensions() - 1 ? etl_traits<sub_expr_t>::dim(v.sub, 0) : nth_size<0, 0, D...>::value;
    }

    static constexpr std::size_t size() {
        return mul_all<D...>::value * etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D2>
    static constexpr std::size_t dim() {
        return D2 >= sizeof...(D) ? etl_traits<sub_expr_t>::template dim<D2 - sizeof...(D)>() : nth_size<D2, 0, D...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(D) + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dyn_rep_r_transformer
 */
template <typename T, std::size_t D>
struct etl_traits<dyn_rep_r_transformer<T, D>> {
    using expr_t     = etl::dyn_rep_r_transformer<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr const std::size_t sub_d = etl_traits<sub_expr_t>::dimensions();

    static std::size_t size(const expr_t& v) {
        return v.m * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d < sub_d ? etl_traits<sub_expr_t>::dim(v.sub, d) : v.reps[d - sub_d];
    }

    static constexpr std::size_t dimensions() {
        return D + etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dyn_rep_l_transformer
 */
template <typename T, std::size_t D>
struct etl_traits<dyn_rep_l_transformer<T, D>> {
    using expr_t     = etl::dyn_rep_l_transformer<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.m * etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d >= D ? etl_traits<sub_expr_t>::dim(v.sub, d - D) : v.reps[d];
    }

    static constexpr std::size_t dimensions() {
        return D + etl_traits<sub_expr_t>::dimensions();
    }
};

//TODO Optimization when forced is not void

/*!
 * \brief Specialization for temporary_unary_expr.
 */
template <typename T, typename A, typename Op, typename Forced>
struct etl_traits<etl::temporary_unary_expr<T, A, Op, Forced>> {
    using expr_t = etl::temporary_unary_expr<T, A, Op, Forced>;
    using a_t    = std::decay_t<A>;

    static constexpr const bool is_fast                 = etl_traits<a_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = true;
    static constexpr const bool needs_temporary_visitor = true;
    static constexpr const bool needs_evaluator_visitor = true;
    static constexpr const order storage_order          = etl_traits<a_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return Op::size(v.a());
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return Op::dim(v.a(), d);
    }

    static constexpr std::size_t size() {
        return Op::template size<a_t>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return Op::template dim<a_t, D>();
    }

    static constexpr std::size_t dimensions() {
        return Op::dimensions();
    }
};

/*!
 * \brief Specialization for temporary_binary_expr.
 */
template <typename T, typename A, typename B, typename Op, typename Forced>
struct etl_traits<etl::temporary_binary_expr<T, A, B, Op, Forced>> {
    using expr_t = etl::temporary_binary_expr<T, A, B, Op, Forced>;
    using a_t    = std::decay_t<A>;
    using b_t    = std::decay_t<B>;

    static constexpr const bool is_fast                 = etl_traits<a_t>::is_fast && etl_traits<b_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = true;
    static constexpr const bool needs_temporary_visitor = true;
    static constexpr const bool needs_evaluator_visitor = true;
    static constexpr const order storage_order          = etl_traits<a_t>::is_generator ? etl_traits<b_t>::storage_order : etl_traits<a_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return Op::size(v.a(), v.b());
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return Op::dim(v.a(), v.b(), d);
    }

    static constexpr std::size_t size() {
        return Op::template size<a_t, b_t>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return Op::template dim<a_t, b_t, D>();
    }

    static constexpr std::size_t dimensions() {
        return Op::dimensions();
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::sum_r_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::mean_r_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t /*unused*/) {
        return etl::dim<0>(v.sub);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<0>();
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for (sum-mean)_r_transformer
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::sum_l_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::mean_l_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename std::decay_t<T>::sub_type>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl::size(v.sub) / etl::dim<0>(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl::dim(v.sub, d + 1);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

template <typename T, std::size_t C1, std::size_t C2>
struct etl_traits<p_max_pool_p_transformer<T, C1, C2>> {
    using expr_t     = p_max_pool_p_transformer<T, C1, C2>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub) / (C1 * C2);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == dimensions() - 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C2;
        } else if (d == dimensions() - 2) {
            return etl_traits<sub_expr_t>::dim(v.sub, d) / C1;
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, d);
        }
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / (C1 * C2);
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return D == dimensions() - 1 ? etl_traits<sub_expr_t>::template dim<D>() / C2
                                     : D == dimensions() - 2 ? etl_traits<sub_expr_t>::template dim<D>() / C1
                                                             : etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for flipping transformers
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<cpp::or_c<
                         cpp::is_specialization_of<etl::hflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::vflip_transformer, std::decay_t<T>>,
                         cpp::is_specialization_of<etl::fflip_transformer, std::decay_t<T>>,
                         is_3<etl::p_max_pool_h_transformer, std::decay_t<T>>>::value>> {
    using expr_t     = T;
    using sub_expr_t = std::decay_t<typename T::sub_type>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        if (D == 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_assert(d == 0, "Invalid dimension");
        cpp_unused(d);

        return size(v);
    }

    static constexpr std::size_t size() {
        return D == 1 ? etl_traits<sub_expr_t>::template dim<1>() : etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D2>
    static constexpr std::size_t dim() {
        static_assert(D2 == 0, "Invalid dimension");

        return size();
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for sub_view
 */
template <typename T>
struct etl_traits<etl::sub_view<T>> {
    using expr_t     = etl::sub_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;
    static constexpr const bool vectorizable            = has_direct_access<sub_expr_t>::value && storage_order == order::RowMajor;

    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.parent) / etl_traits<sub_expr_t>::dim(v.parent, 0);
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.parent, d + 1);
    }

    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

/*!
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, std::size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, Dims...>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static constexpr std::size_t size(const expr_t& /*unused*/) {
        return mul_all<Dims...>::value;
    }

    static std::size_t dim(const expr_t& /*unused*/, std::size_t d) {
        return dyn_nth_size<Dims...>(d);
    }

    static constexpr std::size_t size() {
        return mul_all<Dims...>::value;
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return nth_size<D, 0, Dims...>::value;
    }

    static constexpr std::size_t dimensions() {
        return sizeof...(Dims);
    }
};

/*!
 * \brief Specialization for dyn_matrix_view.
 */
template <typename T>
struct etl_traits<etl::dyn_matrix_view<T>> {
    using expr_t     = etl::dyn_matrix_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.rows * v.columns;
    }

    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d == 0 ? v.rows : v.columns;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Specialization for dyn_vector_view.
 */
template <typename T>
struct etl_traits<etl::dyn_vector_view<T>> {
    using expr_t     = etl::dyn_vector_view<T>;
    using sub_expr_t = std::decay_t<T>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;

    static std::size_t size(const expr_t& v) {
        return v.rows;
    }

    static std::size_t dim(const expr_t& v, std::size_t /*d*/) {
        return v.rows;
    }

    static constexpr std::size_t dimensions() {
        return 1;
    }
};

template <typename V>
struct etl_traits<etl::magic_view<V>> {
    using expr_t = etl::magic_view<V>;

    static constexpr const bool is_fast                 = false;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;

    static std::size_t size(const expr_t& v) {
        return v.n * v.n;
    }

    static std::size_t dim(const expr_t& v, std::size_t /*unused*/) {
        return v.n;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

template <std::size_t N, typename V>
struct etl_traits<etl::fast_magic_view<V, N>> {
    using expr_t = etl::fast_magic_view<V, N>;

    static constexpr const bool is_fast                 = true;
    static constexpr const bool is_value                = false;
    static constexpr const bool is_generator            = false;
    static constexpr const bool vectorizable            = false;
    static constexpr const bool needs_temporary_visitor = false;
    static constexpr const bool needs_evaluator_visitor = false;
    static constexpr const order storage_order          = order::RowMajor;

    static constexpr std::size_t size() {
        return N * N;
    }

    static std::size_t size(const expr_t& /*unused*/) {
        return N * N;
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        return N;
    }

    static std::size_t dim(const expr_t& /*e*/, std::size_t /*unused*/) {
        return N;
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Optimized expression simply use the same traits as its expression
template <typename Expr>
struct etl_traits<etl::optimized_expr<Expr>> : etl_traits<Expr> {};

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
