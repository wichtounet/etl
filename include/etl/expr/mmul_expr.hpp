//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/mmul.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Assert for the validity of the matrix-matrix multiplication operation
 * \param a The left side matrix
 * \param b The right side matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_disable_if(all_fast<A, B, C>::value)>
void check_mm_mul_sizes(const A& a, const B& b, C& c) {
    cpp_assert(
        dim<1>(a) == dim<0>(b)         //interior dimensions
            && dim<0>(a) == dim<0>(c)  //exterior dimension 1
            && dim<1>(b) == dim<1>(c), //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

/*!
 * \brief Assert for the validity of the matrix-matrix multiplication operation
 * \param a The left side matrix
 * \param b The right side matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_enable_if(all_fast<A, B, C>::value)>
void check_mm_mul_sizes(const A& a, const B& b, C& c) {
    static_assert(
        etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()         //interior dimensions
            && etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>()  //exterior dimension 1
            && etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<1>(), //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

/*!
 * \brief Assert for the validity of the vector-matrix multiplication operation
 * \param a The left side vector
 * \param b The right side matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_disable_if(all_fast<A, B, C>::value)>
void check_vm_mul_sizes(const A& a, const B& b, C& c) {
    cpp_assert(
        dim<0>(a) == dim<0>(b)         //exterior dimension 1
            && dim<1>(b) == dim<0>(c), //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

/*!
 * \brief Assert for the validity of the vector-matrix multiplication operation
 * \param a The left side vector
 * \param b The right side matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_enable_if(all_fast<A, B, C>::value)>
void check_vm_mul_sizes(const A& a, const B& b, C& c) {
    static_assert(
        etl_traits<A>::template dim<0>() == etl_traits<B>::template dim<0>()         //exterior dimension 1
            && etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<0>(), //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

/*!
 * \brief Assert for the validity of the matrix-vector multiplication operation
 * \param a The left side matrix
 * \param b The right side vector
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_disable_if(all_fast<A, B, C>::value)>
void check_mv_mul_sizes(const A& a, const B& b, C& c) {
    cpp_assert(
        dim<1>(a) == dim<0>(b)        //interior dimensions
            && dim<0>(a) == dim<0>(c) //exterior dimension 1
        ,
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

/*!
 * \brief Assert for the validity of the matrix-vector multiplication operation
 * \param a The left side matrix
 * \param b The right side vector
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_enable_if(all_fast<A, B, C>::value)>
void check_mv_mul_sizes(const A& a, const B& b, C& c) {
    static_assert(
        etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()        //interior dimensions
            && etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>() //exterior dimension 1
        ,
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

} //end of namespace detail

/*!
 * \brief A basic matrix-matrix multiplication expression
 * \tparam T The value type
 * \tparam Impl The implementation class
 */
template <typename T, typename Impl>
struct basic_mm_mul_expr : impl_expr<basic_mm_mul_expr<T, Impl>, T> {
    using value_type = T; ///< The value type
    using this_type  = basic_mm_mul_expr<T, Impl>; ///< This expression type

    /*!
     * \brief The result type for given sub types
     * \tparam A The lhs expression type
     * \tparam B The rhs expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

    static constexpr bool is_gpu = cublas_enabled; ///< Indicate if this expressions may run on GPU

    /*!
     * \brief Apply the expression
     * \param a The lhs expression
     * \param b The rhs expression
     * \param c The expression where to store the results
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
        detail::check_mm_mul_sizes(a, b, c);

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "mm_mul";
    }

    /*!
     * \brief Returns the size of the expression given a and b
     * \return the size of the expression
     */
    template <typename A, typename B>
    static std::size_t size(const A& a, const B& b) {
        return etl::dim<0>(a) * etl::dim<1>(b);
    }

    /*!
     * \brief Returns the dth dimension of the expression given a and b
     * \param a The left hand side
     * \param b The right hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
        if (d == 0) {
            return etl::dim<0>(a);
        } else {
            return etl::dim<1>(b);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return etl_traits<A>::template dim<0>() * etl_traits<B>::template dim<1>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return decay_traits<A>::storage_order;
    }

    /*!
     * \brief Returns the Dth dimension of the expression
     * \return the Dth dimension of the expression
     */
    template <typename A, typename B, std::size_t D>
    static constexpr std::size_t dim() {
        return D == 0 ? decay_traits<A>::template dim<0>() : decay_traits<B>::template dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Expression for matrix-matrix multiplication
 */
template <typename T>
using mm_mul_expr = basic_mm_mul_expr<T, detail::mm_mul_impl>;

/*!
 * \brief Expression for Strassen matrix-matrix multiplication
 */
template <typename T>
using strassen_mm_mul_expr = basic_mm_mul_expr<T, detail::strassen_mm_mul_impl>;

/*!
 * \brief A basic vector-matrix multiplication expression
 * \tparam T The value type
 * \tparam Impl The implementation class
 */
template <typename T, typename Impl>
struct basic_vm_mul_expr : impl_expr<basic_vm_mul_expr<T, Impl>, T> {
    using value_type = T; ///< The value type
    using this_type  = basic_vm_mul_expr<T, Impl>; ///< This expression type

    static constexpr bool is_gpu = cublas_enabled; ///< Indicate if this expressions may run on GPU

    /*!
     * \brief The result type for given sub types
     * \tparam A The lhs expression type
     * \tparam B The rhs expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

    /*!
     * \brief Apply the expression
     * \param a The lhs expression
     * \param b The rhs expression
     * \param c The expression where to store the results
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "Vector-Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 1 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 1, "Invalid dimensions for vecto-matrix multiplication");
        detail::check_vm_mul_sizes(a, b, c);

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "vm_mul";
    }

    /*!
     * \brief Returns the size of the expression given a and b
     * \param a The left hand side
     * \param b The right hand side
     * \return the size of the expression
     */
    template <typename A, typename B>
    static std::size_t size(const A& a, const B& b) {
        cpp_unused(a);
        return etl::dim<1>(b);
    }

    /*!
     * \brief Returns the dth dimension of the expression given a and b
     * \param a The left hand side
     * \param b The right hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
        cpp_unused(a);
        cpp_unused(d);
        return etl::dim<1>(b);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return etl_traits<B>::template dim<1>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Returns the Dth dimension of the expression
     * \return the Dth dimension of the expression
     */
    template <typename A, typename B, std::size_t D>
    static constexpr std::size_t dim() {
        return decay_traits<B>::template dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Expression for vector-matrix multiplication
 */
template <typename T>
using vm_mul_expr = basic_vm_mul_expr<T, detail::vm_mul_impl>;

/*!
 * \brief A basic matrix-vector multiplication expression
 * \tparam T The value type
 * \tparam Impl The implementation class
 */
template <typename T, typename Impl>
struct basic_mv_mul_expr : impl_expr<basic_mv_mul_expr<T, Impl>, T> {
    using value_type = T;                          ///< The value type
    using this_type  = basic_mv_mul_expr<T, Impl>; ///< This expression type

    static constexpr bool is_gpu = cublas_enabled; ///< Indicate if this expressions may run on GPU

    /*!
     * \brief The result type for given sub types
     * \tparam A The lhs expression type
     * \tparam B The rhs expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

    /*!
     * \brief Apply the expression
     * \param a The lhs expression
     * \param b The rhs expression
     * \param c The expression where to store the results
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "Vector-Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 1 && decay_traits<C>::dimensions() == 1, "Invalid dimensions for vector-matrix multiplication");
        detail::check_mv_mul_sizes(a, b, c);

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "mv_mul";
    }

    /*!
     * \brief Returns the size of the expression given a and b
     * \return the size of the expression
     */
    template <typename A, typename B>
    static std::size_t size(const A& a, const B& /*b*/) {
        return etl::dim<0>(a);
    }

    /*!
     * \brief Returns the dth dimension of the expression given a and b
     * \param a The left hand side
     * \param b The right hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
        cpp_unused(b);
        cpp_unused(d);
        return etl::dim<0>(a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return etl_traits<A>::template dim<0>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Returns the Dth dimension of the expression
     * \return the Dth dimension of the expression
     */
    template <typename A, typename B, std::size_t D>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<0>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Expression for matrix-vector multiplication
 */
template <typename T>
using mv_mul_expr = basic_mv_mul_expr<T, detail::mv_mul_impl>;

} //end of namespace etl
