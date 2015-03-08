//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MULTIPLICATION_HPP
#define ETL_MULTIPLICATION_HPP

#include <algorithm>

#include "config.hpp"
#include "cblas.hpp"

namespace etl {

namespace detail {

template<typename A, typename B, typename C, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mmul_sizes(const A& a, const B& b, C& c){
    cpp_assert(
            dim<1>(a) == dim<0>(b)          //interior dimensions
        &&  dim<0>(a) == dim<0>(c)          //exterior dimension 1
        &&  dim<1>(b) == dim<1>(c),         //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mmul_sizes(const A&, const B&, C&){
    static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()          //interior dimensions
        &&  etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>()          //exterior dimension 1
        &&  etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<1>(),         //exterior dimension 2
        "Invalid sizes for multiplication");
}

template<typename A, typename B, typename C, typename Enable = void>
struct mmul_impl {
    static void apply(A&& a, B&& b, C&& c){
        c = 0;

        for(std::size_t i = 0; i < rows(a); i++){
            for(std::size_t k = 0; k < columns(a); k++){
                for(std::size_t j = 0; j < columns(b); j++){
                    c(i,j) += a(i,k) * b(k,j);
                }
            }
        }
    }
};

template<typename A, typename B, typename C>
struct is_blas_dgemm : cpp::bool_constant_c<cpp::and_c<
          is_cblas_enabled
        , is_double_precision<A>, is_double_precision<B>, is_double_precision<C>
        , has_direct_access<A>, has_direct_access<B>, has_direct_access<C>
    >> {};

template<typename A, typename B, typename C>
struct is_blas_sgemm : cpp::bool_constant_c<cpp::and_c<
          is_cblas_enabled
        , is_single_precision<A>, is_single_precision<B>, is_single_precision<C>
        , has_direct_access<A>, has_direct_access<B>, has_direct_access<C>
    >> {};

template<typename A, typename B, typename C>
struct mmul_impl<A, B, C, std::enable_if_t<is_blas_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        blas_sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

} //end of namespace detail

template<typename A, typename B, typename C>
C& mmul(A&& a, B&& b, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
    detail::check_mmul_sizes(a,b,c);

    detail::mmul_impl<A,B,C>::apply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));

    return c;
}

template<typename A, typename B, cpp::enable_if_all_u<decay_traits<A>::is_fast, decay_traits<B>::is_fast> = cpp::detail::dummy>
auto mmul(A&& a, B&& b){
    fast_dyn_matrix<typename std::decay_t<A>::value_type, decay_traits<A>::template dim<0>(), decay_traits<B>::template dim<1>()> c;

    mmul(a,b,c);

    return c;
}

template<typename A, typename B, cpp::disable_if_all_u<decay_traits<A>::is_fast, decay_traits<B>::is_fast> = cpp::detail::dummy>
auto mmul(A&& a, B&& b){
    dyn_matrix<value_t<A>> c(dim(a, 0), dim(b, 1));

    mmul(a,b,c);

    return c;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto lazy_mmul(LE&& lhs, RE&& rhs) -> stable_transform_binary_helper<LE, RE, mmul_transformer> {
    //TODO Check matrices sizes
    return {mmul_transformer<build_type<LE>, build_type<RE>>(lhs, rhs)};
}

inline std::size_t nextPowerOfTwo(std::size_t n) {
    return std::pow(2, static_cast<std::size_t>(std::ceil(std::log2(n))));
}

template<typename A, typename B, typename C>
void strassen_mmul_r(const A& a, const B& b, C& c){
    using type = typename A::value_type;

    auto n = dim<0>(a);

    //1x1 matrix mul
    if (n == 1) {
        c(0,0) = a(0,0) * b(0,0);
    } else if(n == 2){
        auto a11 = a(0,0);
        auto a12 = a(0,1);
        auto a21 = a(1,0);
        auto a22 = a(1,1);

        auto b11 = b(0,0);
        auto b12 = b(0,1);
        auto b21 = b(1,0);
        auto b22 = b(1,1);

        auto p1 = (a11 + a22) * (b11 + b22);
        auto p2 = (a12 - a22) * (b21 + b22);
        auto p3 = a11 * (b12 - b22);
        auto p4 = a22 * (b21 - b11);
        auto p5 = (a11 + a12) * b22;
        auto p6 = (a21 + a22) * b11;
        auto p7 = (a21 - a11) * (b11 + b12);

        c(0,0) = p1 + p4 + p2 - p5;
        c(0,1) = p3 + p5;
        c(1,0) = p6 + p4;
        c(1,1) = p1 + p3 + p7 - p6;
    } else if(n == 4){
        //This is entirely done on stack

        auto new_n = n / 2;

        etl::fast_matrix<type, 2, 2> a11;
        etl::fast_matrix<type, 2, 2> a12;
        etl::fast_matrix<type, 2, 2> a21;
        etl::fast_matrix<type, 2, 2> a22;

        etl::fast_matrix<type, 2, 2> b11;
        etl::fast_matrix<type, 2, 2> b12;
        etl::fast_matrix<type, 2, 2> b21;
        etl::fast_matrix<type, 2, 2> b22;

        etl::fast_matrix<type, 2, 2> p1;
        etl::fast_matrix<type, 2, 2> p2;
        etl::fast_matrix<type, 2, 2> p3;
        etl::fast_matrix<type, 2, 2> p4;
        etl::fast_matrix<type, 2, 2> p5;

        for (std::size_t i = 0; i < new_n; i++) {
            for (std::size_t j = 0; j < new_n; j++) {
                a11(i,j) = a(i,j);
                a12(i,j) = a(i,j + new_n);
                a21(i,j) = a(i + new_n,j);
                a22(i,j) = a(i + new_n,j + new_n);

                b11(i,j) = b(i,j);
                b12(i,j) = b(i,j + new_n);
                b21(i,j) = b(i + new_n,j);
                b22(i,j) = b(i + new_n,j + new_n);
            }
        }

        strassen_mmul_r(a11 + a22, b11 + b22, p1);
        strassen_mmul_r(a12 - a22, b21 + b22, p2);
        strassen_mmul_r(a22, b21 - b11, p4);
        strassen_mmul_r(a11 + a12, b22, p5);

        auto c11 = p1 + p4 + p2 - p5;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i,j) = c11(i,j);
            }
        }

        strassen_mmul_r(a11, b12 - b22, p3);

        auto c12 = p3 + p5;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i,j + new_n) = c12(i,j);
            }
        }

        strassen_mmul_r(a21 + a22, b11, p2);
        strassen_mmul_r(a21 - a11, b11 + b12, p5);

        auto c21 = p2 + p4;
        auto c22 = p1 + p3 + p5 - p2;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i + new_n,j) = c21(i,j);
                c(i + new_n,j + new_n) = c22(i,j);
            }
        }
    } else {
        auto new_n = n / 2;

        etl::dyn_matrix<type> a11(new_n, new_n);
        etl::dyn_matrix<type> a12(new_n, new_n);
        etl::dyn_matrix<type> a21(new_n, new_n);
        etl::dyn_matrix<type> a22(new_n, new_n);

        etl::dyn_matrix<type> b11(new_n, new_n);
        etl::dyn_matrix<type> b12(new_n, new_n);
        etl::dyn_matrix<type> b21(new_n, new_n);
        etl::dyn_matrix<type> b22(new_n, new_n);

        etl::dyn_matrix<type> p1(new_n, new_n);
        etl::dyn_matrix<type> p2(new_n, new_n);
        etl::dyn_matrix<type> p3(new_n, new_n);
        etl::dyn_matrix<type> p4(new_n, new_n);
        etl::dyn_matrix<type> p5(new_n, new_n);

        for (std::size_t i = 0; i < new_n; i++) {
            for (std::size_t j = 0; j < new_n; j++) {
                a11(i,j) = a(i,j);
                a12(i,j) = a(i,j + new_n);
                a21(i,j) = a(i + new_n,j);
                a22(i,j) = a(i + new_n,j + new_n);

                b11(i,j) = b(i,j);
                b12(i,j) = b(i,j + new_n);
                b21(i,j) = b(i + new_n,j);
                b22(i,j) = b(i + new_n,j + new_n);
            }
        }

        strassen_mmul_r(a11 + a22, b11 + b22, p1);
        strassen_mmul_r(a12 - a22, b21 + b22, p2);
        strassen_mmul_r(a22, b21 - b11, p4);
        strassen_mmul_r(a11 + a12, b22, p5);

        auto c11 = p1 + p4 + p2 - p5;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i,j) = c11(i,j);
            }
        }

        strassen_mmul_r(a11, b12 - b22, p3);

        auto c12 = p3 + p5;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i,j + new_n) = c12(i,j);
            }
        }

        strassen_mmul_r(a21 + a22, b11, p2);
        strassen_mmul_r(a21 - a11, b11 + b12, p5);

        auto c21 = p2 + p4;
        auto c22 = p1 + p3 + p5 - p2;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i + new_n,j) = c21(i,j);
                c(i + new_n,j + new_n) = c22(i,j);
            }
        }
    }
}

template<typename A, typename B, typename C>
C& strassen_mmul(const A& a, const B& b, C& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(etl_traits<A>::dimensions() == 2 && etl_traits<B>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
    detail::check_mmul_sizes(a,b,c);

    c = 0;

    //For now, assume matrices are of size 2^nx2^n

    auto n = std::max(dim<0>(a), std::max(dim<1>(a), dim<1>(b)));
    auto m = nextPowerOfTwo(n);

    if(dim<0>(a) == m && dim<0>(b) == m && dim<1>(a) == m && dim<1>(b) == m){
        strassen_mmul_r(a, b, c);
    } else {
        using type = typename A::value_type;

        etl::dyn_matrix<type> a_prep(m, m, static_cast<type>(0));
        etl::dyn_matrix<type> b_prep(m, m, static_cast<type>(0));
        etl::dyn_matrix<type> c_prep(m, m, static_cast<type>(0));

        for(std::size_t i=0; i< dim<0>(a); i++) {
            for (std::size_t j=0; j<dim<1>(a); j++) {
                a_prep(i,j) = a(i,j);
            }
        }

        for(std::size_t i=0; i< dim<0>(b); i++) {
            for (std::size_t j=0; j<dim<1>(b); j++) {
                b_prep(i,j) = b(i,j);
            }
        }

        strassen_mmul_r(a_prep, b_prep, c_prep);

        for(std::size_t i=0; i< dim<0>(c); i++) {
            for (std::size_t j=0; j<dim<1>(c); j++) {
                c(i,j) = c_prep(i,j);
            }
        }
    }

    return c;
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::is_fast, decay_traits<B>::is_fast, decay_traits<C>::is_fast,
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(reshape<1, decay_traits<B>::template dim<0>()>(a), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::is_fast, decay_traits<B>::is_fast, decay_traits<C>::is_fast,
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(a, reshape<decay_traits<A>::template dim<1>(),1>(b), c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!decay_traits<A>::is_fast, !decay_traits<B>::is_fast, !decay_traits<C>::is_fast>::value,
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2,
    cpp::not_u<decay_traits<A>::is_fast>::value
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(reshape(a, 1, dim<0>(b)), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!decay_traits<A>::is_fast, !decay_traits<B>::is_fast, !decay_traits<C>::is_fast>::value,
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1,
    cpp::not_u<decay_traits<B>::is_fast>::value
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(a, reshape(b, dim<1>(a), 1), c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!decay_traits<A>::is_fast, !decay_traits<B>::is_fast, !decay_traits<C>::is_fast>::value,
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2,
    decay_traits<A>::is_fast
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(reshape<1, decay_traits<B>::template dim<0>()>(a), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!decay_traits<A>::is_fast, !decay_traits<B>::is_fast, !decay_traits<C>::is_fast>::value,
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1,
    decay_traits<B>::is_fast
> = cpp::detail::dummy>
C& auto_vmmul(A&& a, B&& b, C& c){
    return mmul(a, reshape<decay_traits<A>::template dim<1>(),1>(b), c);
}

} //end of namespace etl

#endif
