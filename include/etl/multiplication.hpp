//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MULTIPLICATION_HPP
#define ETL_MULTIPLICATION_HPP

#include <algorithm>

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

} //end of namespace detail

template<typename A, typename B, typename C>
static C& mmul(const A& a, const B& b, C& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(etl_traits<A>::dimensions() == 2 && etl_traits<B>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
    detail::check_mmul_sizes(a,b,c);

    c = 0;

    for(std::size_t i = 0; i < rows(a); i++){
        for(std::size_t j = 0; j < columns(b); j++){
            for(std::size_t k = 0; k < columns(a); k++){
                c(i,j) += a(i,k) * b(k,j);
            }
        }
    }

    return c;
}

inline std::size_t nextPowerOfTwo(std::size_t n) {
    return std::pow(2, static_cast<std::size_t>(std::ceil(std::log2(n))));
}

template<typename A, typename B, typename C>
static void strassen_mmul_r(const A& a, const B& b, C& c){
    using type = typename A::value_type;

    auto n = dim<0>(a);

    //1x1 matrix mul
    if (n == 1) {
        c(0,0) = a(0,0) * b(0,0);
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

        etl::dyn_matrix<type> c11(new_n, new_n);
        etl::dyn_matrix<type> c12(new_n, new_n);
        etl::dyn_matrix<type> c21(new_n, new_n);
        etl::dyn_matrix<type> c22(new_n, new_n);
 
        etl::dyn_matrix<type> p1(new_n, new_n);
        etl::dyn_matrix<type> p2(new_n, new_n);
        etl::dyn_matrix<type> p3(new_n, new_n);
        etl::dyn_matrix<type> p4(new_n, new_n);
        etl::dyn_matrix<type> p5(new_n, new_n);
        etl::dyn_matrix<type> p6(new_n, new_n);
        etl::dyn_matrix<type> p7(new_n, new_n);

        etl::dyn_matrix<type> a_result(new_n, new_n);
        etl::dyn_matrix<type> b_result(new_n, new_n);

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

        a_result = a11 + a22;
        b_result = b11 + b22;
        strassen_mmul_r(a_result, b_result, p1);

        a_result = a21 + a22;
        strassen_mmul_r(a_result, b11, p2);

        b_result = b12 - b22;
        strassen_mmul_r(a11, b_result, p3);

        b_result = b21 - b11;
        strassen_mmul_r(a22, b_result, p4);

        a_result = a11 + a12;
        strassen_mmul_r(a_result, b22, p5);

        a_result = a21 - a11;
        b_result = b11 + b12;
        strassen_mmul_r(a_result, b_result, p6);

        a_result = a12 - a22;
        b_result = b21 + b22;
        strassen_mmul_r(a_result, b_result, p7);

        c12 = p3 + p5;
        c11 = p1 + p4 + p7 - p5;
        c21 = p2 + p4;
        c22 = p1 + p3 + p6 - p2;

        for (std::size_t i = 0; i < new_n ; i++) {
            for (std::size_t j = 0 ; j < new_n ; j++) {
                c(i,j) = c11(i,j);
                c(i,j + new_n) = c12(i,j);
                c(i + new_n,j) = c21(i,j);
                c(i + new_n,j + new_n) = c22(i,j);
            }
        }
    }
}

template<typename A, typename B, typename C>
static C& strassen_mmul(const A& a, const B& b, C& c){
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
    etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast,
    etl_traits<A>::dimensions() == 1, etl_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(reshape<1, etl_traits<B>::template dim<0>()>(a), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast,
    etl_traits<A>::dimensions() == 2, etl_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(a, reshape<etl_traits<A>::template dim<1>(),1>(b), c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!etl_traits<A>::is_fast, !etl_traits<B>::is_fast, !etl_traits<C>::is_fast>::value,
    etl_traits<A>::dimensions() == 1, etl_traits<B>::dimensions() == 2,
    cpp::not_u<etl_traits<A>::is_fast>::value
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(reshape(a, 1, dim<0>(b)), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!etl_traits<A>::is_fast, !etl_traits<B>::is_fast, !etl_traits<C>::is_fast>::value,
    etl_traits<A>::dimensions() == 2, etl_traits<B>::dimensions() == 1,
    cpp::not_u<etl_traits<B>::is_fast>::value
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(a, reshape(b, dim<1>(a), 1), c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!etl_traits<A>::is_fast, !etl_traits<B>::is_fast, !etl_traits<C>::is_fast>::value,
    etl_traits<A>::dimensions() == 1, etl_traits<B>::dimensions() == 2,
    etl_traits<A>::is_fast
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(reshape<1, etl_traits<B>::template dim<0>()>(a), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    cpp::or_u<!etl_traits<A>::is_fast, !etl_traits<B>::is_fast, !etl_traits<C>::is_fast>::value,
    etl_traits<A>::dimensions() == 2, etl_traits<B>::dimensions() == 1,
    etl_traits<B>::is_fast
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(a, reshape<etl_traits<A>::template dim<1>(),1>(b), c);
}

} //end of namespace etl

#endif
