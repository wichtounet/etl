//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MMUL_EXPR_HPP
#define ETL_MMUL_EXPR_HPP

#include <algorithm>

#include "impl/mmul.hpp"

#include "traits_lite.hpp"
#include "temporary.hpp"

namespace etl {

namespace detail {

template<typename A, typename B, typename C, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mm_mul_sizes(const A& a, const B& b, C& c){
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
void check_mm_mul_sizes(const A&, const B&, C&){
    static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()          //interior dimensions
        &&  etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>()          //exterior dimension 1
        &&  etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<1>(),         //exterior dimension 2
        "Invalid sizes for multiplication");
}

template<typename A, typename B, typename C, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_vm_mul_sizes(const A& a, const B& b, C& c){
    cpp_assert(
            dim<0>(a) == dim<0>(b)          //exterior dimension 1
        &&  dim<1>(b) == dim<0>(c),         //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_vm_mul_sizes(const A&, const B&, C&){
    static_assert(
            etl_traits<A>::template dim<0>() == etl_traits<B>::template dim<0>()          //exterior dimension 1
        &&  etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<0>(),         //exterior dimension 2
        "Invalid sizes for multiplication");
}

template<typename A, typename B, typename C, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mv_mul_sizes(const A& a, const B& b, C& c){
    cpp_assert(
            dim<1>(a) == dim<0>(b)          //interior dimensions
        &&  dim<0>(a) == dim<0>(c)          //exterior dimension 1
        , "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mv_mul_sizes(const A&, const B&, C&){
    static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()          //interior dimensions
        &&  etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>()          //exterior dimension 1
        , "Invalid sizes for multiplication");
}

} //end of namespace detail

template<typename T, template<typename...> class Impl>
struct basic_mm_mul_expr {
    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_matrix<value_t<A>>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<decay_traits<A>::is_fast && decay_traits<B>::is_fast>> {
        using type = fast_dyn_matrix<typename std::decay_t<A>::value_type, decay_traits<A>::template dim<0>(), decay_traits<B>::template dim<1>()>;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&&, B&&){
        return new result_type<A, B>();
    }

    template<typename A, typename B, cpp_disable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&& a, B&& b){
        return new result_type<A, B>(etl::dim<0>(a), etl::dim<1>(b));
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
        detail::check_mm_mul_sizes(a,b,c);

        Impl<decltype(make_temporary(std::forward<A>(a))), decltype(make_temporary(std::forward<B>(b))), C,void>::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "mm_mul";
    }

    template<typename A, typename B>
    static std::size_t size(const A& a, const B& b){
        return etl::dim<0>(a) * etl::dim<1>(b);
    }

    template<typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d){
        if(d == 0){
            return etl::dim<0>(a);
        } else {
            return etl::dim<1>(b);
        }
    }

    template<typename A, typename B>
    static constexpr std::size_t size(){
        return etl_traits<A>::template dim<0>() * etl_traits<B>::template dim<1>();
    }

    template<typename A, typename B, std::size_t D>
    static constexpr std::size_t dim(){
        return D == 0 ? etl_traits<A>::template dim<0>() : etl_traits<B>::template dim<1>();
    }

    static constexpr std::size_t dimensions(){
        return 2;
    }
};

template<typename T>
using mm_mul_expr = basic_mm_mul_expr<T, detail::mm_mul_impl>;

template<typename T>
using strassen_mm_mul_expr = basic_mm_mul_expr<T, detail::strassen_mm_mul_impl>;

//Vector matrix multiplication

template<typename T, template<typename...> class Impl>
struct basic_vm_mul_expr {
    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_vector<value_t<A>>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<decay_traits<A>::is_fast && decay_traits<B>::is_fast>> {
        using type = fast_dyn_matrix<typename std::decay_t<A>::value_type, decay_traits<B>::template dim<1>()>;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&&, B&&){
        return new result_type<A, B>();
    }

    template<typename A, typename B, cpp_disable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&&, B&& b){
        return new result_type<A, B>(etl::dim<1>(b));
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Vector-Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 1 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 1, "Invalid dimensions for vecto-matrix multiplication");
        detail::check_vm_mul_sizes(a,b,c);

        Impl<decltype(make_temporary(std::forward<A>(a))), decltype(make_temporary(std::forward<B>(b))), C, void>::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "vm_mul";
    }

    template<typename A, typename B>
    static std::size_t size(const A&, const B& b){
        return etl::dim<1>(b);
    }

    template<typename A, typename B>
    static std::size_t dim(const A&, const B& b, std::size_t){
        return etl::dim<1>(b);
    }

    template<typename A, typename B>
    static constexpr std::size_t size(){
        return etl_traits<B>::template dim<1>();
    }

    template<typename A, typename B, std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<B>::template dim<1>();
    }

    static constexpr std::size_t dimensions(){
        return 1;
    }
};

template<typename T>
using vm_mul_expr = basic_vm_mul_expr<T, detail::vm_mul_impl>;

//Matrix Vector multiplication

template<typename T, template<typename...> class Impl>
struct basic_mv_mul_expr {
    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_vector<value_t<A>>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<decay_traits<A>::is_fast && decay_traits<B>::is_fast>> {
        using type = fast_dyn_matrix<typename std::decay_t<A>::value_type, decay_traits<A>::template dim<0>()>;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&&, B&&){
        return new result_type<A, B>();
    }

    template<typename A, typename B, cpp_disable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&& a, B&&){
        return new result_type<A, B>(etl::dim<0>(a));
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Vector-Matrix multiplication only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 1 && decay_traits<C>::dimensions() == 1, "Invalid dimensions for vecto-matrix multiplication");
        detail::check_mv_mul_sizes(a,b,c);

        Impl<decltype(make_temporary(std::forward<A>(a))), decltype(make_temporary(std::forward<B>(b))), C, void>::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "mv_mul";
    }

    template<typename A, typename B>
    static std::size_t size(const A& a, const B&){
        return etl::dim<0>(a);
    }

    template<typename A, typename B>
    static std::size_t dim(const A& a, const B&, std::size_t){
        return etl::dim<0>(a);
    }

    template<typename A, typename B>
    static constexpr std::size_t size(){
        return etl_traits<A>::template dim<0>();
    }

    template<typename A, typename B, std::size_t D>
    static constexpr std::size_t dim(){
        return etl_traits<A>::template dim<0>();
    }

    static constexpr std::size_t dimensions(){
        return 1;
    }
};

template<typename T>
using mv_mul_expr = basic_mv_mul_expr<T, detail::mv_mul_impl>;

} //end of namespace etl

#endif
