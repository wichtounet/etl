//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FFT_EXPR_HPP
#define ETL_FFT_EXPR_HPP

#include <complex>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "traits_lite.hpp"

//Get the implementations
#include "impl/fft.hpp"

namespace etl {

template<typename T, std::size_t D, template<typename...> class Impl>
struct basic_fft_expr {
    template<typename A, class Enable = void>
    struct result_type_builder {
        using type = dyn_vector<std::complex<value_t<A>>>;
    };

    template<typename A>
    struct result_type_builder<A, std::enable_if_t<decay_traits<A>::is_fast>> {
        using type = fast_dyn_matrix<std::complex<value_t<A>>, etl::template dim<0,A>()>;
    };

    template<typename A>
    using result_type = typename result_type_builder<A>::type;

    template<typename A, cpp_enable_if(decay_traits<A>::is_fast)>
    static result_type<A>* allocate(A&&){
        return new result_type<A>();
    }

    template<typename A, cpp_disable_if(decay_traits<A>::is_fast)>
    static result_type<A>* allocate(A&& a){
        return new result_type<A>(etl::template dim<0>(a));
    }

    template<typename A, typename C>
    static void apply(A&& a, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "Fast-Fourrier Transform only supported for ETL expressions");

        Impl<decltype(make_temporary(std::forward<A>(a))), C, void>::apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "fft";
    }

    template<typename A>
    static std::size_t dim(const A& a, std::size_t d){
        return etl_traits<A>::dim(a, d);
    }

    template<typename A>
    static std::size_t size(const A& a){
        return etl::size(a);
    }

    template<typename A, std::size_t DD>
    static constexpr std::size_t dim(){
        return decay_traits<A>::template dim<DD>();
    }

    template<typename A>
    static constexpr std::size_t size(){
        return etl::decay_traits<A>::size();
    }

    static constexpr std::size_t dimensions(){
        return D;
    }
};

//1D FFT

template<typename T>
using fft1_expr = basic_fft_expr<T, 1, detail::fft_impl>;

} //end of namespace etl

#endif
