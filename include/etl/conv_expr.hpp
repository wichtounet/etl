//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONVOLUTION_EXPR_HPP
#define ETL_CONVOLUTION_EXPR_HPP

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "traits_lite.hpp"

//Get the implementations
#include "impl/conv.hpp"

namespace etl {

enum class conv_type {
    VALID,
    SAME,
    FULL
};

namespace detail {

template<conv_type TT, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    if(TT == conv_type::FULL){
        cpp_assert(dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1, "Invalid sizes for 'full' convolution");
    } else if(TT == conv_type::SAME){
        cpp_assert(dim<0>(conv) == dim<0>(input), "Invalid sizes for 'same' convolution");
    } else if(TT == conv_type::VALID){
        cpp_assert(dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1, "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<conv_type TT, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I&, const K&, const C&){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    static_assert(
        TT != conv_type::FULL || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1,
        "Invalid sizes for 'full'convolution");
    static_assert(
        TT != conv_type::SAME || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>(),
        "Invalid sizes for 'same'convolution");
    static_assert(
        TT != conv_type::VALID || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1,
        "Invalid sizes for 'valid'convolution");
}

template<conv_type TT, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    if(TT == conv_type::FULL){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1
            &&  dim<1>(conv) == dim<1>(input) + dim<1>(kernel) - 1,
            "Invalid sizes for 'full' convolution");
    } else if(TT == conv_type::SAME){
        cpp_assert(dim<0>(conv) == dim<0>(input) && dim<1>(conv) == dim<1>(input), "Invalid sizes for 'same' convolution");
    } else if(TT == conv_type::VALID){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1
            &&  dim<1>(conv) == dim<1>(input) - dim<1>(kernel) + 1,
            "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<conv_type TT, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I&, const K&, const C&){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    static_assert(
        TT != conv_type::FULL || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() + etl_traits<K>::template dim<1>() - 1),
        "Invalid sizes for 'full'convolution");
    static_assert(
        TT != conv_type::SAME || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>()
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>()),
        "Invalid sizes for 'same'convolution");
    static_assert(
        TT != conv_type::VALID || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() - etl_traits<K>::template dim<1>() + 1),
        "Invalid sizes for 'valid'convolution");
}

} //end of namespace detail

//D1 = 1D
template<typename T, bool D1, conv_type TT, template<typename...> class Impl>
struct basic_conv1_expr {
    using this_type = basic_conv1_expr<T, D1, TT, Impl>;

    template<typename A, typename B, std::size_t D>
    static constexpr std::size_t dim(){
        if(TT == conv_type::VALID){
            return decay_traits<A>::template dim<D>() - decay_traits<B>::template dim<D>() + 1;
        } else if(TT == conv_type::SAME){
            return decay_traits<A>::template dim<D>();
        } else {
            return decay_traits<A>::template dim<D>() + decay_traits<B>::template dim<D>() - 1;
        }
    }

    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_vector<value_t<A>>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<decay_traits<A>::is_fast && decay_traits<B>::is_fast>> {
        using type = fast_dyn_matrix<typename std::decay_t<A>::value_type, this_type::template dim<A,B,0>()>;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&&, B&&){
        return new result_type<A, B>();
    }

    template<typename A, typename B, cpp_disable_if(decay_traits<A>::is_fast && decay_traits<B>::is_fast)>
    static result_type<A,B>* allocate(A&& a, B&& b){
        return new result_type<A, B>(this_type::dim(a, b, 0));
    }

    template<typename A, typename B, typename C, cpp_enable_if_cst(D1)>
    static void check(const A& a, const B& b, const C& c){
        detail::check_conv_1d_sizes<TT>(a, b, c);
    }
    
    template<typename A, typename B, typename C, cpp_disable_if_cst(D1)>
    static void check(const A& a, const B& b, const C& c){
        detail::check_conv_1d_sizes<TT>(a, b, c);
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

        check(a, b, c);

        Impl<A,B,C>::apply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }

    static std::string desc() noexcept {
        if(TT == conv_type::VALID){
            return "conv_valid";
        } else if(TT == conv_type::SAME){
            return "conv_same";
        } else {
            return "conv_full";
        }
    }

    template<typename A, typename B>
    static std::size_t size(const A& a, const B& b){
        if(D1){
            return this_type::dim(a, b, 0);
        } else {
            return this_type::dim(a, b, 0) * this_type::dim(a, b, 1);
        }
    }

    template<typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d){
        if(TT == conv_type::VALID){
            return etl_traits<A>::dim(a, d) - etl_traits<B>::dim(b, d) + 1;
        } else if(TT == conv_type::SAME){
            return etl_traits<A>::dim(a, d);
        } else {
            return etl_traits<A>::dim(a, d) + etl_traits<B>::dim(b, d) + 1;
        }
    }

    template<typename A, typename B, cpp_enable_if_cst(D1)>
    static constexpr std::size_t size(){
        return this_type::dim<A, B, 0>();
    }

    template<typename A, typename B, cpp_disable_if_cst(D1)>
    static constexpr std::size_t size(){
        return this_type::dim<A, B, 0>() * this_type::dim<A, B, 1>();
    }

    static constexpr std::size_t dimensions(){
        return D1 ? 1 : 2;
    }
};

template<typename T>
using conv1_valid_expr = basic_conv1_expr<T, true, conv_type::VALID, detail::conv1_valid_impl>;

template<typename T>
using conv1_same_expr = basic_conv1_expr<T, true, conv_type::SAME, detail::conv1_same_impl>;

template<typename T>
using conv1_full_expr = basic_conv1_expr<T, true, conv_type::FULL, detail::conv1_full_impl>;

//2D convolutions

template<typename I, typename K, typename C>
C& convolve_2d_full(const I& input, const K& kernel, C&& conv){

    detail::conv2_full_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_2d_same(const I& input, const K& kernel, C&& conv){

    detail::conv2_same_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_2d_valid(const I& input, const K& kernel, C&& conv){

    detail::conv2_valid_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

//Deep convolutions

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_full(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_full(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_same(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_same(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_valid(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_valid(input(i), kernel(i), conv(i));
    }

    return conv;
}

} //end of namespace etl

#endif
